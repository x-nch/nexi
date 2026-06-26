"""Step 3 — Intent Normalization (Contract 4). LiteLLM-backed classifier."""
import hashlib
import json
import re
from typing import Any
from uuid import UUID

import httpx
from agentmemory import create_memory, search_memory

from ..config import settings
from ..models.intent import Intent, IntentClass, ActionType, Urgency
from ..utils.audit import emit_event
from xnch.security.injection_guard import scan_input, InjectionResult


class PolicyViolation(Exception):
    pass


_CLASSIFICATION_SYSTEM_PROMPT = """You are an intent classifier for a systems-management agent.
Classify the user's raw input into one of these intent classes:

- QUERY: Read-only information retrieval (list, show, get, fetch, display)
- DECISION: Planning, design, recommendation, analysis for decision support
- EXECUTION: Mutating actions (deploy, rollback, write, delete, backup, run)
- ESCALATION: Escalate to human operator

Respond with a JSON object using this schema:
{
  "intent_class": "QUERY|DECISION|EXECUTION|ESCALATION",
  "action_type": "<appropriate action from the list below>",
  "entity_class": "<guessed entity type: ML_MODEL|SERVICE|DATABASE|CLUSTER|FILE|SCRIPT|SCHEMA|RESOURCE>",
  "urgency": "LOW|NORMAL|HIGH|CRITICAL",
  "entity_id": "<the noun phrase identifying the target>",
  "clarifications_needed": ["<question if ambiguous>"]
}

Action types by class:
QUERY → READ_FILE, LIST, ANALYZE, QUERY
DECISION → PLAN, ANALYZE
EXECUTION → DEPLOY, ROLLBACK, BACKUP, RESTORE, WRITE_FILE, DELETE_FILE, RUN_COMMAND, RUN_SCRIPT, STAGE, MUTATE
ESCALATION → ESCALATE

If the input is ambiguous (multiple interpretations possible), populates clarifications_needed
with specific questions to disambiguate. Otherwise leave it empty."""


class ClarificationRequired(Exception):
    def __init__(self, session_id: UUID, ambiguity_score: float, questions: list[str] | None = None) -> None:
        self.session_id = session_id
        self.ambiguity_score = ambiguity_score
        self.questions = questions or []
        msg = f"Ambiguity score {ambiguity_score:.2f} exceeds threshold"
        if self.questions:
            msg += f"; questions: {'; '.join(self.questions)}"
        super().__init__(msg)


# Stage 1: rule-based pre-filter (confidence=1.0, ambiguity=0.0)
_RULES: list[tuple[re.Pattern, IntentClass, str]] = [
    (re.compile(r"^(list|show all|enumerate)\b", re.I), IntentClass.QUERY, "LIST"),
    (re.compile(r"^(show|read|get|fetch|display)\b", re.I), IntentClass.QUERY, "READ_FILE"),
    (re.compile(r"^(analyze|inspect|check|scan)\b", re.I), IntentClass.QUERY, "ANALYZE"),
    (re.compile(r"^(deploy|launch|start|spin up)\b", re.I), IntentClass.EXECUTION, "DEPLOY"),
    (re.compile(r"^(rollback|revert)\b", re.I), IntentClass.EXECUTION, "ROLLBACK"),
    (re.compile(r"^back(up|fill)\b", re.I), IntentClass.EXECUTION, "BACKUP"),
    (re.compile(r"^(delete|remove|drop)\b", re.I), IntentClass.EXECUTION, "DELETE_FILE"),
    (re.compile(r"^(write|create|add)\b", re.I), IntentClass.EXECUTION, "WRITE_FILE"),
    (re.compile(r"^(plan|design|propose)\b", re.I), IntentClass.DECISION, "PLAN"),
    (re.compile(r"^(recommend|suggest|advise)\b", re.I), IntentClass.DECISION, "ANALYZE"),
    (re.compile(r"^escalate\b", re.I), IntentClass.ESCALATION, "ESCALATE"),
]


def _rule_classify(raw_input: str) -> tuple[IntentClass, str, float, float] | None:
    for pattern, intent_class, action_type in _RULES:
        if pattern.search(raw_input.strip()):
            return intent_class, action_type, 1.0, 0.0
    return None


def _extract_entity(raw_input: str) -> tuple[str, str]:
    tokens = raw_input.strip().split()
    entity_id = " ".join(tokens[1:]) if len(tokens) > 1 else "unknown"
    entity_class_hints = {
        "model": "ML_MODEL", "service": "SERVICE", "db": "DATABASE",
        "database": "DATABASE", "cluster": "CLUSTER", "file": "FILE",
        "script": "SCRIPT", "schema": "SCHEMA",
    }
    for token in tokens:
        for hint, cls in entity_class_hints.items():
            if hint in token.lower():
                return entity_id, cls
    return entity_id, "RESOURCE"


def _recall_intent(raw_input: str) -> Intent | None:
    try:
        results = search_memory("intent-classifications", raw_input, n_results=3, include_embeddings=False)
        for item in results:
            mem = item.get("document", "") if isinstance(item, dict) else str(item)
            if isinstance(mem, str):
                try:
                    data = json.loads(mem)
                    if data.get("raw_input", "").lower().strip() == raw_input.lower().strip():
                        return Intent(
                            session_id=UUID(int=0),
                            intent_class=IntentClass(data["intent_class"]),
                            action_type=ActionType(data["action_type"]),
                            target_entity_id=data.get("entity_id", "unknown"),
                            target_entity_class=data.get("entity_class", "RESOURCE"),
                            urgency=Urgency(data.get("urgency", "NORMAL")),
                            ambiguity_score=0.0,
                            raw_input_hash="",
                            raw_input=raw_input,
                        )
                except Exception:
                    pass
    except Exception:
        pass
    return None


def _persist_intent(raw_input: str, intent: Intent) -> None:
    try:
        create_memory(
            "intent-classifications",
            json.dumps({
                "raw_input": raw_input,
                "intent_class": intent.intent_class,
                "action_type": intent.action_type,
                "entity_id": intent.target_entity_id,
                "entity_class": intent.target_entity_class,
                "urgency": intent.urgency,
            }),
            metadata={
                "intent_class": intent.intent_class,
                "action_type": intent.action_type,
            },
        )
    except Exception:
        pass


class IntentInterpreter:
    """Contract 4 two-stage classifier: rule-based pre-filter + LiteLLM fallback."""

    async def interpret(self, raw_input: str, session_id: UUID, trace_id: str) -> Intent:
        raw_input_hash = "sha256:" + hashlib.sha256(raw_input.encode()).hexdigest()

        injection_result = scan_input(raw_input)
        if not injection_result.is_clean:
            emit_event(trace_id, "intent_interpreter", "INJECTION_BLOCKED",
                       {"risk_score": injection_result.risk_score,
                        "matched_patterns": injection_result.matched_patterns})
            raise PolicyViolation(f"Input failed injection scan (risk={injection_result.risk_score:.2f})")

        emit_event(trace_id, "intent_interpreter", "CLASSIFY_START")

        rule_result = _rule_classify(raw_input)
        if rule_result:
            intent_class, action_type, confidence, ambiguity_score = rule_result
            classification_method = "rule"

            if ambiguity_score > 0.7:
                emit_event(trace_id, "intent_interpreter", "CLARIFICATION_REQUIRED",
                           {"ambiguity_score": ambiguity_score})
                raise ClarificationRequired(session_id, ambiguity_score)

            entity_id, entity_class = _extract_entity(raw_input)
            intent = Intent(
                session_id=session_id,
                intent_class=intent_class,
                action_type=ActionType(action_type),
                target_entity_id=entity_id,
                target_entity_class=entity_class,
                urgency=Urgency.NORMAL,
                ambiguity_score=ambiguity_score,
                raw_input_hash=raw_input_hash,
                raw_input=raw_input,
            )
            emit_event(trace_id, "intent_interpreter", "INTENT_CLASSIFIED",
                       {"intent_class": intent_class, "action_type": action_type,
                        "ambiguity_score": ambiguity_score, "method": classification_method})
            _persist_intent(raw_input, intent)
            return intent

        recalled = _recall_intent(raw_input)
        if recalled is not None:
            emit_event(trace_id, "intent_interpreter", "INTENT_RECALLED",
                       {"intent_class": recalled.intent_class, "action_type": recalled.action_type})
            recalled.session_id = session_id
            recalled.raw_input_hash = raw_input_hash
            recalled.raw_input = raw_input
            return recalled

        try:
            intent = await self._classify_with_llm(raw_input, session_id, trace_id, raw_input_hash)
            _persist_intent(raw_input, intent)
            return intent
        except ClarificationRequired:
            raise
        except Exception as exc:
            emit_event(trace_id, "intent_interpreter", "LLM_CLASSIFY_FAILED",
                       {"error": str(exc)})
            entity_id, entity_class = _extract_entity(raw_input)
            intent = Intent(
                session_id=session_id,
                intent_class=IntentClass.QUERY,
                action_type=ActionType.ANALYZE,
                target_entity_id=entity_id,
                target_entity_class=entity_class,
                urgency=Urgency.NORMAL,
                ambiguity_score=0.5,
                raw_input_hash=raw_input_hash,
                raw_input=raw_input,
            )
            _persist_intent(raw_input, intent)
            return intent

    async def _classify_with_llm(
        self,
        raw_input: str,
        session_id: UUID,
        trace_id: str,
        raw_input_hash: str,
    ) -> Intent:
        emit_event(trace_id, "intent_interpreter", "LLM_CLASSIFY_START")

        async with httpx.AsyncClient(
            base_url=settings.litellm_proxy_url, timeout=settings.litellm_proxy_timeout_s
        ) as client:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": settings.intent_classifier_model,
                    "messages": [
                        {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
                        {"role": "user", "content": raw_input},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            body = resp.json()
            content = body["choices"][0]["message"]["content"]

        parsed = json.loads(content)
        emit_event(trace_id, "intent_interpreter", "LLM_CLASSIFY_DONE",
                   {"llm_output": parsed})

        intent_class = IntentClass(parsed.get("intent_class", "QUERY"))
        action_type_str = parsed.get("action_type", "ANALYZE")
        entity_class = parsed.get("entity_class", "RESOURCE")
        urgency = Urgency(parsed.get("urgency", "NORMAL"))
        entity_id = parsed.get("entity_id", "unknown")
        clarifications_needed = parsed.get("clarifications_needed", [])

        if clarifications_needed:
            raise ClarificationRequired(
                session_id=session_id,
                ambiguity_score=0.8,
                questions=clarifications_needed,
            )

        intent = Intent(
            session_id=session_id,
            intent_class=intent_class,
            action_type=ActionType(action_type_str),
            target_entity_id=entity_id,
            target_entity_class=entity_class,
            urgency=urgency,
            ambiguity_score=0.0,
            raw_input_hash=raw_input_hash,
            raw_input=raw_input,
        )
        emit_event(trace_id, "intent_interpreter", "INTENT_CLASSIFIED",
                   {"intent_class": intent_class, "action_type": action_type_str,
                    "ambiguity_score": 0.0, "method": "llm"})
        return intent
