"""Step 3 — Intent Normalization (Contract 4)."""
import hashlib
import re
from typing import Any
from uuid import UUID

from ..models.intent import Intent, IntentClass, ActionType, Urgency
from ..utils.audit import emit_event


class ClarificationRequired(Exception):
    def __init__(self, session_id: UUID, ambiguity_score: float) -> None:
        self.session_id = session_id
        self.ambiguity_score = ambiguity_score
        super().__init__(f"Ambiguity score {ambiguity_score:.2f} exceeds threshold")


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
    # Minimal entity extraction for v0 — returns the noun phrase after the verb
    tokens = raw_input.strip().split()
    entity_id = tokens[1] if len(tokens) > 1 else "unknown"
    # Crude class heuristic
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


class IntentInterpreter:
    """Contract 4 two-stage classifier. v0 uses rule-based only; model stage is a stub."""

    async def interpret(self, raw_input: str, session_id: UUID, trace_id: str) -> Intent:
        raw_input_hash = "sha256:" + hashlib.sha256(raw_input.encode()).hexdigest()
        emit_event(trace_id, "intent_interpreter", "CLASSIFY_START")

        rule_result = _rule_classify(raw_input)
        if rule_result:
            intent_class, action_type, confidence, ambiguity_score = rule_result
            classification_method = "rule"
        else:
            # v0 stub: model classification deferred — fall back to QUERY/ANALYZE with high ambiguity
            intent_class = IntentClass.QUERY
            action_type = "ANALYZE"
            confidence = 0.5
            ambiguity_score = 1.0 - confidence
            classification_method = "model_stub"

        # Contract 4 §Fallback: if confidence < 0.7 and ambiguity ≤ 0.7, proceed with ambiguity flag
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
        )
        emit_event(trace_id, "intent_interpreter", "INTENT_CLASSIFIED",
                   {"intent_class": intent_class, "action_type": action_type,
                    "ambiguity_score": ambiguity_score, "method": classification_method})
        return intent
