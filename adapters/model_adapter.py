"""Model Adapter — Contract 5 fallback chain:
  1. vllm-primary     (timeout > 30s → next)
  2. vllm-secondary   (timeout > 45s or unavailable → next)
  3. llama-cpp-python (any failure → next)
  4. rule-based       (fallback of last resort)
"""
import hashlib
import json
import uuid
from typing import Any

import httpx

from ..config import settings
from ..models import PlanOption, GenerationPath
from ..models.options import ActionSpec
from ..models.intent import IntentClass


_RULE_BASED_TEMPLATES: dict[str, list[dict]] = {
    IntentClass.QUERY: [
        {"action_type": "READ_FILE", "action_spec": {"target": "", "params": {"operation": "read", "scope": "requested_entity_only"}},
         "stated_rationale": "Read-only retrieval with minimal scope", "estimated_side_effects": [], "reversible": True},
        {"action_type": "LIST", "action_spec": {"target": "", "params": {"operation": "list", "scope": "requested_entity_only"}},
         "stated_rationale": "Non-modifying list operation", "estimated_side_effects": [], "reversible": True},
        {"action_type": "ANALYZE", "action_spec": {"target": "", "params": {"operation": "analyze", "scope": "requested_entity_only"}},
         "stated_rationale": "Analysis only, no state change", "estimated_side_effects": [], "reversible": True},
    ],
    IntentClass.DECISION: [
        {"action_type": "PLAN", "action_spec": {"target": "", "params": {"operation": "draft_plan", "commit": False}},
         "stated_rationale": "Draft plan without commitment", "estimated_side_effects": [], "reversible": True},
        {"action_type": "ANALYZE", "action_spec": {"target": "", "params": {"operation": "analyze", "commit": False}},
         "stated_rationale": "Analysis to inform decision", "estimated_side_effects": [], "reversible": True},
        {"action_type": "ESCALATE", "action_spec": {"target": "", "params": {"operation": "escalate", "reason": "inference_unavailable"}},
         "stated_rationale": "Escalate to operator — inference unavailable for decision support", "estimated_side_effects": [], "reversible": True},
    ],
    IntentClass.EXECUTION: [
        {"action_type": "BACKUP", "action_spec": {"target": "", "params": {"operation": "backup", "scope": "affected_entities"}},
         "stated_rationale": "Backup before any execution; safe first step", "estimated_side_effects": ["storage_write"], "reversible": True},
        {"action_type": "ANALYZE", "action_spec": {"target": "", "params": {"operation": "dry_run", "commit": False}},
         "stated_rationale": "Dry-run analysis without execution", "estimated_side_effects": [], "reversible": True},
        {"action_type": "ESCALATE", "action_spec": {"target": "", "params": {"operation": "escalate", "reason": "inference_unavailable"}},
         "stated_rationale": "Escalate to operator — inference unavailable for execution planning", "estimated_side_effects": [], "reversible": True},
    ],
    IntentClass.ESCALATION: [
        {"action_type": "ESCALATE", "action_spec": {"target": "", "params": {"operation": "escalate", "reason": "inference_unavailable"}},
         "stated_rationale": "Escalate as originally requested", "estimated_side_effects": [], "reversible": True},
        {"action_type": "READ_FILE", "action_spec": {"target": "", "params": {"operation": "read", "scope": "audit_log"}},
         "stated_rationale": "Read audit log to inform escalation context", "estimated_side_effects": [], "reversible": True},
        {"action_type": "ANALYZE", "action_spec": {"target": "", "params": {"operation": "analyze", "scope": "recent_decisions"}},
         "stated_rationale": "Analyze recent decisions for escalation context", "estimated_side_effects": [], "reversible": True},
    ],
}

_FORBIDDEN_RULE_BASED = {
    "RUN_COMMAND", "RUN_SCRIPT", "DEPLOY", "ROLLBACK",
    "DELETE_FILE", "MUTATE",
}


def _payload_hash(action_spec: dict) -> str:
    digest = hashlib.sha256(json.dumps(action_spec, sort_keys=True).encode()).hexdigest()
    return f"sha256:{digest}"


def _build_rule_based_option(template: dict, target_entity_id: str) -> PlanOption:
    spec_raw = dict(template["action_spec"])
    spec_raw["target"] = target_entity_id
    spec = ActionSpec(
        type=template["action_type"],
        target=target_entity_id,
        params=spec_raw.get("params", {}),
    )
    return PlanOption(
        option_id=uuid.uuid4(),
        action_type=template["action_type"],
        action_spec=spec,
        stated_rationale=template["stated_rationale"],
        estimated_side_effects=template["estimated_side_effects"],
        reversible=True,
        payload_hash=_payload_hash(spec.model_dump()),
    )


def _rule_based_options(intent_class: str, target_entity_id: str) -> list[PlanOption]:
    templates = _RULE_BASED_TEMPLATES.get(intent_class, _RULE_BASED_TEMPLATES[IntentClass.ESCALATION])
    return [_build_rule_based_option(t, target_entity_id) for t in templates]


class ModelAdapter:
    """Routes constrained generation requests through the fallback chain."""

    async def generate_options(
        self,
        intent_class: str,
        target_entity_id: str,
        target_entity_class: str,
        context_summary: dict[str, Any],
        n: int = 5,
    ) -> tuple[list[PlanOption], GenerationPath]:
        prompt_payload = self._build_prompt(
            intent_class, target_entity_id, target_entity_class, context_summary, n
        )

        for attempt, (url, timeout) in enumerate([
            (settings.vllm_primary_url, settings.vllm_primary_timeout_s),
            (settings.vllm_secondary_url, settings.vllm_secondary_timeout_s),
        ]):
            if not url:
                continue
            try:
                options = await self._call_vllm(url, timeout, prompt_payload, intent_class, target_entity_id)
                if options:
                    return options, GenerationPath.MODEL
            except Exception:
                pass

        # llama-cpp-python stub — same interface, local inference
        try:
            options = await self._call_llama_cpp(prompt_payload, intent_class, target_entity_id)
            if options:
                return options, GenerationPath.MODEL
        except Exception:
            pass

        return _rule_based_options(intent_class, target_entity_id), GenerationPath.RULE_BASED

    def _build_prompt(
        self,
        intent_class: str,
        target_entity_id: str,
        target_entity_class: str,
        context_summary: dict[str, Any],
        n: int,
    ) -> dict[str, Any]:
        return {
            "template_version": "gen-v1.0",
            "intent": {
                "class": intent_class,
                "entity_id": target_entity_id,
                "entity_class": target_entity_class,
            },
            "context_summary": context_summary,
            "output_schema": {
                "type": "array",
                "items": {
                    "option_id": "uuid",
                    "action_type": "string",
                    "action_spec": {"target": "string", "params": "object"},
                    "stated_rationale": "string",
                    "estimated_side_effects": "string[]",
                    "reversible": "bool",
                },
                "minItems": 3,
                "maxItems": 7,
                "count": n,
            },
            "instruction": "Generate only. Do not evaluate. Do not select.",
        }

    async def _call_vllm(
        self,
        base_url: str,
        timeout: float,
        prompt_payload: dict,
        intent_class: str,
        target_entity_id: str,
    ) -> list[PlanOption]:
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": settings.model_id,
                    "messages": [
                        {"role": "system", "content": "You are an option generator. Return valid JSON only."},
                        {"role": "user", "content": json.dumps(prompt_payload)},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            raw_options = resp.json()["choices"][0]["message"]["content"]
            return self._parse_options(raw_options, target_entity_id)

    async def _call_llama_cpp(
        self,
        prompt_payload: dict,
        intent_class: str,
        target_entity_id: str,
    ) -> list[PlanOption]:
        # llama-cpp-python exposes an OpenAI-compatible server on localhost:8080 by default
        async with httpx.AsyncClient(base_url="http://localhost:8080", timeout=60.0) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are an option generator. Return valid JSON only."},
                        {"role": "user", "content": json.dumps(prompt_payload)},
                    ],
                    "grammar": None,
                },
            )
            resp.raise_for_status()
            raw_options = resp.json()["choices"][0]["message"]["content"]
            return self._parse_options(raw_options, target_entity_id)

    def _parse_options(self, raw: str | dict, target_entity_id: str) -> list[PlanOption]:
        data = raw if isinstance(raw, list) else json.loads(raw)
        if isinstance(data, dict):
            data = data.get("options", list(data.values())[0] if data else [])

        options = []
        for item in data:
            try:
                spec = ActionSpec(
                    type=item.get("action_type", ""),
                    target=item.get("action_spec", {}).get("target", target_entity_id),
                    params=item.get("action_spec", {}).get("params", {}),
                )
                opt = PlanOption(
                    option_id=uuid.uuid4(),
                    action_type=item["action_type"].upper(),
                    action_spec=spec,
                    stated_rationale=item.get("stated_rationale", ""),
                    estimated_side_effects=item.get("estimated_side_effects", []),
                    reversible=item.get("reversible", True),
                    payload_hash=_payload_hash(spec.model_dump()),
                )
                options.append(opt)
            except Exception:
                continue
        return options
