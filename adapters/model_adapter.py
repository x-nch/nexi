import hashlib
import json
import logging
import time
import uuid
from typing import Any

import httpx

from ..config import settings
from ..models import PlanOption, GenerationPath
from ..models.options import ActionSpec
from ..models.intent import IntentClass
from xnch.observability.langfuse_client import trace_llm_call

logger = logging.getLogger(__name__)


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
    """Routes constrained generation through OpenCode Go API (DeepSeek V4)."""

    def _api_headers(self) -> dict[str, str]:
        """Build authorization headers for OpenCode Go API."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if settings.opencode_go_api_key:
            headers["Authorization"] = f"Bearer {settings.opencode_go_api_key}"
        return headers

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

        try:
            options = await self._call_opencode_go(
                prompt_payload, intent_class, target_entity_id, settings.model_id
            )
            if options:
                return options, GenerationPath.MODEL
        except Exception as exc:
            logger.warning("OpenCode Go API call failed: %s", exc)

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

    async def _call_opencode_go(
        self,
        prompt_payload: dict,
        intent_class: str,
        target_entity_id: str,
        model_name: str,
    ) -> list[PlanOption]:
        """Call OpenCode Go API (DeepSeek V4) for option generation."""
        if not settings.opencode_go_api_key:
            logger.warning("OpenCode Go API key unset — failing closed to rule-based options")
            return []
        prompt_text = json.dumps(prompt_payload)
        t0 = time.time()
        async with httpx.AsyncClient(
            base_url=settings.opencode_go_api_url,
            timeout=settings.opencode_go_api_timeout_s,
            headers=self._api_headers(),
        ) as client:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are an option generator. Return valid JSON only."},
                        {"role": "user", "content": prompt_text},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            raw_options = resp.json()["choices"][0]["message"]["content"]
            latency_ms = int((time.time() - t0) * 1000)
            tokens_used = resp.json().get("usage", {}).get("total_tokens", 0)
            await trace_llm_call(
                prompt=prompt_text,
                response=raw_options,
                model=model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )
            return self._parse_options(raw_options, target_entity_id)

    # Legacy fallback methods (kept for emergency rollback — disabled by default)

    async def _call_litellm(
        self,
        base_url: str,
        timeout: float,
        prompt_payload: dict,
        intent_class: str,
        target_entity_id: str,
        model_name: str,
    ) -> list[PlanOption]:
        """Legacy: LiteLLM proxy fallback. Only called if vllm_primary_url is set."""
        prompt_text = json.dumps(prompt_payload)
        t0 = time.time()
        _headers = {"Authorization": f"Bearer {settings.litellm_api_key}"} if settings.litellm_api_key else {}
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout, headers=_headers) as client:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are an option generator. Return valid JSON only."},
                        {"role": "user", "content": prompt_text},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            raw_options = resp.json()["choices"][0]["message"]["content"]
            latency_ms = int((time.time() - t0) * 1000)
            tokens_used = resp.json().get("usage", {}).get("total_tokens", 0)
            await trace_llm_call(
                prompt=prompt_text,
                response=raw_options,
                model=model_name,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
            )
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
