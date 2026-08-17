"""Step 5 — Constrained option generation via LiteLLM structured output."""
import json
import uuid
from typing import Any

import httpx

from ..adapters.model_adapter import ModelAdapter
from ..config import settings
from ..models import SessionContext, Intent, ContextManifest, PlanOption
from ..models.options import GenerationPath, ActionSpec
from ..utils.audit import emit_event
from xnch.routing.classifier import classify_request


_OPTION_GENERATION_SYSTEM_PROMPT = """You are an option generator for a systems-management agent.
Given an intent summary and context, generate exactly N distinct PlanOption objects.

Each option must include:
- action_type: one of READ_FILE, WRITE_FILE, DELETE_FILE, LIST, RUN_COMMAND, RUN_SCRIPT,
  DEPLOY, ROLLBACK, STAGE, MUTATE, BACKUP, RESTORE, PLAN, ANALYZE, ESCALATE, QUERY
- action_spec: {"target": "<entity target>", "params": {<key-value params>}}
- stated_rationale: why this option is viable
- estimated_side_effects: list of strings describing potential state changes
- reversible: whether the action can be undone

Respond with a JSON object:
{"options": [list of PlanOption objects]}

Available actions: {available_actions}

Generate exactly N={options_count} options. Return valid JSON only."""


def _build_context_summary(manifest: ContextManifest) -> dict:
    outcomes = {"S": 0, "P": 0, "F": 0}
    for ep in manifest.episodes:
        if ep.outcome == "SUCCESS":
            outcomes["S"] += 1
        elif ep.outcome == "PARTIAL":
            outcomes["P"] += 1
        elif ep.outcome == "FAILURE":
            outcomes["F"] += 1

    dominant = None
    if manifest.patterns:
        dominant = max(manifest.patterns, key=lambda p: p.confidence)

    summary = {
        "recent_outcomes": f"{outcomes['S']}S/{outcomes['P']}P/{outcomes['F']}F",
        "dominant_pattern": (
            f"{dominant.success_rate:.2f} success (conf={dominant.confidence:.2f})"
            if dominant else "no pattern"
        ),
    }

    if manifest.experiences:
        ranked = sorted(manifest.experiences, key=lambda e: e.confidence, reverse=True)
        summary["recent_lessons"] = [
            f"{e.lesson} ({e.verdict})" for e in ranked
        ]

    return summary


def _build_option_prompt(intent: Intent, context_summary: dict, n: int) -> str:
    available_actions = [
        "READ_FILE", "WRITE_FILE", "DELETE_FILE", "LIST",
        "RUN_COMMAND", "RUN_SCRIPT", "DEPLOY", "ROLLBACK",
        "STAGE", "MUTATE", "BACKUP", "RESTORE",
        "PLAN", "ANALYZE", "ESCALATE", "QUERY",
    ]
    intent_summary = {
        "class": intent.intent_class,
        "action_type": intent.action_type,
        "target_entity_id": intent.target_entity_id,
        "target_entity_class": intent.target_entity_class,
        "urgency": intent.urgency,
    }
    prompt_data = {
        "intent": intent_summary,
        "context": context_summary,
    }
    system_prompt = _OPTION_GENERATION_SYSTEM_PROMPT.format(
        available_actions=json.dumps(available_actions),
        options_count=n,
    )
    return system_prompt, json.dumps(prompt_data)


async def generate_options(
    adapter: ModelAdapter,
    session: SessionContext,
    intent: Intent,
    manifest: ContextManifest,
    n: int = 5,
) -> tuple[list[PlanOption], GenerationPath]:
    emit_event(session.trace_id, "option_generator", "GENERATION_START",
               {"n": n, "intent_class": intent.intent_class})

    context_summary = _build_context_summary(manifest)

    model_route = classify_request(
        raw_input=intent.raw_input,
        actor_role=session.actor.role,
        metadata={
            "intent_class": intent.intent_class,
            "complexity_score": 0.5,
            "privacy_sensitive": False,
        },
    )

    try:
        system_msg, user_msg = _build_option_prompt(intent, context_summary, n)
        async with httpx.AsyncClient(
            base_url=settings.litellm_proxy_url, timeout=settings.litellm_proxy_timeout_s
        ) as client:
            resp = await client.post(
                "/chat/completions",
                json={
                    "model": model_route.model_name,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            items = parsed.get("options", [])
            options = []
            for item in items:
                spec = ActionSpec(
                    type=item.get("action_type", ""),
                    target=item.get("action_spec", {}).get("target", intent.target_entity_id),
                    params=item.get("action_spec", {}).get("params", {}),
                )
                options.append(PlanOption(
                    option_id=uuid.uuid4(),
                    action_type=item["action_type"].upper(),
                    action_spec=spec,
                    stated_rationale=item.get("stated_rationale", ""),
                    estimated_side_effects=item.get("estimated_side_effects", []),
                    reversible=item.get("reversible", True),
                    payload_hash="sha256:" + __import__("hashlib").sha256(json.dumps(spec.model_dump(), sort_keys=True).encode()).hexdigest(),
                ))

        if options:
            emit_event(session.trace_id, "option_generator", "GENERATION_COMPLETE",
                       {"options_count": len(options), "path": "MODEL"})
            return options, GenerationPath.MODEL
    except Exception:
        pass

    options, path = await adapter.generate_options(
        intent_class=intent.intent_class,
        target_entity_id=intent.target_entity_id,
        target_entity_class=intent.target_entity_class,
        context_summary=context_summary,
        n=n,
    )

    emit_event(session.trace_id, "option_generator", "GENERATION_COMPLETE",
               {"options_count": len(options), "path": path})
    return options, path
