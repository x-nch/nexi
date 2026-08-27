"""Reflector — Summary step producing structured experiential lessons.

Wraps the LLM reflection call and persistence to xnch's experience store.
Injected callables keep this unit-testable without an LLM or network.
"""
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic import BaseModel

from ..adapters.xnch_client import XnchClient
from ..config import settings

logger = logging.getLogger(__name__)

DEFAULT_VERDICTS = {"ALLOW", "ALLOW_WITH_WARNINGS", "BLOCK", "MODIFY", "DEFER"}

_REFLECTION_SYSTEM_PROMPT = """You are a reflection engine for the XNCH autonomous agent.
Given the outcome of a decision, reflect on what happened and extract a reusable lesson.

Return JSON with exactly these fields:
- verdict: one of ALLOW, ALLOW_WITH_WARNINGS, BLOCK, MODIFY, DEFER (how future identical actions should be treated)
- lesson: one-sentence actionable lesson for future planning
- insight: one-sentence explanation of what went right or wrong
- applicability: pipe-separated intent_class|action_type|entity_class scope (or "" if universal)
"""


class ReflectionRecord(BaseModel):
    verdict: str
    lesson: str
    insight: str
    applicability: str = ""


def _context_signature(intent_class: str, action_type: str, entity_class: str, actor_role: str) -> str:
    canonical = "|".join([
        intent_class.lower(), action_type.lower(), entity_class.lower(), actor_role.lower()
    ])
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


class Reflector:
    def __init__(
        self,
        llm_fn: Callable[..., Awaitable[dict[str, Any]]],
        write_fn: Callable[..., Awaitable[None]],
    ) -> None:
        self._llm = llm_fn
        self._write = write_fn

    async def reflect(
        self,
        session_id: str,
        trace_id: str,
        intent_class: str,
        action_type: str,
        entity_class: str,
        actor_role: str,
        outcome: str,
        prediction_delta: float,
        context_summary: dict[str, Any],
    ) -> ReflectionRecord | None:
        try:
            raw = await self._llm(
                outcome=outcome,
                prediction_delta=prediction_delta,
                context_summary=context_summary,
            )
        except Exception as exc:
            logger.warning("Reflection LLM call failed (trace=%s): %s", trace_id, exc)
            return None

        record = self._parse(raw)
        if record is None:
            return None

        try:
            await self._write(
                context_signature=_context_signature(
                    intent_class, action_type, entity_class, actor_role
                ),
                intent_class=intent_class,
                action_type=action_type,
                entity_class=entity_class,
                actor_role=actor_role,
                outcome=outcome,
                lesson=record.lesson,
                insight=record.insight,
                verdict=record.verdict,
                applicability=record.applicability or "|".join([intent_class, action_type, entity_class]),
            )
        except Exception as exc:
            logger.warning("Reflection write failed (trace=%s): %s", trace_id, exc)
            return None

        logger.info(
            "Reflection stored: %s|%s|%s|%s verdict=%s",
            intent_class, action_type, entity_class, actor_role, record.verdict,
        )
        return record

    def _parse(self, raw: dict[str, Any]) -> ReflectionRecord | None:
        verdict = raw.get("verdict", "")
        lesson = (raw.get("lesson") or "").strip()
        if verdict not in DEFAULT_VERDICTS or not lesson:
            logger.warning("Reflection rejected: missing verdict or lesson (%s)", raw)
            return None
        return ReflectionRecord(
            verdict=verdict,
            lesson=lesson,
            insight=(raw.get("insight") or "").strip(),
            applicability=(raw.get("applicability") or "").strip(),
        )


async def _default_llm_fn(
    outcome: str,
    prediction_delta: float,
    context_summary: dict[str, Any],
) -> dict[str, Any]:
    """Reflection via LiteLLM proxy; returns parsed JSON dict."""
    user_prompt = (
        "Based on the following outcome data, generate a reflection JSON.\n"
        "Return exactly: {\"verdict\": \"...\", \"lesson\": \"...\", \"insight\": \"...\", \"applicability\": \"...\"}\n\n"
        + json.dumps({
            "outcome": outcome,
            "prediction_delta": prediction_delta,
            "context_summary": context_summary,
        })
    )
    _headers = {"Content-Type": "application/json"}
    if settings.opencode_go_api_key:
        _headers["Authorization"] = f"Bearer {settings.opencode_go_api_key}"
    async with httpx.AsyncClient(
        base_url=settings.opencode_go_api_url, timeout=settings.opencode_go_api_timeout_s
    ) as client:
        resp = await client.post(
            "/chat/completions",
            json={
                "model": settings.reflection_model,
                "messages": [
                    {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
                "max_tokens": 400,
            },
            headers=_headers,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
        return parsed if isinstance(parsed, dict) else {}


def build_reflector(xnch: XnchClient) -> Reflector:
    """Production Reflector wired to the LiteLLM proxy and xnch experience store."""
    return Reflector(llm_fn=_default_llm_fn, write_fn=xnch.write_experience)
