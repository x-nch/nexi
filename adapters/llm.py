"""Shared chat-completions client for Nexi's inference.

All internal LLM call sites (intent classification, option generation, the
model adapter, reflection, and the chat tool loop) go through
:func:`chat_completion`, which resolves a provider + model via
:mod:`nexi.adapters.model_router` and POSTs to the resolved backend's
``/chat/completions`` endpoint.

This keeps Nexi the single decision-maker for *which* model serves each request
(chat + internals by default) while making it trivial to point a specific call
at opencode or openrouter explicitly (per-agent / per-step overrides).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from ..config import settings
from .model_router import PROVIDER_OPENROUTER, resolve

logger = logging.getLogger(__name__)


def _api_headers(api_key: str, provider: str) -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        prefix = ""
        if provider == PROVIDER_OPENROUTER:
            # OpenRouter returns HTTP-200-with-error-body on bad keys; sending
            # the marker header lets it surface a clean 401 instead.
            headers["HTTP-Referer"] = "https://github.com/x-nch/xnch"
            headers["X-Title"] = "xnch/nexi"
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


async def chat_completion(
    messages: list[dict[str, Any]],
    *,
    intent_class: str = "QUERY",
    budget: str | None = None,
    provider: str | None = None,
    model_id: str | None = None,
    response_format: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    json_mode: bool = False,
) -> tuple[dict[str, Any], ModelResolution]:
    """POST a chat completion to the resolved provider.

    Returns ``(response_json, resolution)``. On network/HTTP error the caller
    decides fallback behavior (rule-based options, next provider, etc.).
    """
    from .model_router import ModelResolution  # local to avoid circular import at module load

    target = resolve(
        intent_class=intent_class,
        budget=budget or getattr(settings, "model_budget", "balanced"),
        provider=provider,
        model_id=model_id,
    )
    payload: dict[str, Any] = {"model": target.model_id, "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format:
        payload["response_format"] = response_format
    elif json_mode:
        payload["response_format"] = {"type": "json_object"}
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    t0 = time.time()
    headers = _api_headers(target.api_key, target.provider)
    async with httpx.AsyncClient(
        base_url=target.base_url, timeout=target.timeout_s, headers=headers
    ) as client:
        resp = await client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        body = resp.json()
    await _emit_trace(messages, body, target, int((time.time() - t0) * 1000))
    return body, target


async def _emit_trace(
    messages: list[dict[str, Any]], body: dict[str, Any], target: "ModelResolution", latency_ms: int
) -> None:
    """Best-effort Langfuse trace for the call (never blocks the response path)."""
    try:
        from xnch.observability.langfuse_client import trace_llm_call

        # Truncate the message payload for trace storage.
        prompt = json.dumps(messages[-1] if messages else {})[:4000]
        content = ""
        try:
            content = body["choices"][0]["message"].get("content", "") or ""
        except Exception:
            content = ""
        await trace_llm_call(
            prompt=prompt,
            response=content[:8000],
            model=target.model_id,
            latency_ms=latency_ms,
            tokens_used=body.get("usage", {}).get("total_tokens", 0),
        )
    except Exception as exc:  # tracing must never fail the request
        logger.debug("LLM trace emission failed: %s", exc)


def extract_content(body: dict[str, Any]) -> str:
    """Extract the assistant text from a chat-completion response body."""
    try:
        return body["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""
