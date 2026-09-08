"""Shared chat-completions client for Nexi's inference.

All internal LLM call sites (intent classification, option generation, the
model adapter, reflection, and the chat tool loop) go through
:func:`chat_completion`, which resolves a provider + model via
:mod:`nexi.adapters.model_router` and POSTs to the resolved backend's
``/chat/completions`` endpoint.

:func:`chat_completion_with_fallback` adds a cross-provider safety net: when the
primary provider (e.g. the local LiteLLM/vLLM proxy) errors, it fails over to
OpenRouter using **free models only** (``:free``-suffixed ids or the
``openrouter_free_models`` allowlist), never a paid catalog entry.

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


def is_free_model(model_id: str | None) -> bool:
    """True when a model id is OpenRouter free-tier: ``:free`` suffix or in the
    configured ``openrouter_free_models`` allowlist."""
    if not model_id:
        return False
    lowered = model_id.lower()
    if lowered.endswith(":free"):
        return True
    free_list = [m.lower() for m in settings.openrouter_free_models]
    return lowered in free_list


async def chat_completion_with_fallback(
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
    method: str | None = None,  # "auto" → dynamic selection via Redis rankings
) -> tuple[dict[str, Any], ModelResolution]:
    """Chat completion with cross-provider failover to OpenRouter free models.

    Tries the primary provider (default: nexi-default → local LiteLLM/vLLM when
    configured). On any transport/HTTP error it fails over to OpenRouter via
    ``settings.openrouter_free_model`` — guarded to free-tier ids only — when an
    OpenRouter API key is configured. The returned ``ModelResolution`` reflects
    which backend actually served the call.

    When *method* is ``"auto"`` (or defaults to ``settings.model_method``),
    consult Redis-backed rankings before the static resolution path: the
    top-ranked free model for the intent seeds the primary attempt. A selector
    failure (Redis down, no rankings) degrades to static resolution; it never
    triggers the OpenRouter failover on its own.
    """
    from .model_router import ModelResolution  # local to avoid circular import at module load

    effective_method = method or settings.model_method
    if effective_method == "auto" and not (provider or model_id):
        try:
            from .model_selector import ModelSelector

            selector = ModelSelector()
            best = await selector.get_best_model(intent_class, budget=budget or "balanced")
            if best and best.provider:
                provider = best.provider
                model_id = best.id
        except Exception as exc:
            logger.warning("Auto-selector failed: %s; falling back to static resolution", exc)

    try:
        return await chat_completion(
            messages,
            intent_class=intent_class,
            budget=budget,
            provider=provider,
            model_id=model_id,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            json_mode=json_mode,
        )
    except Exception as primary_exc:
        if not settings.openrouter_api_key:
            logger.warning(
                "Primary chat provider failed and no OpenRouter key is set; "
                "failing through: %s",
                primary_exc,
            )
            raise
        free_model = settings.openrouter_free_model
        if not is_free_model(free_model):
            logger.warning(
                "openrouter_free_model=%r is not free-tier; not failing over. "
                "Primary error: %s",
                free_model,
                primary_exc,
            )
            raise
        logger.warning(
            "Primary chat provider failed (%s); falling back to OpenRouter "
            "free model %r",
            primary_exc,
            free_model,
        )
        return await chat_completion(
            messages,
            intent_class=intent_class,
            budget=budget,
            provider=PROVIDER_OPENROUTER,
            model_id=free_model,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            json_mode=json_mode,
        )


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
