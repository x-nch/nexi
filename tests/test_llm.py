"""Tests for chat_completion_with_fallback auto model selection.

Covers the ``method="auto"`` path: when Redis rankings exist, the wrapper
delegates to :class:`~nexi.adapters.model_selector.ModelSelector` and resolves
to the top-ranked model; when no rankings exist (or the selector fails), it
falls back to static ``resolve()``.
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import fakeredis


async def test_auto_method_uses_selector_when_rankings_exist(monkeypatch):
    """When method='auto' and Redis has rankings, resolve to the top-ranked model."""
    client = fakeredis.FakeRedis()
    rankings = [{
        "model_id": "nvidia/nemotron-3-super-120b-a12b:free",
        "provider": "openrouter",
        "score": 0.80,
        "quality": 0.75,
        "latency_ms": 3500,
        "context_window": 131072,
        "elo": 1250.0,
        "tier": "strong",
        "strengths": ["QUERY"],
    }]
    client.set("model_selector:rankings", json.dumps(rankings), ex=86400)
    monkeypatch.setattr("nexi.adapters.model_selector._redis_client", client)

    from nexi.adapters.llm import chat_completion_with_fallback

    fake_post = AsyncMock()
    fake_post.return_value.status_code = 200
    fake_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "hello"}}],
        "model": "nvidia/nemotron-3-super-120b-a12b:free",
    }
    fake_post.return_value.raise_for_status = MagicMock()
    fake_post.return_value.headers = {"x-request-id": "test"}

    monkeypatch.setattr("nexi.adapters.llm.httpx.AsyncClient.__aenter__", AsyncMock(return_value=MagicMock(post=fake_post)))
    monkeypatch.setattr("nexi.adapters.llm.httpx.AsyncClient.__aexit__", AsyncMock(return_value=False))

    body, resolution = await chat_completion_with_fallback(
        messages=[{"role": "user", "content": "hi"}],
        method="auto",
    )
    assert resolution.model_id == "nvidia/nemotron-3-super-120b-a12b:free"
    assert resolution.provider == "openrouter"


async def test_auto_method_falls_back_to_static_when_no_rankings(monkeypatch):
    """When method='auto' but Redis is empty, use static resolve()."""
    import fakeredis
    monkeypatch.setattr("nexi.adapters.model_selector._redis_client", fakeredis.FakeRedis())
    from nexi.adapters.llm import chat_completion_with_fallback

    fake_post = AsyncMock()
    fake_post.return_value.status_code = 200
    fake_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "hello"}}],
        "model": "ornith",
    }
    fake_post.return_value.raise_for_status = MagicMock()
    fake_post.return_value.headers = {"x-request-id": "test"}

    monkeypatch.setattr("nexi.adapters.llm.httpx.AsyncClient.__aenter__", AsyncMock(return_value=MagicMock(post=fake_post)))
    monkeypatch.setattr("nexi.adapters.llm.httpx.AsyncClient.__aexit__", AsyncMock(return_value=False))

    body, resolution = await chat_completion_with_fallback(
        messages=[{"role": "user", "content": "hi"}],
        method="auto",
    )
    # Falls back to static resolution — litellm/ornith on this env
    assert resolution.provider in ("litellm", "openrouter", "opencode")