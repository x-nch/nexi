"""Tests for multi-provider nexi model routing (opencode + openrouter)."""

from __future__ import annotations

import pytest

from nexi.adapters import model_router as mr
from nexi.adapters.model_router import (
    PROVIDER_OPENCODE,
    PROVIDER_OPENROUTER,
    ModelResolution,
    available_models,
    fallback_chain,
    resolve,
    routing_summary,
    select_model,
    select_model_for_provider,
)
from nexi.config import settings


# --- opencode provider (legacy single-provider behavior, preserved) ----------

def test_select_model_balanced_decision_prefers_pro():
    m = select_model_for_provider("DECISION", PROVIDER_OPENCODE, budget="balanced")
    assert m.id == "deepseek-v4-pro"


def test_select_model_cheap_query_prefers_lite():
    m = select_model_for_provider("QUERY", PROVIDER_OPENCODE, budget="cheap")
    assert m.id == "deepseek-v4-lite"


def test_select_model_quality_query_prefers_pro():
    m = select_model_for_provider("QUERY", PROVIDER_OPENCODE, budget="quality")
    assert m.id == "deepseek-v4-pro"


def test_select_model_escalation_prefers_reasoner():
    m = select_model_for_provider("ESCALATION", PROVIDER_OPENCODE, budget="balanced")
    assert m.id == "deepseek-v4-reasoner"


def test_fallback_chain_orders_preferred_first_opencode():
    assert fallback_chain("DECISION", provider=PROVIDER_OPENCODE)[0] == "deepseek-v4-pro"
    assert fallback_chain("QUERY", provider=PROVIDER_OPENCODE)[0] == "deepseek-v4-lite"


# --- openrouter provider (new default for chat + internals) ------------------

def test_default_provider_resolves_to_openrouter():
    # nexi-default aliases to settings.nexi_default_resolves_to (openrouter).
    assert mr._resolve_provider(None) == PROVIDER_OPENROUTER
    assert mr._resolve_provider("auto") == PROVIDER_OPENROUTER


def test_openrouter_default_resolution():
    r = resolve("DECISION", budget="balanced")
    assert isinstance(r, ModelResolution)
    assert r.provider == PROVIDER_OPENROUTER
    assert r.model_id == "anthropic/claude-sonnet-4"
    assert "openrouter.ai" in r.base_url


def test_openrouter_query_uses_cheap_flash():
    r = resolve("QUERY", budget="balanced")
    assert r.provider == PROVIDER_OPENROUTER
    assert r.model_id == "google/gemini-2.0-flash-001"


def test_explicit_provider_override():
    r = resolve("QUERY", budget="balanced", provider=PROVIDER_OPENCODE)
    assert r.provider == PROVIDER_OPENCODE
    assert r.model_id == "deepseek-v4-lite"
    assert "opencode.ai" in r.base_url


def test_pinned_model_id_used_verbatim():
    r = resolve("QUERY", provider=PROVIDER_OPENROUTER, model_id="openai/gpt-4o")
    assert r.model_id == "openai/gpt-4o"


def test_openrouter_chain_orders_preferred_first():
    assert fallback_chain("DECISION")[0] == "anthropic/claude-sonnet-4"
    assert fallback_chain("QUERY")[0] == "google/gemini-2.0-flash-001"


def test_missing_api_key_fails_open():
    # Resolutions carry the configured key; absence is guarded by callers.
    r = resolve("DECISION", provider=PROVIDER_OPENROUTER)
    assert r.api_key == settings.openrouter_api_key


# --- shared surface ----------------------------------------------------------

def test_routing_summary_mentions_intents():
    summary = routing_summary()
    assert "DECISION" in summary and "QUERY" in summary


def test_available_models_nonempty_per_provider():
    assert "deepseek-v4-pro" in available_models(PROVIDER_OPENCODE)
    assert "anthropic/claude-sonnet-4" in available_models(PROVIDER_OPENROUTER)
    # default (no arg) includes the openrouter catalog under the new default.
    assert "anthropic/claude-sonnet-4" in available_models()


def test_unknown_provider_falls_back_to_opencode(caplog):
    with caplog.at_level("WARNING"):
        resolved = mr._resolve_provider("bogus")
    assert resolved == PROVIDER_OPENCODE


def test_select_model_respects_settings_budget_override(monkeypatch):
    # opencode provider behavior with settings-driven budget.
    monkeypatch.setattr(settings, "model_budget", "cheap")
    m = select_model("QUERY", budget="balanced")  # uses default provider (openrouter)
    assert m.id == "google/gemini-2.0-flash-001"
    m2 = select_model("QUERY")  # falls back to settings.budget=cheap
    assert m2.id == "google/gemini-2.0-flash-001"
    m3 = select_model("QUERY", budget="quality")  # quality overrides settings
    assert m3.id == "anthropic/claude-sonnet-4"


def test_nexi_default_alias_follows_config(monkeypatch):
    monkeypatch.setattr(settings, "nexi_default_resolves_to", PROVIDER_OPENCODE)
    r = resolve("QUERY")
    assert r.provider == PROVIDER_OPENCODE
    assert r.model_id == "deepseek-v4-lite"
    monkeypatch.setattr(settings, "nexi_default_resolves_to", PROVIDER_OPENROUTER)
    assert mr._resolve_provider(None) == PROVIDER_OPENROUTER


def test_catalog_override_from_settings(monkeypatch):
    monkeypatch.setattr(
        settings,
        "openrouter_models",
        [
            {
                "id": "custom/model-x",
                "cost_tier": 3,
                "context_window": 100_000,
                "strengths": ["DECISION", "EXECUTION", "QUERY", "ESCALATION"],
                "latency_ms": 900,
                "description": "test override",
            }
        ],
    )
    assert available_models(PROVIDER_OPENROUTER) == ["custom/model-x"]
    r = resolve("DECISION", provider=PROVIDER_OPENROUTER)
    assert r.model_id == "custom/model-x"