"""Tests for dynamic opencode-go model routing."""

from __future__ import annotations

from nexi.adapters import model_router as mr
from nexi.adapters.model_router import (
    available_models,
    fallback_chain,
    routing_summary,
    select_model,
)
from nexi.config import settings


def test_select_model_balanced_decision_prefers_pro():
    m = select_model("DECISION", budget="balanced")
    assert m.id == "deepseek-v4-pro"


def test_select_model_cheap_query_prefers_lite():
    m = select_model("QUERY", budget="cheap")
    assert m.id == "deepseek-v4-lite"


def test_select_model_quality_query_prefers_pro():
    m = select_model("QUERY", budget="quality")
    assert m.id == "deepseek-v4-pro"


def test_select_model_escalation_prefers_reasoner():
    m = select_model("ESCALATION", budget="balanced")
    assert m.id == "deepseek-v4-reasoner"


def test_fallback_chain_orders_preferred_first():
    assert fallback_chain("DECISION")[0] == "deepseek-v4-pro"
    assert fallback_chain("QUERY")[0] == "deepseek-v4-lite"


def test_routing_summary_mentions_intents():
    summary = routing_summary()
    assert "DECISION" in summary and "QUERY" in summary


def test_available_models_nonempty():
    assert "deepseek-v4-pro" in available_models()


def test_select_model_respects_settings_budget_override(monkeypatch):
    # High-stakes intents only have reasoning candidates, so a "cheap" budget still
    # picks a reasoning model; use QUERY (which has a cheap candidate) to prove override.
    monkeypatch.setattr(settings, "model_budget", "cheap")
    m = select_model("QUERY", budget="balanced")  # explicit arg wins -> lite
    assert m.id == "deepseek-v4-lite"
    m2 = select_model("QUERY")  # falls back to settings.budget=cheap -> lite
    assert m2.id == "deepseek-v4-lite"
    m3 = select_model("QUERY", budget="quality")  # quality overrides settings -> pro
    assert m3.id == "deepseek-v4-pro"
