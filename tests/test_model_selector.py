"""Tests for the free model selector scoring + ranking core."""

from __future__ import annotations

import math

import pytest

from nexi.adapters.model_selector import (
    TIER_SCORES,
    context_score,
    elo_to_score,
    normalized_latency,
    rank_models,
    tier_from_elo,
)


def test_tier_from_elo_frontier():
    assert tier_from_elo(None, "frontier") == 1.0
    assert tier_from_elo(None, "weak") == 0.25


def test_tier_from_elo_falls_back_to_mid_when_no_tier():
    assert tier_from_elo(None, None) == 0.5


def test_tier_from_elo_elo_wins_over_tier():
    # frontier tier but mediocre elo -> elo-derived score clamps below tier max
    score = tier_from_elo(1100.0, "frontier")
    assert score < 1.0
    assert score > 0.0


def test_context_score_scales_to_1m():
    assert context_score(1_000_000) == 1.0
    assert context_score(200_000) == 0.2
    assert context_score(0) == 0.0


def test_normalized_latency_caps_at_10s():
    assert normalized_latency(0) == 0.0
    assert normalized_latency(500) == pytest.approx(0.05)
    assert normalized_latency(20_000) == 1.0


def test_elo_to_score_midpoint_and_bounds():
    assert elo_to_score(0.0) == pytest.approx(0.05)
    assert elo_to_score(1200.0) == pytest.approx(0.05)
    assert elo_to_score(1500.0) == pytest.approx(1.0 - math.exp(-1.0), abs=1e-6)
    assert elo_to_score(20_000.0) == pytest.approx(1.0)


def test_rank_models_handles_missing_elo():
    models = [
        {"provider": "openrouter", "model_id": "a", "context_window": 128_000},
        {"provider": "openrouter", "model_id": "b", "context_window": 32_000},
    ]
    ranked = rank_models(models, {}, {"a": "frontier", "b": "weak"})
    assert ranked[0]["elo"] is None
    assert ranked[0]["model_id"] == "a"
    assert ranked[-1]["model_id"] == "b"


def test_rank_models_empty_input():
    assert rank_models([], {}, {}) == []


def test_rank_models_sorts_by_score_desc():
    models = [
        {"provider": "openrouter", "model_id": "a", "context_window": 128_000},
        {"provider": "openrouter", "model_id": "b", "context_window": 32_000},
    ]
    elo = {"a": 1300.0, "b": 900.0}
    tiers = {"a": "frontier", "b": "weak"}
    ranked = rank_models(models, elo, tiers)
    assert ranked[0]["model_id"] == "a"
    assert ranked[-1]["model_id"] == "b"
    assert set(ranked[0]) >= {
        "provider", "model_id", "score", "quality", "latency_ms",
        "context_window", "elo", "tier",
    }
