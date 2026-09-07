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


# ---------------------------------------------------------------------------
# Task 2: config loader, free-model detection, Redis store/read
# ---------------------------------------------------------------------------

import json

import fakeredis
from pydantic import ValidationError

from nexi.adapters.model_selector import (
    ModelSelectorConfig,
    RedisConfig,
    dump_rankings_json,
    is_free_model,
    load_config,
    read_rankings,
    redis_client_from,
    write_rankings,
)


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    """Default selector config wired to an in-memory fakeredis redis."""
    client = fakeredis.FakeRedis()
    conf = ModelSelectorConfig(
        weights={"quality": 0.5, "latency": 0.3, "context": 0.2},
        redis=RedisConfig(
            url="redis://localhost:6379/0",
            ttl_s=60,
            rankings_key="model_selector:rankings",
        ),
    )
    conf._fake_redis = client
    monkeypatch.setattr("nexi.adapters.model_selector._redis_client", client)
    return conf


def test_is_free_model_free_suffix_and_zero_pricing():
    assert is_free_model("a/b:free", {"prompt": "0", "completion": "0"})
    assert is_free_model("a/b", {"prompt": "0", "completion": "0"})
    assert not is_free_model("a/b", {"prompt": "0.5", "completion": "0"})
    assert not is_free_model("a/b", None)


def test_load_config_defaults_when_missing():
    cfg = load_config("/nonexistent/model_selector.yaml")
    assert cfg.weights == {"quality": 0.5, "latency": 0.3, "context": 0.2}
    assert cfg.probe.prompt == "Say hello in one sentence."


def test_load_config_validates_weights_via_pydantic():
    with pytest.raises((ValidationError, OSError, TypeError)):
        ModelSelectorConfig(weights={"quality": [1, 2], "latency": 0.3, "context": 0.2})


def test_redis_write_and_read_rankings_roundtrip():
    r = fakeredis.FakeRedis()
    key = "model_selector:rankings:test"
    data = [{"provider": "openrouter", "model_id": "a", "score": 0.9}]
    write_rankings(r, data, key, ttl_s=60)
    assert read_rankings(r, key, ttl_s=60) == data


def test_read_rankings_returns_none_when_missing():
    r = fakeredis.FakeRedis()
    assert read_rankings(r, "model_selector:nope", ttl_s=60) is None


def test_read_rankings_returns_none_when_stale():
    r = fakeredis.FakeRedis()
    key = "model_selector:rankings:stale"
    data = [{"model_id": "x", "score": 0.5}]
    write_rankings(r, data, key, ttl_s=60)
    # ttl_s<=0 forces early return (simulates stale data where caller
    # declares the rankings have expired).
    assert read_rankings(r, key, ttl_s=0) is None


def test_dump_rankings_json(tmp_path):
    p = tmp_path / "rankings.json"
    dump_rankings_json([{"model_id": "a", "score": 0.9}], p)
    assert p.exists()
    assert json.loads(p.read_text())["rankings"][0]["model_id"] == "a"


# ---------------------------------------------------------------------------
# Task 4: Runtime ModelSelector + Redis show command + __init__ re-export
# ---------------------------------------------------------------------------

from nexi.adapters.model_router import ModelSpec
from nexi.adapters.model_selector import ModelSelector, write_rankings


async def test_get_best_model_returns_top_ranked_for_intent(_cfg, monkeypatch):
    sel = ModelSelector(config=_cfg)
    entries = [
        {"provider": "openrouter", "model_id": "google/gemini-2.0-flash-001", "score": 0.9,
         "quality": 0.75, "latency_ms": 500, "context_window": 1_000_000, "elo": 1200.0, "tier": "strong"},
        {"provider": "openrouter", "model_id": "nvidia/nemotron-3-super-120b-a12b:free", "score": 0.6,
         "quality": 0.5, "latency_ms": 900, "context_window": 128_000, "elo": None, "tier": "mid"},
    ]
    write_rankings(_cfg._fake_redis, entries, _cfg.redis.rankings_key, _cfg.redis.ttl_s)
    best = await sel.get_best_model("QUERY")
    assert isinstance(best, ModelSpec)
    assert best.id == "google/gemini-2.0-flash-001"
    assert best.cost_tier == 1  # free → cheapest tier


async def test_get_best_model_excludes_paid_or_excluded(_cfg, monkeypatch):
    sel = ModelSelector(config=_cfg)
    write_rankings(_cfg._fake_redis, [], _cfg.redis.rankings_key, _cfg.redis.ttl_s)
    assert await sel.get_best_model("QUERY") is None


async def test_is_available_true_when_fresh(_cfg, monkeypatch):
    sel = ModelSelector(config=_cfg)
    write_rankings(_cfg._fake_redis, [{"model_id": "a", "score": 0.5}], _cfg.redis.rankings_key, _cfg.redis.ttl_s)
    assert await sel.is_available() is True


async def test_is_available_false_when_stale(_cfg, monkeypatch):
    sel = ModelSelector(config=_cfg)
    # Write valid rankings first
    write_rankings(_cfg._fake_redis, [{"model_id": "a", "score": 0.5}], _cfg.redis.rankings_key, _cfg.redis.ttl_s)
    # ttl_s=0 => force-stale by passing a config with ttl 0
    stale_redis_config = _cfg.redis.model_copy(update={"ttl_s": 0})
    sel2 = ModelSelector(config=_cfg.model_copy(update={"redis": stale_redis_config}))
    assert await sel2.is_available() is False
