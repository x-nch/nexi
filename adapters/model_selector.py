"""Free-model ranking core shared by the selector CLI and the runtime module.

The CLI discovers models, probes latency and writes rankings to Redis; the
runtime module reads those rankings and returns the best free model for an
intent. This module holds the pure scoring used by both.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path

import redis as redis_lib
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: dict[str, float] = {"quality": 0.5, "latency": 0.3, "context": 0.2}

TIER_SCORES: dict[str, float] = {"frontier": 1.0, "strong": 0.75, "mid": 0.5, "weak": 0.25}

_MID_TIER_ELO = 1200.0
_ELO_SPREAD = 300.0


def context_score(ctx: int) -> float:
    """Score a context window to [0,1], saturating at 1M tokens."""
    return min(1.0, max(0.0, ctx / 1_000_000))


def normalized_latency(ms: int, cap_ms: int = 10_000) -> float:
    """Latency to [0,1] with 0 = instant, 1 = >= cap (saturated)."""
    return min(1.0, max(0.0, ms / cap_ms))


def elo_to_score(elo: float) -> float:
    """Map an LMSYS-style elo to a (0,1] score centered on a 1200 midpoint."""
    return max(0.05, min(1.0, 1.0 - math.exp(-(elo - _MID_TIER_ELO) / _ELO_SPREAD)))


def tier_from_elo(elo: float | None, tier: str | None) -> float:
    """Quality score from elo (if numeric) else the manual tier score."""
    if elo is not None:
        normalized = elo_to_score(elo)
        if tier:
            return min(normalized, TIER_SCORES.get(tier.lower(), 0.5))
        return normalized
    return TIER_SCORES.get((tier or "mid").lower(), 0.5)


def rank_models(
    models: list[dict],
    elo: dict[str, float],
    tiers: dict[str, str],
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Score and sort a list of model dicts by composite rank (desc)."""
    w = weights or DEFAULT_WEIGHTS
    scored: list[dict] = []
    for m in models:
        mid = m["model_id"]
        elo_val = elo.get(mid)
        quality = tier_from_elo(elo_val, tiers.get(mid))
        lat = int(m.get("latency_ms", 0))
        ctx = int(m.get("context_window", 64_000))
        score = (
            w.get("quality", 0.5) * quality
            + w.get("latency", 0.3) * (1.0 - normalized_latency(lat))
            + w.get("context", 0.2) * context_score(ctx)
        )
        scored.append(
            {
                "provider": m["provider"],
                "model_id": mid,
                "score": round(score, 4),
                "quality": round(quality, 4),
                "latency_ms": lat,
                "context_window": ctx,
                "elo": round(elo_val, 1) if elo_val is not None else None,
                "tier": tiers.get(mid),
            }
        )
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored


class ProberConfig(BaseModel):
    prompt: str = "Say hello in one sentence."
    timeout_s: float = 10.0
    runs: int = 3


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379/0"
    ttl_s: int = 86_400
    rankings_key: str = "model_selector:rankings"


class ModelSelectorConfig(BaseModel):
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    probe: ProberConfig = Field(default_factory=ProberConfig)
    quality_tiers: dict[str, float] = Field(default_factory=lambda: dict(TIER_SCORES))
    manual_tiers: dict[str, str] = Field(default_factory=dict)
    redis: RedisConfig = Field(default_factory=RedisConfig)


_redis_client: "redis_lib.Redis | None" = None


def redis_client_from(config: ModelSelectorConfig) -> "redis_lib.Redis":
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    _redis_client = redis_lib.from_url(config.redis.url, decode_responses=True)
    return _redis_client


def load_config(path: str | None = None) -> ModelSelectorConfig:
    """Load YAML config over defaults.  Missing/corrupt file degrades to defaults."""
    path = path or os.environ.get("NEXI_MODEL_SELECTOR_CONFIG", "") or None
    cfg = ModelSelectorConfig()
    if not path or not os.path.exists(path):
        if path:
            logger.warning("model-selector config %s not found; using defaults", path)
        return cfg
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = ModelSelectorConfig.model_validate(raw)
    except Exception as exc:
        logger.warning("Failed to load model-selector config %s: %s; using defaults", path, exc)
    return cfg


def is_free_model(model_id: str, pricing: dict | None) -> bool:
    """True for OpenRouter free models: ``:free`` suffix or zero prompt price."""
    if model_id.endswith(":free"):
        return True
    if not pricing:
        return False
    try:
        return float(pricing.get("prompt", 1)) == 0.0
    except (TypeError, ValueError):
        return False


def write_rankings(client: "redis_lib.Redis", rankings: list[dict], key: str, ttl_s: int) -> None:
    client.set(key, json.dumps(rankings), ex=ttl_s)


def read_rankings(client: "redis_lib.Redis", key: str, ttl_s: int) -> list[dict] | None:
    """Return rankings or None if missing, stale (ttl_s<=0), or unparseable."""
    if ttl_s <= 0:
        return None
    try:
        raw = client.get(key)
    except Exception as exc:
        logger.warning("Redis read failed for %s: %s", key, exc)
        return None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Bad rankings JSON under %s: %s", key, exc)
        return None


def dump_rankings_json(rankings: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_at": "", "rankings": rankings}
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
