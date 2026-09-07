"""Free-model ranking core shared by the selector CLI and the runtime module.

The CLI discovers models, probes latency and writes rankings to Redis; the
runtime module reads those rankings and returns the best free model for an
intent. This module holds the pure scoring used by both.
"""

from __future__ import annotations

import logging
import math

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
