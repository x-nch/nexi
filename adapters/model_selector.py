"""Free-model ranking core shared by the selector CLI and the runtime module.

The CLI discovers models, probes latency and writes rankings to Redis; the
runtime module reads those rankings and returns the best free model for an
intent. This module holds the pure scoring used by both.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import os
import time
from pathlib import Path

import httpx
import redis as redis_lib
import yaml
from pydantic import BaseModel, Field

from .model_router import PROVIDER_OPENROUTER, ModelSpec

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


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _safe_context(item: dict) -> int:
    for key in ("context_length", "max_context_length", "context_window"):
        raw = item.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return 64_000


async def discover_opencode(client: httpx.AsyncClient, api_url: str) -> list[dict]:
    """List models exposed by an OpenCode Go endpoint (auth'd subscription).

    Returns ``[]`` when the upstream is unreachable or errors, so one dead
    provider never aborts a run.
    """
    try:
        resp = await client.get(f"{api_url}/models")
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("opencode discovery failed at %s: %s", api_url, exc)
        return []
    if isinstance(data, list):
        models = data
    elif isinstance(data, dict):
        models = data.get("data") or data.get("models") or []
    else:
        return []
    out = []
    for item in models:
        if not isinstance(item, dict):
            continue
        mid = item.get("id") or item.get("model")
        if not mid:
            continue
        out.append(
            {
                "provider": "opencode",
                "model_id": mid,
                "context_window": _safe_context(item),
                "latency_ms": 0,
            }
        )
    return out


async def discover_openrouter(client: httpx.AsyncClient, api_url: str) -> list[dict]:
    """List OpenRouter free models only.

    Returns ``[]`` when the upstream is unreachable or errors.
    """
    try:
        resp = await client.get(f"{api_url}/models")
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception as exc:
        logger.warning("openrouter discovery failed at %s: %s", api_url, exc)
        return []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("data", [])
    else:
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not mid or not is_free_model(mid, item.get("pricing")):
            continue
        out.append(
            {
                "provider": "openrouter",
                "model_id": mid,
                "context_window": _safe_context(item),
                "latency_ms": 0,
            }
        )
    return out


async def fetch_elo(client: httpx.AsyncClient) -> dict[str, float]:
    """Fetch LMSYS chatbot-arena elo by model name (best-effort; {} on failure)."""
    url = "https://huggingface.co/api/datasets/lmsys/chatbot-arena-leaderboard/parquet"
    try:
        resp = await client.get(url)
        resp.raise_for_status()
        rows = resp.json()
        if not isinstance(rows, list):
            logger.warning("ELO fetch returned %s, expected a list of rows", type(rows).__name__)
            return {}
        elo: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("model_name") or row.get("model")
            val = row.get("elo")
            if name and isinstance(val, (int, float)):
                elo[name] = float(val)
        return elo
    except Exception as exc:
        logger.warning("ELO fetch failed: %s", exc)
        return {}


async def probe_latency(
    client: httpx.AsyncClient,
    provider: str,
    model_id: str,
    base_url: str,
    api_key: str,
    prompt: str,
    timeout_s: float,
    runs: int,
) -> dict:
    """Send a minimal completion and return ttft/total medians (ms).

    ``ttft_ms`` is the median time to the first streamed chunk and ``total_ms``
    the median full-response time across ``runs``; ``{ok: False}`` on any
    transport or HTTP error.
    """
    samples_total: list[float] = []
    samples_ttft: list[float] = []
    for _ in range(max(1, runs)):
        started = time.monotonic()
        first_chunk_at: float | None = None
        try:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 16,
                    "stream": True,
                },
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout_s,
            ) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                    if b"data: [DONE]" in chunk:
                        break
            done = time.monotonic()
        except Exception as exc:
            logger.warning("probe %s/%s failed: %s", provider, model_id, exc)
            return {
                "provider": provider,
                "model_id": model_id,
                "ok": False,
                "ttft_ms": None,
                "total_ms": None,
                "probed_at": _dt.datetime.now().isoformat(),
            }
        samples_total.append((done - started) * 1000)
        first_at = first_chunk_at if first_chunk_at is not None else done
        samples_ttft.append((first_at - started) * 1000)

    return {
        "provider": provider,
        "model_id": model_id,
        "ok": True,
        "ttft_ms": round(_median(samples_ttft), 1),
        "total_ms": round(_median(samples_total), 1),
        "probed_at": _dt.datetime.now().isoformat(),
    }


INTENT_STRENGTHS: dict[str, set[str]] = {
    "DECISION": {"DECISION", "EXECUTION"},
    "EXECUTION": {"EXECUTION"},
    "QUERY": {"QUERY", "ESCALATION"},
    "ESCALATION": {"QUERY", "ESCALATION"},
}

_FREE_TIER = 1  # free models always rank cheapest for cost budgeting


def _ranking_to_model_spec(entry: dict) -> ModelSpec:
    return ModelSpec(
        id=entry["model_id"],
        cost_tier=_FREE_TIER,
        context_window=int(entry.get("context_window", 64_000)),
        strengths=INTENT_STRENGTHS.get("QUERY", {"QUERY", "ESCALATION"}).copy(),
        latency_ms=int(entry.get("latency_ms", 0)),
        description=(
            f"free {entry.get('provider', '?')} model, elo={entry.get('elo')}, "
            f"tier={entry.get('tier')}, score={entry.get('score')}"
        ),
    )


class ModelSelector:
    """Reads Redis rankings and returns the best free model for an intent."""

    def __init__(self, config: ModelSelectorConfig | None = None, redis=None):
        self._config = config or load_config()
        self._redis = redis or redis_client_from(self._config)

    async def get_best_model(
        self,
        intent: str,
        budget: str = "balanced",
        exclude: list[str] | None = None,
    ) -> ModelSpec | None:
        ranked = await self.get_ranked_models(intent, limit=50)
        excluded = set(exclude or [])
        for model_spec in ranked:
            if model_spec.id in excluded:
                continue
            return model_spec
        return None

    async def get_ranked_models(self, intent: str, limit: int = 5) -> list[ModelSpec]:
        ranked = read_rankings(self._redis, self._config.redis.rankings_key, self._config.redis.ttl_s)
        if not ranked:
            return []
        # Take top 'limit' models by score (rankings are already sorted descending by score)
        return [_ranking_to_model_spec(entry) for entry in ranked[:limit]]

    async def is_available(self) -> bool:
        return bool(read_rankings(self._redis, self._config.redis.rankings_key, self._config.redis.ttl_s))
