"""Dynamic model selection for the OpenCode Go (hosted DeepSeek V4) backend.

Nexi's inference is not pinned to a single model. This router picks the best
available opencode-go model for each request based on:

- *task* — the intent class (DECISION/EXECUTION need strong reasoning;
  QUERY/ESCALATION can use a faster/cheaper model),
- *price* — a configurable cost budget (cheap | balanced | quality),
- *need* — fall back across the catalog when the preferred model errors.

The catalog is the set of models the opencode-go subscription exposes. It is a
built-in default (overridable via ``settings.opencode_go_models``) so the persona
can describe live model routing without hardcoding a model id anywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    id: str
    cost_tier: int  # 1 = cheap/fast, 3 = premium/reasoning
    context_window: int
    strengths: set[str] = field(default_factory=set)  # intent classes served well
    latency_ms: int = 0
    description: str = ""


# Default OpenCode Go (hosted DeepSeek V4) catalog.
DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec(
        "deepseek-v4-pro", cost_tier=3, context_window=128_000,
        strengths={"DECISION", "EXECUTION"}, latency_ms=1_800,
        description="Flagship reasoning model for high-stakes planning.",
    ),
    ModelSpec(
        "deepseek-v4-reasoner", cost_tier=3, context_window=128_000,
        strengths={"DECISION", "EXECUTION", "ESCALATION"}, latency_ms=2_200,
        description="Extended-chain reasoning for ambiguous / escalation cases.",
    ),
    ModelSpec(
        "deepseek-v4-lite", cost_tier=1, context_window=64_000,
        strengths={"QUERY", "ESCALATION"}, latency_ms=400,
        description="Fast, cheap model for retrieval and classification.",
    ),
]

# Per-intent ordered [preferred, fallback] model ids.
_INTENT_TIERS: dict[str, list[str]] = {
    "EXECUTION": ["deepseek-v4-pro", "deepseek-v4-reasoner"],
    "DECISION": ["deepseek-v4-pro", "deepseek-v4-reasoner"],
    "ESCALATION": ["deepseek-v4-reasoner", "deepseek-v4-lite"],
    "QUERY": ["deepseek-v4-lite", "deepseek-v4-pro"],
}


def _registry() -> dict[str, ModelSpec]:
    """Model catalog: built-in defaults, optionally extended by settings."""
    models = {m.id: m for m in DEFAULT_MODELS}
    raw = getattr(settings, "opencode_go_models", None)
    if raw:
        try:
            for item in raw:
                models[item["id"]] = ModelSpec(
                    id=item["id"],
                    cost_tier=int(item.get("cost_tier", 2)),
                    context_window=int(item.get("context_window", 64_000)),
                    strengths=set(item.get("strengths", [])),
                    latency_ms=int(item.get("latency_ms", 0)),
                    description=item.get("description", ""),
                )
        except Exception as exc:
            logger.warning("Failed to parse opencode_go_models override: %s", exc)
    return models


def select_model(intent_class: str, budget: str = "balanced") -> ModelSpec:
    """Choose the best opencode-go model for ``intent_class`` under ``budget``.

    ``budget`` is one of ``cheap`` (lowest cost tier), ``quality`` (highest
    cost tier), or ``balanced`` (preferred model for the intent, then fallback).
    """
    registry = _registry()
    ordered = _INTENT_TIERS.get(intent_class, [settings.model_id])
    candidates = [registry[i] for i in ordered if i in registry]
    if not candidates:
        fallback = registry.get(settings.model_id) or next(iter(registry.values()))
        return fallback

    if budget == "cheap":
        candidates.sort(key=lambda m: m.cost_tier)
    elif budget == "quality":
        candidates.sort(key=lambda m: -m.cost_tier)
    # balanced: keep intent-preferred ordering (best first)
    return candidates[0]


def fallback_chain(intent_class: str) -> list[str]:
    """Ordered model ids to try for an intent (preferred first)."""
    registry = _registry()
    tiers = _INTENT_TIERS.get(intent_class, [settings.model_id])
    chain = [i for i in tiers if i in registry]
    if not chain and settings.model_id in registry:
        chain = [settings.model_id]
    return chain


def routing_summary() -> str:
    """Human-readable description of the live routing policy for the persona."""
    parts = []
    for intent, chain in _INTENT_TIERS.items():
        parts.append(f"{intent}→{chain[0]}")
    return "; ".join(parts)


def available_models() -> list[str]:
    """Ids of models the opencode-go subscription exposes (for the persona)."""
    return list(_registry().keys())
