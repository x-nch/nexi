"""Dynamic, multi-provider model selection for Nexi.

Nexi's inference is not pinned to a single model or provider. This router
resolves each request to a concrete ``{provider, model_id, base_url, api_key}``
target based on:

- *task* — the intent class (DECISION/EXECUTION need strong reasoning;
  QUERY/ESCALATION can use a faster/cheaper model),
- *price* — a configurable cost budget (cheap | balanced | quality),
- *provider* — which backend serves the request (opencode | openrouter |
  nexi-default), resolved per call site,
- *need* — fall back across the catalog when the preferred model errors.

Nexi is the default decision-maker for chat + internals: intent classification,
option generation, reflection and the evaluator route through this router, and
``nexi-default`` aliases to the configured default provider (opencode or
openrouter) at call time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import settings

logger = logging.getLogger(__name__)

# Provider ids understood by the router / LLM client.
PROVIDER_OPENCODE = "opencode"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_LITELLM = "litellm"
PROVIDER_NEXI_DEFAULT = "nexi-default"
DEFAULT_PROVIDERS = (PROVIDER_OPENCODE, PROVIDER_OPENROUTER, PROVIDER_LITELLM, PROVIDER_NEXI_DEFAULT)


@dataclass
class ModelSpec:
    id: str
    cost_tier: int = 2  # 1 = cheap/fast, 3 = premium/reasoning
    context_window: int = 64_000
    strengths: set[str] = field(default_factory=set)  # intent classes served well
    latency_ms: int = 0
    description: str = ""
    provider: str = ""


@dataclass(frozen=True)
class ModelResolution:
    """A concrete inference target — enough to POST a /chat/completions."""

    provider: str
    model_id: str
    base_url: str
    api_key: str = ""
    timeout_s: float = 60.0


# --- Provider catalogs --------------------------------------------------------

# Default OpenCode Go (hosted DeepSeek V4) catalog.
DEFAULT_OPENCODE_MODELS: list[ModelSpec] = [
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

# Default OpenRouter catalog. Keep the model ids stable ("author/model") for a
# ready-to-run default; override via NEXI_OPENROUTER_MODELS.
DEFAULT_OPENROUTER_MODELS: list[ModelSpec] = [
    ModelSpec(
        "anthropic/claude-sonnet-4", cost_tier=3, context_window=200_000,
        strengths={"DECISION", "EXECUTION", "QUERY", "ESCALATION"}, latency_ms=1_500,
        description="Claude Sonnet — strong general reasoning and tool use.",
    ),
    ModelSpec(
        "openai/gpt-4o", cost_tier=3, context_window=128_000,
        strengths={"DECISION", "EXECUTION"}, latency_ms=1_200,
        description="GPT-4o — general reasoning for high-stakes planning.",
    ),
    ModelSpec(
        "google/gemini-2.0-flash-001", cost_tier=1, context_window=1_000_000,
        strengths={"QUERY", "ESCALATION"}, latency_ms=500,
        description="Gemini Flash — fast, cheap for retrieval and classification.",
    ),
]

# Default litellm (local vLLM proxy) catalog. The proxy itself maps the
# model name (ornith) to concrete backends; override via NEXI_LITELLM_MODELS.
DEFAULT_LITELLM_MODELS: list[ModelSpec] = [
    ModelSpec(
        "ornith", cost_tier=2, context_window=64_000,
        strengths={"DECISION", "EXECUTION", "QUERY", "ESCALATION"}, latency_ms=800,
        description="Local ornith vLLM served via the LiteLLM proxy.",
    ),
]

# Per-intent ordered [preferred, fallback] model ids, keyed by provider. Ids
# must exist in the provider's own catalog.
_INTENT_TIERS: dict[str, dict[str, list[str]]] = {
    PROVIDER_OPENCODE: {
        "EXECUTION": ["deepseek-v4-pro", "deepseek-v4-reasoner"],
        "DECISION": ["deepseek-v4-pro", "deepseek-v4-reasoner"],
        "ESCALATION": ["deepseek-v4-reasoner", "deepseek-v4-lite"],
        "QUERY": ["deepseek-v4-lite", "deepseek-v4-pro"],
    },
    PROVIDER_OPENROUTER: {
        "EXECUTION": ["anthropic/claude-sonnet-4", "openai/gpt-4o"],
        "DECISION": ["anthropic/claude-sonnet-4", "openai/gpt-4o"],
        "ESCALATION": ["openai/gpt-4o", "google/gemini-2.0-flash-001"],
        "QUERY": ["google/gemini-2.0-flash-001", "anthropic/claude-sonnet-4"],
    },
    PROVIDER_LITELLM: {
        "EXECUTION": ["ornith"],
        "DECISION": ["ornith"],
        "ESCALATION": ["ornith"],
        "QUERY": ["ornith"],
    },
}


def _intent_tiers(provider: str) -> dict[str, list[str]]:
    return _INTENT_TIERS.get(provider, _INTENT_TIERS[PROVIDER_OPENCODE])


def _catalog_from_settings(raw: Any, defaults: list[ModelSpec]) -> dict[str, ModelSpec]:
    # An explicit settings override replaces the built-in catalog entirely.
    models: dict[str, ModelSpec] = {}
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
            logger.warning("Failed to parse a model-catalog override: %s", exc)
            models = {}
    if not models:
        models = {m.id: m for m in defaults}
    return models


def _registry(provider: str) -> dict[str, ModelSpec]:
    """Model catalog for a provider: built-in defaults + settings override."""
    if provider == PROVIDER_OPENROUTER:
        return _catalog_from_settings(
            getattr(settings, "openrouter_models", None), DEFAULT_OPENROUTER_MODELS
        )
    if provider == PROVIDER_LITELLM:
        return _catalog_from_settings(
            getattr(settings, "litellm_models", None), DEFAULT_LITELLM_MODELS
        )
    # opencode (and nexi-default → resolved upstream) use the opencode catalog.
    return _catalog_from_settings(
        getattr(settings, "opencode_go_models", None), DEFAULT_OPENCODE_MODELS
    )


def _resolve_provider(provider: str | None) -> str:
    """Resolve ``provider`` to a real backend id.

    ``None``/``nexi-default`` → ``settings.nexi_default_resolves_to`` (or the
    legacy default ``settings.model_id`` family behavior via opencode). When a
    LiteLLM proxy is configured (``settings.litellm_proxy_url``) the default
    resolves to ``litellm`` so the local vLLM model serves chat + internals.
    """
    requested = provider or settings.default_provider
    if requested in (PROVIDER_NEXI_DEFAULT, "auto", ""):
        if settings.litellm_proxy_url:
            return PROVIDER_LITELLM
        resolved = (
            settings.nexi_default_resolves_to
            if settings.nexi_default_resolves_to
            else PROVIDER_OPENCODE
        )
        if resolved not in DEFAULT_PROVIDERS:
            logger.warning(
                "nexi_default_resolves_to=%r not a known provider; using opencode",
                resolved,
            )
            return PROVIDER_OPENCODE
        return resolved
    if requested in DEFAULT_PROVIDERS:
        return requested
    logger.warning("Unknown provider %r; defaulting to opencode", requested)
    return PROVIDER_OPENCODE


def select_model(intent_class: str, budget: str = "balanced") -> ModelSpec:
    """Choose the best model (within the default provider) for an intent.

    Backwards-compatible with the legacy single-provider API.
    """
    provider = _resolve_provider(settings.default_provider)
    return select_model_for_provider(intent_class, provider, budget)


def select_model_for_provider(intent_class: str, provider: str, budget: str = "balanced") -> ModelSpec:
    """Choose the best model for ``intent_class`` under ``budget`` on ``provider``."""
    registry = _registry(provider)
    ordered = _intent_tiers(provider).get(intent_class, [])
    candidates = [registry[i] for i in ordered if i in registry]
    if not candidates:
        candidates = list(registry.values())
    if not candidates:
        candidates = [ModelSpec(id=settings.model_id)]
    if budget == "cheap":
        candidates.sort(key=lambda m: m.cost_tier)
    elif budget == "quality":
        candidates.sort(key=lambda m: -m.cost_tier)
    # balanced: keep intent-preferred ordering (best first)
    return candidates[0]


def fallback_chain(intent_class: str, provider: str | None = None) -> list[str]:
    """Ordered model ids to try for an intent (preferred first) on a provider."""
    provider = _resolve_provider(provider)
    registry = _registry(provider)
    tiers = _intent_tiers(provider).get(intent_class, [])
    chain = [i for i in tiers if i in registry]
    if not chain:
        chain = [m.id for m in sorted(registry.values(), key=lambda m: -m.cost_tier)]
    if not chain:
        chain = [settings.model_id]
    return chain


def resolve(
    intent_class: str,
    budget: str = "balanced",
    provider: str | None = None,
    model_id: str | None = None,
) -> ModelResolution:
    """Resolve a request to a concrete inference target.

    ``provider``/``model_id`` are explicit overrides (per-agent / per-step). When
    ``model_id`` is given, it is used verbatim within the resolved provider.
    """
    provider = _resolve_provider(provider)
    if model_id:
        spec = ModelSpec(id=model_id)
    else:
        spec = select_model_for_provider(intent_class, provider, budget)

    if provider == PROVIDER_OPENROUTER:
        return ModelResolution(
            provider=provider,
            model_id=spec.id,
            base_url=settings.openrouter_api_url,
            api_key=settings.openrouter_api_key,
            timeout_s=settings.openrouter_api_timeout_s,
        )
    if provider == PROVIDER_LITELLM:
        return ModelResolution(
            provider=provider,
            model_id=spec.id,
            base_url=settings.litellm_proxy_url,
            api_key=settings.litellm_api_key,
            timeout_s=settings.litellm_proxy_timeout_s,
        )
    return ModelResolution(
        provider=PROVIDER_OPENCODE,
        model_id=spec.id,
        base_url=settings.opencode_go_api_url,
        api_key=settings.opencode_go_api_key,
        timeout_s=settings.opencode_go_api_timeout_s,
    )


def routing_summary() -> str:
    """Human-readable description of the live routing policy for the persona."""
    provider = _resolve_provider(settings.default_provider)
    tiers = _intent_tiers(provider)
    parts = []
    for intent in ("QUERY", "DECISION", "EXECUTION", "ESCALATION"):
        chain = tiers.get(intent, [])
        if chain:
            parts.append(f"{intent}→{chain[0]}")
    return f"provider={provider}; " + "; ".join(parts)


def available_models(provider: str | None = None) -> list[str]:
    """Ids of models a provider exposes (for the persona)."""
    provider = _resolve_provider(provider)
    return list(_registry(provider).keys())
