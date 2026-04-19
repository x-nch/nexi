"""Step 5 — Constrained option generation via Model Adapter."""
from ..adapters.model_adapter import ModelAdapter
from ..models import SessionContext, Intent, ContextManifest, PlanOption
from ..models.options import GenerationPath
from ..utils.audit import emit_event


def _build_context_summary(manifest: ContextManifest) -> dict:
    outcomes = {"S": 0, "P": 0, "F": 0}
    for ep in manifest.episodes:
        if ep.outcome == "SUCCESS":
            outcomes["S"] += 1
        elif ep.outcome == "PARTIAL":
            outcomes["P"] += 1
        elif ep.outcome == "FAILURE":
            outcomes["F"] += 1

    dominant = None
    if manifest.patterns:
        dominant = max(manifest.patterns, key=lambda p: p.confidence)

    return {
        "recent_outcomes": f"{outcomes['S']}S/{outcomes['P']}P/{outcomes['F']}F",
        "dominant_pattern": (
            f"{dominant.success_rate:.2f} success (conf={dominant.confidence:.2f})"
            if dominant else "no pattern"
        ),
    }


async def generate_options(
    adapter: ModelAdapter,
    session: SessionContext,
    intent: Intent,
    manifest: ContextManifest,
    n: int = 5,
) -> tuple[list[PlanOption], GenerationPath]:
    emit_event(session.trace_id, "option_generator", "GENERATION_START",
               {"n": n, "intent_class": intent.intent_class})

    context_summary = _build_context_summary(manifest)

    options, path = await adapter.generate_options(
        intent_class=intent.intent_class,
        target_entity_id=intent.target_entity_id,
        target_entity_class=intent.target_entity_class,
        context_summary=context_summary,
        n=n,
    )

    emit_event(session.trace_id, "option_generator", "GENERATION_COMPLETE",
               {"options_count": len(options), "path": path})
    return options, path
