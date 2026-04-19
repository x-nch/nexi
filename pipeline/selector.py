"""Step 9 — Decision selection and Decision Record assembly."""
from uuid import UUID

from ..models import (
    SessionContext, Intent, ContextManifest,
    PlanOption, EvaluatedOption, DecisionRecord,
)
from ..models.options import SelectionRationale, GenerationPath
from ..utils.audit import emit_event


def select_decision(
    session: SessionContext,
    intent: Intent,
    manifest: ContextManifest,
    options: list[PlanOption],
    evaluated: list[EvaluatedOption],
    n_generated: int,
    n_blocked: int,
    generation_path: GenerationPath,
) -> DecisionRecord:
    ranked = sorted(evaluated, key=lambda e: e.composite_score, reverse=True)

    if not ranked:
        return _escalated_record(session, intent, manifest, evaluated, n_generated, n_blocked, generation_path)

    best = ranked[0]
    second_best = ranked[1] if len(ranked) > 1 else None
    confidence = best.composite_score - (second_best.composite_score if second_best else 0.0)

    emit_event(session.trace_id, "selector", "OPTION_SELECTED",
               {"option_id": str(best.option_id), "composite": best.composite_score,
                "confidence": round(confidence, 4)})

    return DecisionRecord(
        session_id=session.session_id,
        intent_ref=intent.intent_id,
        context_manifest_ref=manifest.manifest_id,
        system_state_version=session.system_state_version,
        options_generated=n_generated,
        options_blocked=n_blocked,
        options_evaluated=evaluated,
        selected_option_id=best.option_id,
        selection_rationale=SelectionRationale(
            score_breakdown=best.scores.model_dump(),
            weight_config_version=best.weight_config_version,
        ),
        confidence=round(confidence, 4),
        escalation_triggered=False,
        generation_path=generation_path,
    )


def _escalated_record(
    session: SessionContext,
    intent: Intent,
    manifest: ContextManifest,
    evaluated: list[EvaluatedOption],
    n_generated: int,
    n_blocked: int,
    generation_path: GenerationPath,
) -> DecisionRecord:
    emit_event(session.trace_id, "selector", "ESCALATION_TRIGGERED")
    return DecisionRecord(
        session_id=session.session_id,
        intent_ref=intent.intent_id,
        context_manifest_ref=manifest.manifest_id,
        system_state_version=session.system_state_version,
        options_generated=n_generated,
        options_blocked=n_blocked,
        options_evaluated=evaluated,
        selected_option_id=None,
        selection_rationale=SelectionRationale(
            score_breakdown={},
            weight_config_version="n/a",
        ),
        confidence=0.0,
        escalation_triggered=True,
        generation_path=generation_path,
    )
