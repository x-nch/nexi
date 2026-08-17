"""Shared single pipeline pass — extracted from session_start for reuse by the goal loop."""
import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from fastapi import HTTPException

from ..adapters import XnchClient, ModelAdapter
from ..config import settings
from ..models import SessionContext
from ..pipeline.intent_interpreter import IntentInterpreter, ClarificationRequired
from ..pipeline.context_loader import load_context
from ..pipeline.option_generator import generate_options
from ..pipeline.policy_filter import PolicyFilter, AllOptionsBlocked
from ..pipeline.evaluator import Evaluator
from ..pipeline.selector import select_decision
from ..pipeline.plan_compiler import compile_action_spec, PlanCompilationError
from ..pipeline.dispatch import dispatch_execution, TokenExpired
from ..utils.audit import emit_event

logger = logging.getLogger(__name__)


@dataclass
class PipelinePassResult:
    status: str  # "EXECUTING" | "ESCALATED"
    decision_id: UUID | None = None
    execution_ref: UUID | None = None
    audit_ref: UUID | None = None
    hold_id: UUID | None = None
    estimated_completion_ms: int | None = None


async def run_pipeline_pass(
    *,
    xnch: XnchClient,
    model_adapter: ModelAdapter,
    policy_filter: PolicyFilter,
    intent_interpreter: IntentInterpreter,
    session: SessionContext,
    raw_input: str,
    simulation: dict[str, Any] | None = None,
    goal_id: UUID | None = None,
) -> PipelinePassResult:
    """Run one full decision pipeline pass (interpret → dispatch)."""
    # Step 3 — Intent interpretation
    try:
        intent = await intent_interpreter.interpret(
            raw_input, session.session_id, str(session.trace_id)
        )
    except ClarificationRequired:
        raise

    # Step 4 — Context manifest (hard stop on failure)
    try:
        manifest = await load_context(xnch, session, intent)
    except Exception as exc:
        logger.error("Context manifest load failed: %s", exc)
        raise HTTPException(status_code=503, detail="DEGRADED: context manifest unavailable")

    # Fetch weight config for this intent class (cached per session)
    try:
        weight_config = await xnch.get_weight_config(intent.intent_class)
    except Exception:
        weight_config = None

    evaluator = Evaluator(weight_config)

    # Step 5 — Option generation
    raw_options, generation_path = await generate_options(
        model_adapter, session, intent, manifest, settings.options_count
    )

    # Step 6 — Policy alignment filter
    try:
        surviving = await policy_filter.filter(session, raw_options)
    except AllOptionsBlocked:
        hold_id = uuid4()
        emit_event(str(session.trace_id), "nexi", "ESCALATED_ALL_BLOCKED",
                   {"hold_id": str(hold_id)})
        return PipelinePassResult(status="ESCALATED", hold_id=hold_id)

    # Step 7 — Scoring
    evaluated = evaluator.score(surviving, intent, manifest, session)

    # Step 8 — Outcome simulation (conditional)
    evaluated = evaluator.simulate_and_rescore(evaluated, surviving, manifest, intent, session)

    # Step 9 — Selection
    n_blocked = len(raw_options) - len(surviving)
    decision = select_decision(
        session, intent, manifest,
        [opt for opt, _ in surviving], evaluated,
        n_generated=len(raw_options),
        n_blocked=n_blocked,
        generation_path=generation_path,
    )

    if decision.escalation_triggered:
        hold_id = uuid4()
        return PipelinePassResult(status="ESCALATED", hold_id=hold_id)

    # Step 10a — Plan compilation
    selected_opt = next(
        (opt for opt, _ in surviving if opt.option_id == decision.selected_option_id),
        None,
    )
    if selected_opt is None:
        raise HTTPException(status_code=500, detail="Selected option not found")

    try:
        compiled = compile_action_spec(selected_opt)
    except PlanCompilationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if not compiled.nodes:
        raise HTTPException(status_code=422, detail="compiled DAG has no nodes")
    node = compiled.nodes[0]
    validated_action_spec = {
        "type": node.action_type,
        "target": node.target,
        "params": node.params,
    }

    # Resolve intent/entity context and predicted score for episode creation
    intent_class = intent.intent_class.value if intent else ""
    entity_class = intent.target_entity_class if intent else ""
    opt_scores = {
        eo.option_id: eo.composite_score
        for eo in decision.options_evaluated
    }
    outcome_score_predicted = opt_scores.get(decision.selected_option_id, 0.5)

    # Step 10 — Final verdict
    try:
        verdict = await xnch.submit_verdict(
            session, decision, validated_action_spec, selected_opt.payload_hash,
            intent_class=intent_class,
            entity_class=entity_class,
            outcome_score_predicted=outcome_score_predicted,
            goal_id=goal_id,
        )
    except Exception as exc:
        error_body = str(exc)
        if "STALE_SESSION" in error_body:
            # Re-read context to get fresh system_state_version, then retry once
            try:
                fresh_manifest = await load_context(xnch, session, intent)
                fresh_version = fresh_manifest.system_state_version
                # Update session with fresh version
                session.system_state_version = fresh_version
                verdict = await xnch.submit_verdict(
                    session, decision, validated_action_spec, selected_opt.payload_hash,
                    intent_class=intent_class,
                    entity_class=entity_class,
                    outcome_score_predicted=outcome_score_predicted,
                    goal_id=goal_id,
                )
            except Exception as retry_exc:
                raise HTTPException(status_code=409, detail=f"STALE_SESSION: retry failed: {retry_exc}")
        else:
            raise HTTPException(status_code=502, detail=f"Verdict submission failed: {error_body}")

    if verdict.verdict == "BLOCK":
        hold_id = uuid4()
        return PipelinePassResult(status="ESCALATED", hold_id=hold_id)

    # Step 11 — Execution dispatch (async handoff)
    execution_runner_url = settings.execution_runner_url
    try:
        dispatch_payload = await dispatch_execution(
            session, decision, verdict, validated_action_spec, execution_runner_url,
            simulation=simulation, goal_id=goal_id,
        )
    except TokenExpired:
        # Resubmit to xnch for a new token, same decision_id
        verdict = await xnch.submit_verdict(
            session, decision, validated_action_spec, selected_opt.payload_hash,
            intent_class=intent_class,
            entity_class=entity_class,
            outcome_score_predicted=outcome_score_predicted,
            goal_id=goal_id,
        )
        dispatch_payload = await dispatch_execution(
            session, decision, verdict, validated_action_spec, execution_runner_url,
            simulation=simulation, goal_id=goal_id,
        )

    # Step 12 — Intermediate response to user
    emit_event(str(session.trace_id), "nexi", "EXECUTING",
               {"execution_ref": str(dispatch_payload.execution_ref)})

    # Step 14 callback is handled async when execution-runner posts to xnch /execution/outcome
    # and xnch fires our callback at POST /callback/outcome

    return PipelinePassResult(
        status="EXECUTING",
        decision_id=decision.decision_id,
        execution_ref=dispatch_payload.execution_ref,
        estimated_completion_ms=_estimate_completion_ms(manifest),
        audit_ref=verdict.audit_ref,
    )


def _estimate_completion_ms(manifest) -> int:
    if not manifest.episodes:
        return 30_000
    completed = [ep for ep in manifest.episodes if ep.duration_ms]
    if not completed:
        return 30_000
    avg = sum(ep.duration_ms for ep in completed) / len(completed)
    return int(avg)
