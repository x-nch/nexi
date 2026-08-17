"""Nexi v0 — decision engine FastAPI application."""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse as JSONResponse
from pydantic import BaseModel

from .adapters import XnchClient, ModelAdapter
from .character.capability_builder import (
    build_capabilities,
    fetch_tool_inventory,
    get_generated_overlay_path,
    render_overlay,
    write_overlay,
)
from .character.prompt_loader import load_capabilities
from .config import settings
from .infra.discovery import build_snapshot, probe_services
from .models import SessionContext
from .pipeline import (
    IntentInterpreter, ClarificationRequired,
    load_context,
    generate_options,
    PolicyFilter, AllOptionsBlocked,
    Evaluator,
    select_decision,
    compile_action_spec, PlanCompilationError,
    dispatch_execution,
)
from .pipeline.dispatch import TokenExpired
from .pipeline.reflector import Reflector, build_reflector
from .utils.audit import emit_event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_xnch: XnchClient | None = None
_model_adapter: ModelAdapter | None = None
_policy_filter: PolicyFilter | None = None
_intent_interpreter: IntentInterpreter | None = None
_reflector: Reflector | None = None

# Auto-refreshed capability / infra awareness snapshot.
_capability_state: dict[str, Any] = {
    "capabilities": None,
    "snapshot": None,
    "last_refresh": None,
}


async def _refresh_capabilities(force_write: bool) -> dict[str, Any]:
    """Probe + rebuild the capabilities snapshot; optionally persist the overlay."""
    snapshot = build_snapshot()
    snapshot.status = await probe_services(snapshot)
    inventory = await fetch_tool_inventory()
    caps = build_capabilities(snapshot, inventory)

    changed = False
    if force_write:
        content = render_overlay(caps)
        changed = write_overlay(get_generated_overlay_path(), content)

    _capability_state.update(
        capabilities=caps,
        snapshot=snapshot,
        last_refresh=datetime.now(timezone.utc),
    )
    if changed:
        emit_event(
            str(uuid4()), "nexi", "CAPABILITIES_UPDATED",
            {"generated_at": caps.get("generated_at")},
        )
    return caps


async def _capability_refresh_loop() -> None:
    """Startup build then periodic full refresh + realtime probe updates."""
    try:
        await _refresh_capabilities(force_write=True)
    except Exception as exc:
        logger.warning("Initial capability refresh failed: %s", exc)

    elapsed = 0.0
    while True:
        await asyncio.sleep(settings.probe_interval_s)
        elapsed += settings.probe_interval_s
        force_write = elapsed >= settings.capability_refresh_interval_s
        if force_write:
            elapsed = 0.0
        try:
            await _refresh_capabilities(force_write=force_write)
        except Exception as exc:
            logger.warning("Periodic capability refresh failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _xnch, _model_adapter, _policy_filter, _intent_interpreter, _reflector
    _xnch = XnchClient()
    _model_adapter = ModelAdapter()
    _policy_filter = PolicyFilter(_xnch)
    _intent_interpreter = IntentInterpreter()
    _reflector = build_reflector(_xnch) if settings.reflection_enabled else None

    capability_task: asyncio.Task | None = None
    if settings.capability_auto_refresh:
        capability_task = asyncio.get_running_loop().create_task(_capability_refresh_loop())

    yield

    if capability_task is not None:
        capability_task.cancel()
        with asyncio.suppress(asyncio.CancelledError):
            await capability_task
    await _xnch.aclose()


app = FastAPI(title="Nexi", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    session_id: UUID
    trace_id: UUID
    actor: dict[str, Any]
    system_state_version: str
    policy_version: str
    raw_input: str
    priority: str = "NORMAL"
    idempotency_key: UUID


class SessionStartResponse(BaseModel):
    status: str
    decision_id: UUID | None = None
    execution_ref: UUID | None = None
    estimated_completion_ms: int | None = None
    audit_ref: UUID | None = None
    clarification_required: bool = False
    hold_id: UUID | None = None
    error: str | None = None


class ClarifyRequest(BaseModel):
    amended_input: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/session/start", response_model=SessionStartResponse)
async def session_start(
    body: SessionStartRequest,
    background_tasks: BackgroundTasks,
) -> SessionStartResponse:
    """Entry point called by xnch after actor resolution (Step 2 → Step 3)."""
    session = SessionContext.model_validate(body.model_dump())
    emit_event(session.trace_id, "nexi", "SESSION_START_RECEIVED",
               {"session_id": str(session.session_id)})

    if _intent_interpreter is None:
        return SessionStartResponse(status="ERROR", error="intent interpreter not available")

    # Step 3 — Intent interpretation
    try:
        intent = await _intent_interpreter.interpret(
            body.raw_input, session.session_id, str(session.trace_id)
        )
    except ClarificationRequired as exc:
        return SessionStartResponse(
            status="CLARIFICATION_REQUIRED",
            clarification_required=True,
        )

    # Step 4 — Context manifest (hard stop on failure)
    try:
        manifest = await load_context(_xnch, session, intent)
    except Exception as exc:
        logger.error("Context manifest load failed: %s", exc)
        raise HTTPException(status_code=503, detail="DEGRADED: context manifest unavailable")

    # Fetch weight config for this intent class (cached per session)
    try:
        weight_config = await _xnch.get_weight_config(intent.intent_class)
    except Exception:
        weight_config = None

    evaluator = Evaluator(weight_config)

    # Step 5 — Option generation
    n = settings.options_count
    raw_options, generation_path = await generate_options(
        _model_adapter, session, intent, manifest, n
    )

    # Step 6 — Policy alignment filter
    try:
        surviving = await _policy_filter.filter(session, raw_options)
    except AllOptionsBlocked:
        hold_id = uuid4()
        emit_event(str(session.trace_id), "nexi", "ESCALATED_ALL_BLOCKED",
                   {"hold_id": str(hold_id)})
        return SessionStartResponse(status="ESCALATED", hold_id=hold_id)

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
        return SessionStartResponse(status="ESCALATED", hold_id=hold_id)

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
        return JSONResponse({'error': 'compiled DAG has no nodes'}, status_code=422)
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
        verdict = await _xnch.submit_verdict(
            session, decision, validated_action_spec, selected_opt.payload_hash,
            intent_class=intent_class,
            entity_class=entity_class,
            outcome_score_predicted=outcome_score_predicted,
        )
    except Exception as exc:
        error_body = str(exc)
        if "STALE_SESSION" in error_body:
            # Re-read context to get fresh system_state_version, then retry once
            try:
                fresh_manifest = await load_context(_xnch, session, intent)
                fresh_version = fresh_manifest.system_state_version
                # Update session with fresh version
                session.system_state_version = fresh_version
                verdict = await _xnch.submit_verdict(
                    session, decision, validated_action_spec, selected_opt.payload_hash,
                    intent_class=intent_class,
                    entity_class=entity_class,
                    outcome_score_predicted=outcome_score_predicted,
                )
            except Exception as retry_exc:
                raise HTTPException(status_code=409, detail=f"STALE_SESSION: retry failed: {retry_exc}")
        else:
            raise HTTPException(status_code=502, detail=f"Verdict submission failed: {error_body}")

    if verdict.verdict == "BLOCK":
        hold_id = uuid4()
        return SessionStartResponse(status="ESCALATED", hold_id=hold_id)

    # Step 11 — Execution dispatch (async handoff)
    execution_runner_url = settings.execution_runner_url
    try:
        dispatch_payload = await dispatch_execution(
            session, decision, verdict, validated_action_spec, execution_runner_url
        )
    except TokenExpired:
        # Resubmit to xnch for a new token, same decision_id
        verdict = await _xnch.submit_verdict(
            session, decision, validated_action_spec, selected_opt.payload_hash,
            intent_class=intent_class,
            entity_class=entity_class,
            outcome_score_predicted=outcome_score_predicted,
        )
        dispatch_payload = await dispatch_execution(
            session, decision, verdict, validated_action_spec, execution_runner_url
        )

    # Step 12 — Intermediate response to user
    emit_event(str(session.trace_id), "nexi", "EXECUTING",
               {"execution_ref": str(dispatch_payload.execution_ref)})

    # Step 14 callback is handled async when execution-runner posts to xnch /execution/outcome
    # and xnch fires our callback at POST /callback/outcome

    return SessionStartResponse(
        status="EXECUTING",
        decision_id=decision.decision_id,
        execution_ref=dispatch_payload.execution_ref,
        estimated_completion_ms=_estimate_completion_ms(manifest),
        audit_ref=verdict.audit_ref,
    )


@app.post("/callback/outcome")
async def outcome_callback(body: dict[str, Any]) -> dict:
    """Step 14 — xnch fires this after writing execution outcome to episodic store."""
    trace_id = body.get("trace_id", "unknown")
    emit_event(trace_id, "nexi", "OUTCOME_CALLBACK_RECEIVED")

    # Compute prediction delta and write back to xnch
    outcome_score_predicted = body.get("outcome_score_predicted", 0.5)
    actual_success = 1.0 if body.get("outcome_status") == "SUCCESS" else 0.0
    prediction_delta = abs(outcome_score_predicted - actual_success)
    early_flag = prediction_delta > 0.3

    # Retry with backoff handled by caller if this fails
    session_id = body.get("session_id")
    episode_id = body.get("episode_id")
    if session_id and episode_id:
        try:
            # Build minimal session for the write call
            actor_data = body.get("actor", {"id": "system", "role": "AGENT", "capability_set": []})
            minimal_session = SessionContext(
                session_id=UUID(session_id),
                trace_id=UUID(trace_id) if trace_id != "unknown" else uuid4(),
                actor=actor_data,
                system_state_version=body.get("system_state_version", ""),
                policy_version=body.get("policy_version", ""),
                idempotency_key=uuid4(),
                raw_input="",
            )
            await _xnch.write_prediction_update(
                minimal_session, UUID(episode_id), prediction_delta, early_flag
            )
        except Exception as exc:
            logger.error("Memory write failed (will retry): %s", exc)
            # TODO: enqueue for exponential backoff retry (max 5 attempts)

    # Summary step — reflect on the outcome and persist an experiential lesson.
    # Fire-and-forget: reflection must never block or break the outcome callback.
    intent_class = body.get("intent_class")
    if (
        _reflector is not None
        and settings.reflection_enabled
        and intent_class
        and body.get("action_type")
    ):
        asyncio.create_task(
            _reflector.reflect(
                session_id=body.get("session_id", "unknown"),
                trace_id=trace_id,
                intent_class=intent_class,
                action_type=body["action_type"],
                entity_class=body.get("entity_class", ""),
                actor_role=body.get("actor_role", "agent"),
                outcome=body.get("outcome_status", "UNKNOWN"),
                prediction_delta=prediction_delta,
                context_summary={
                    "outcome_score_predicted": outcome_score_predicted,
                    "early_flag": early_flag,
                },
            )
        )

    emit_event(trace_id, "nexi", "PREDICTION_DELTA_WRITTEN",
               {"prediction_delta": prediction_delta, "early_flag": early_flag})
    return {"status": "ok"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.get("/nexi/capabilities")
async def nexi_capabilities() -> dict[str, Any]:
    """Realtime merged capabilities (live probe status via the refresh loop)."""
    caps = _capability_state.get("capabilities")
    if caps is None:
        return load_capabilities()
    return caps


@app.post("/nexi/refresh")
async def nexi_refresh() -> dict[str, Any]:
    """On-demand full refresh: topology → tools → probes → overlay write."""
    caps = await _refresh_capabilities(force_write=True)
    return {
        "status": "ok",
        "generated_at": caps.get("generated_at"),
        "hosts": sorted(caps.get("hosts", {})),
        "healthy": caps.get("status", {}).get("healthy", []),
        "down": caps.get("status", {}).get("down", []),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_completion_ms(manifest) -> int:
    if not manifest.episodes:
        return 30_000
    completed = [ep for ep in manifest.episodes if ep.duration_ms]
    if not completed:
        return 30_000
    avg = sum(ep.duration_ms for ep in completed) / len(completed)
    return int(avg)
