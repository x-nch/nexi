"""Nexi v0 — decision engine FastAPI application."""
import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, BackgroundTasks
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
from .pipeline import IntentInterpreter, ClarificationRequired, PolicyFilter
from .pipeline.run import run_pipeline_pass
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

    try:
        result = await run_pipeline_pass(
            xnch=_xnch, model_adapter=_model_adapter,
            policy_filter=_policy_filter, intent_interpreter=_intent_interpreter,
            session=session, raw_input=body.raw_input,
        )
    except ClarificationRequired:
        return SessionStartResponse(status="CLARIFICATION_REQUIRED", clarification_required=True)

    if result.status == "ESCALATED":
        return SessionStartResponse(status="ESCALATED", hold_id=result.hold_id)

    return SessionStartResponse(
        status="EXECUTING",
        decision_id=result.decision_id,
        execution_ref=result.execution_ref,
        estimated_completion_ms=result.estimated_completion_ms,
        audit_ref=result.audit_ref,
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
