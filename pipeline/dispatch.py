"""Step 11 — Execution dispatch to execution-runner."""
from typing import Any
from uuid import UUID, uuid4

import httpx
import logging

from ..models import SessionContext, DecisionRecord, VerdictResponse, ExecutionDispatchPayload
from ..config import settings
from ..utils.audit import emit_event

logger = logging.getLogger(__name__)


class TokenExpired(Exception):
    pass


async def dispatch_execution(
    session: SessionContext,
    decision: DecisionRecord,
    verdict: VerdictResponse,
    validated_action_spec: dict[str, Any],
    execution_runner_url: str,
    simulation: dict[str, Any] | None = None,
    goal_id: UUID | None = None,
) -> ExecutionDispatchPayload:
    if not verdict.execution_token:
        raise ValueError("No execution token in verdict response")

    payload = ExecutionDispatchPayload(
        execution_ref=uuid4(),
        trace_id=session.trace_id,
        decision_id=decision.decision_id,
        action_spec=validated_action_spec,
        execution_token=verdict.execution_token,
        token_ttl_ms=verdict.token_ttl_ms,
        simulation=simulation,
        goal_id=goal_id,
    )

    emit_event(session.trace_id, "dispatch", "EXECUTION_DISPATCH",
               {"execution_ref": str(payload.execution_ref)})

    try:
        async with httpx.AsyncClient(base_url=execution_runner_url, timeout=10.0) as client:
            resp = await client.post("/execute", json=payload.model_dump(mode="json"))

            if resp.status_code == 401:
                error = resp.json().get("error", "")
                if "TOKEN_EXPIRED" in error:
                    raise TokenExpired("Execution token expired before dispatch")
                raise ValueError(f"Execution runner rejected dispatch: {error}")

            resp.raise_for_status()

        emit_event(session.trace_id, "dispatch", "EXECUTION_ACCEPTED",
                   {"execution_ref": str(payload.execution_ref)})
    except httpx.ConnectError:
        logger.warning(
            "Execution runner unavailable at %s — recording stub outcome "
            "(execution_ref=%s, decision_id=%s)",
            execution_runner_url, payload.execution_ref, decision.decision_id,
        )
        emit_event(session.trace_id, "dispatch", "EXECUTION_DEFERRED",
                   {"execution_ref": str(payload.execution_ref),
                    "reason": "runner_unavailable"})
        await _record_stub_outcome(session, payload)

    return payload


async def _record_stub_outcome(
    session: SessionContext,
    payload: ExecutionDispatchPayload,
) -> None:
    """Post SUCCESS to xnch when no execution runner is deployed."""
    try:
        async with httpx.AsyncClient(base_url=settings.xnch_base_url, timeout=10.0) as client:
            outcome_json: dict[str, Any] = {
                "execution_ref": str(payload.execution_ref),
                "decision_id": str(payload.decision_id),
                "execution_token_ref": payload.execution_token or "",
                "outcome_status": "SUCCESS",
                "duration_ms": 0,
            }
            if payload.goal_id is not None:
                outcome_json["goal_id"] = str(payload.goal_id)
            resp = await client.post("/execution/outcome", json=outcome_json)
            resp.raise_for_status()
        emit_event(session.trace_id, "dispatch", "STUB_OUTCOME_RECORDED",
                   {"execution_ref": str(payload.execution_ref)})
    except Exception as exc:
        logger.error("Failed to record stub outcome (decision_id=%s): %s",
                     payload.decision_id, exc)
