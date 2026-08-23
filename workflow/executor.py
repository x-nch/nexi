"""Workflow executor (P2) — claims APPROVED steps from xnch, runs them
through the pipeline, reports outcomes.

Mirrors nexi/goal/driver.py's serialized poll loop. ``workflow_executor_loop``
takes an injected ``execute_fn`` so tests run without the pipeline dependency
tree; ``_default_execute_step`` lazily imports run_pipeline_pass.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from nexi.config import settings

logger = logging.getLogger(__name__)

_LEASE_OWNER = "nexi-wf-executor"

ExecuteFn = Callable[..., Awaitable[Any]]


async def _make_session(xnch) -> dict[str, str]:
    try:
        state = await xnch.get_system_state()
    except Exception as exc:
        logger.warning("get_system_state failed, using empty versions: %s", exc)
        state = {}
    return {
        "system_state_version": state.get("system_state_version", ""),
        "policy_version": state.get("policy_version", ""),
    }


def _step_raw_input(step: dict[str, Any]) -> str:
    target = (step.get("payload") or {}).get("target")
    base = f"[workflow] {step.get('summary', '')}"
    if target:
        base = f"{base} — target: {target}"
    return base


async def _default_execute_step(
    step: dict[str, Any],
    *,
    xnch,
    session_factory=None,
    **pipeline_kwargs: Any,
) -> Any:
    """Default execution path: one pipeline pass per claimed step."""
    from uuid import uuid4

    from nexi.models import Actor, ActorRole, SessionContext
    from nexi.pipeline.run import run_pipeline_pass

    versions = (
        await session_factory(xnch) if session_factory else await _make_session(xnch)
    )
    session = SessionContext(
        session_id=uuid4(),
        trace_id=uuid4(),
        actor=Actor(
            id="agent",
            role=ActorRole.AGENT,
            capability_set=["READ", "QUERY", "DEPLOY"],
        ),
        system_state_version=versions.get("system_state_version", ""),
        policy_version=versions.get("policy_version", ""),
        idempotency_key=uuid4(),
        raw_input="",
        priority="NORMAL",
    )
    return await run_pipeline_pass(
        xnch=xnch,
        session=session,
        raw_input=_step_raw_input(step),
        **pipeline_kwargs,
    )


async def workflow_executor_loop(
    *,
    xnch,
    execute_fn: ExecuteFn = _default_execute_step,
    poll_interval_s: float | None = None,
    lease_owner: str = _LEASE_OWNER,
) -> None:
    """Serialized claim → execute → outcome loop. Survives transient errors."""
    interval = (
        poll_interval_s
        if poll_interval_s is not None
        else settings.workflow_poll_interval_s
    )
    while True:
        await asyncio.sleep(interval)
        try:
            step = await xnch.claim_workflow_step(lease_owner)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("workflow step claim failed: %s", exc)
            continue
        if step is None:
            continue

        step_uuid = step.get("step_uuid", "")
        try:
            result = await execute_fn(step=step, xnch=xnch)
            outcome = (
                "SUCCESS"
                if getattr(result, "status", "EXECUTING") == "EXECUTING"
                else "FAILURE"
            )
        except asyncio.CancelledError:
            # release lease implicitly via expiry; surface cancellation
            logger.error("executor cancelled mid-step (step=%s)", step_uuid)
            raise
        except Exception as exc:
            logger.error("workflow step failed (step=%s): %s", step_uuid, exc)
            outcome = "FAILURE"

        try:
            await xnch.post_step_outcome(step_uuid, outcome_status=outcome)
        except Exception as exc:
            logger.error("outcome post failed (step=%s): %s", step_uuid, exc)
