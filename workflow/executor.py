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

import httpx

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


async def _dispatch_execution(xnch, step: dict[str, Any]) -> str:
    """Dispatch action spec to xnch execution endpoint."""
    # Build action_spec from step fields (workflow step format)
    action_spec = step.get("payload", {}).get("action_spec")
    if not action_spec:
        # Build from step fields: kind -> type, target, args -> params
        kind = step.get("kind") or step.get("payload", {}).get("kind")
        target = step.get("target") or step.get("payload", {}).get("target")
        args = step.get("args") or step.get("payload", {}).get("args") or {}
        if kind and target:
            action_spec = {"type": kind.upper(), "target": target, "params": args}
    
    if not action_spec:
        logger.warning("no action_spec in step payload: keys=%s", list(step.keys()))
        return "FAILURE"

    body = {
        "execution_ref": step.get("step_uuid", ""),
        "decision_id": step.get("step_uuid", ""),
        "action_spec": action_spec,
        "simulation": {},
    }

    xnch_url = settings.xnch_base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(base_url=xnch_url, timeout=60.0) as client:
            resp = await client.post("/execution/execute", json=body)
            resp.raise_for_status()
            data = resp.json()
            return data.get("outcome_status", "SUCCESS")
    except httpx.HTTPError as exc:
        logger.error("execution dispatch failed (step=%s): %s", step.get("step_uuid"), exc)
        return "FAILURE"


def _step_raw_input(step: dict[str, Any]) -> str:
    target = (step.get("payload") or {}).get("target")
    base = f"[workflow] {step.get('summary', '')}"
    if target:
        base = f"{base} — target: {target}"
    return base


def _step_model_override(step: dict[str, Any]) -> tuple[str | None, str | None]:
    """Per-step model override from the claimed step (payload.model_*), if any."""
    payload = step.get("payload") or {}
    return payload.get("model_provider"), payload.get("model_id")


async def _default_execute_step(
    step: dict[str, Any],
    *,
    xnch,
    session_factory=None,
    model_adapter=None,
    policy_filter=None,
    intent_interpreter=None,
    **pipeline_kwargs: Any,
) -> Any:
    """Default execution path: one pipeline pass per claimed step."""
    from uuid import uuid4

    from nexi.models import Actor, ActorRole, SessionContext
    from nexi.pipeline.run import run_pipeline_pass

    if model_adapter is None or policy_filter is None or intent_interpreter is None:
        from nexi.adapters import ModelAdapter
        from nexi.pipeline import IntentInterpreter, PolicyFilter

        model_adapter = model_adapter if model_adapter is not None else ModelAdapter()
        intent_interpreter = (
            intent_interpreter
            if intent_interpreter is not None
            else IntentInterpreter()
        )
        policy_filter = (
            policy_filter if policy_filter is not None else PolicyFilter(xnch)
        )

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
    model_provider, model_id = _step_model_override(step)
    return await run_pipeline_pass(
        xnch=xnch,
        model_adapter=model_adapter,
        policy_filter=policy_filter,
        intent_interpreter=intent_interpreter,
        session=session,
        raw_input=_step_raw_input(step),
        model_provider=model_provider,
        model_id=model_id,
        **pipeline_kwargs,
    )


async def workflow_executor_loop(
    *,
    xnch,
    execute_fn: ExecuteFn = _default_execute_step,
    poll_interval_s: float | None = None,
    lease_owner: str = _LEASE_OWNER,
    model_adapter=None,
    policy_filter=None,
    intent_interpreter=None,
    dispatch_enabled: bool = True,
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
            result = await execute_fn(
                step=step,
                xnch=xnch,
                model_adapter=model_adapter,
                policy_filter=policy_filter,
                intent_interpreter=intent_interpreter,
            )
            pipeline_status = getattr(result, "status", "EXECUTING")
            
            # If pipeline returns EXECUTING, dispatch the action to execution endpoint
            if pipeline_status == "EXECUTING" and dispatch_enabled:
                dispatch_outcome = await _dispatch_execution(xnch, step)
                outcome = dispatch_outcome
            elif pipeline_status == "EXECUTING":
                # dispatch disabled (e.g., tests) - treat EXECUTING as success
                outcome = "SUCCESS"
            else:
                outcome = "FAILURE"
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