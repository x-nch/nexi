"""Goal driver — serialized poll → step loop reusing run_pipeline_pass."""
import asyncio
import logging

from nexi.config import settings
from nexi.models import SessionContext, Actor, ActorRole, Goal
from nexi.pipeline.intent_interpreter import ClarificationRequired
from nexi.pipeline.run import run_pipeline_pass
from .planner import build_step_input, build_simulation

logger = logging.getLogger(__name__)
_LEASE_OWNER = "nexi-goal-driver"
# In-memory consecutive step-error counter (keyed by str(goal_id)).
# Acceptable here because the goal driver is a single serialized loop.
_consecutive_step_errors: dict[str, int] = {}


async def _make_goal_session(xnch) -> SessionContext:
    from uuid import uuid4
    state: dict = {}
    try:
        state = await xnch.get_system_state()
    except Exception as exc:
        logger.warning("get_system_state failed, using empty versions: %s", exc)
    return SessionContext(
        session_id=uuid4(), trace_id=uuid4(),
        actor=Actor(id="agent", role=ActorRole.AGENT,
                    capability_set=["READ", "QUERY", "DEPLOY"]),
        system_state_version=state.get("system_state_version", ""),
        policy_version=state.get("policy_version", ""),
        idempotency_key=uuid4(), raw_input="", priority="NORMAL",
    )


async def _run_goal_step(goal: Goal, *, xnch, model_adapter, policy_filter, intent_interpreter) -> None:
    goal_dict = goal.model_dump(mode="json")
    session = await _make_goal_session(xnch)
    result = await run_pipeline_pass(
        xnch=xnch, model_adapter=model_adapter, policy_filter=policy_filter,
        intent_interpreter=intent_interpreter, session=session,
        raw_input=build_step_input(goal_dict),
        simulation=build_simulation(goal_dict),
        goal_id=goal.goal_id,
    )
    if result.status != "EXECUTING":
        await xnch.update_goal(str(goal.goal_id), status="BLOCKED",
                               progress=f"blocked: {result.status}")


async def goal_driver_loop(*, xnch, model_adapter, policy_filter, intent_interpreter) -> None:
    while True:
        await asyncio.sleep(settings.goal_poll_interval_s)
        try:
            goal = await xnch.claim_next_goal(_LEASE_OWNER)
        except Exception as exc:
            logger.warning("goal claim failed: %s", exc)
            continue
        if goal is None:
            continue
        try:
            await _run_goal_step(goal, xnch=xnch, model_adapter=model_adapter,
                                 policy_filter=policy_filter,
                                 intent_interpreter=intent_interpreter)
            _consecutive_step_errors.pop(str(goal.goal_id), None)
        except asyncio.CancelledError:
            raise
        except ClarificationRequired as exc:
            _consecutive_step_errors.pop(str(goal.goal_id), None)
            logger.warning("goal step needs clarification (goal=%s): %s — blocking", goal.goal_id, exc)
            try:
                await xnch.update_goal(str(goal.goal_id), status="BLOCKED",
                                       progress="blocked: clarification required")
            except Exception as recovery_exc:
                logger.error("goal recovery update failed (goal=%s): %s", goal.goal_id, recovery_exc)
        except Exception as exc:
            key = str(goal.goal_id)
            count = _consecutive_step_errors.get(key, 0) + 1
            _consecutive_step_errors[key] = count
            logger.error("goal step failed (goal=%s): %s", goal.goal_id, exc)
            try:
                if count >= settings.goal_max_consecutive_step_errors:
                    _consecutive_step_errors.pop(key, None)
                    await xnch.update_goal(str(goal.goal_id), status="BLOCKED",
                                           progress=f"blocked: {count} consecutive step errors")
                else:
                    await xnch.update_goal(str(goal.goal_id), status="ACTIVE")
            except Exception as recovery_exc:
                logger.error("goal recovery update failed (goal=%s): %s", goal.goal_id, recovery_exc)
