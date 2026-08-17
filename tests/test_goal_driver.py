"""Goal driver — serialized poll → step loop reusing run_pipeline_pass."""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from nexi.goal.driver import _make_goal_session, _run_goal_step, goal_driver_loop
from nexi.models import Actor, ActorRole, Goal, SessionContext
from nexi.pipeline.run import PipelinePassResult


def _make_goal(**overrides: object) -> Goal:
    """Build a Goal model; optional field overrides via setattr."""
    goal = Goal(owner_actor_id="agent", objective="deploy media service")
    for key, value in overrides.items():
        setattr(goal, key, value)
    return goal


def _make_session() -> SessionContext:
    return SessionContext(
        session_id=uuid4(),
        trace_id=uuid4(),
        actor=Actor(id="agent", role=ActorRole.AGENT,
                    capability_set=["READ", "QUERY", "DEPLOY"]),
        system_state_version="v1",
        policy_version="p1",
        idempotency_key=uuid4(),
        raw_input="",
        priority="NORMAL",
    )


def _make_xnch() -> MagicMock:
    xnch = MagicMock()
    xnch.get_system_state = AsyncMock(
        return_value={"system_state_version": "v9", "policy_version": "p9"}
    )
    xnch.update_goal = AsyncMock()
    xnch.claim_next_goal = AsyncMock()
    return xnch


async def test_make_goal_session_uses_system_state_versions():
    """_make_goal_session threads live versions from get_system_state (R4)."""
    session = await _make_goal_session(_make_xnch())
    assert session.system_state_version == "v9"
    assert session.policy_version == "p9"
    assert session.actor.role is ActorRole.AGENT


async def test_make_goal_session_falls_back_to_empty_versions():
    """_make_goal_session degrades to empty versions when get_system_state fails."""
    xnch = MagicMock()
    xnch.get_system_state = AsyncMock(side_effect=RuntimeError("down"))
    session = await _make_goal_session(xnch)
    assert session.system_state_version == ""
    assert session.policy_version == ""


async def test_run_goal_step_marks_blocked_on_non_executing():
    """Non-EXECUTING pipeline result → update_goal(status="BLOCKED") with goal id."""
    goal = _make_goal()
    xnch = _make_xnch()
    with patch(
        "nexi.goal.driver.run_pipeline_pass",
        new=AsyncMock(return_value=PipelinePassResult(status="ESCALATED")),
    ):
        await _run_goal_step(
            goal, xnch=xnch, model_adapter=MagicMock(),
            policy_filter=MagicMock(), intent_interpreter=MagicMock(),
        )

    xnch.update_goal.assert_awaited_once()
    args, kwargs = xnch.update_goal.call_args
    assert args[0] == str(goal.goal_id)
    assert kwargs["status"] == "BLOCKED"
    assert kwargs["progress"] == "blocked: ESCALATED"


async def test_run_goal_step_passes_derived_inputs_to_pipeline():
    """simulation/goal_id/raw_input are derived from the Goal and passed through."""
    goal = _make_goal(
        simulation_plan=[{"kind": "sim", "name": "s0"}],
        steps_completed=0,
    )
    xnch = _make_xnch()
    run_mock = AsyncMock(return_value=PipelinePassResult(status="EXECUTING"))
    with patch("nexi.goal.driver.run_pipeline_pass", new=run_mock):
        await _run_goal_step(
            goal, xnch=xnch, model_adapter=MagicMock(),
            policy_filter=MagicMock(), intent_interpreter=MagicMock(),
        )

    kwargs = run_mock.call_args.kwargs
    assert kwargs["raw_input"] == "deploy media service"
    assert kwargs["goal_id"] == goal.goal_id
    assert kwargs["simulation"] == {"kind": "sim", "name": "s0"}
    xnch.update_goal.assert_not_awaited()


async def test_goal_driver_loop_polls_and_runs_one_step():
    """Loop claims repeatedly and runs one step before the clock stops it."""
    goal = _make_goal()
    xnch = _make_xnch()
    xnch.claim_next_goal = AsyncMock(side_effect=[goal, None])

    class _StopClock(Exception):
        pass

    sleep_calls = 0

    async def _fake_sleep(_interval: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 3:
            raise _StopClock()

    with (
        patch("nexi.goal.driver.asyncio.sleep", new=_fake_sleep),
        patch("nexi.goal.driver._run_goal_step", new=AsyncMock()) as step_mock,
    ):
        with pytest.raises(_StopClock):
            await goal_driver_loop(
                xnch=xnch, model_adapter=MagicMock(),
                policy_filter=MagicMock(), intent_interpreter=MagicMock(),
            )

    assert xnch.claim_next_goal.await_count >= 2
    step_mock.assert_awaited_once()


async def test_goal_driver_loop_marks_active_on_step_error_and_keeps_polling():
    """A step failure marks the goal ACTIVE and the loop survives (no crash)."""
    goal = _make_goal()
    xnch = _make_xnch()
    xnch.claim_next_goal = AsyncMock(side_effect=[goal, None])

    class _StopClock(Exception):
        pass

    sleep_calls = 0

    async def _fake_sleep(_interval: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 3:
            raise _StopClock()

    with (
        patch("nexi.goal.driver.asyncio.sleep", new=_fake_sleep),
        patch(
            "nexi.goal.driver._run_goal_step",
            new=AsyncMock(side_effect=RuntimeError("step boom")),
        ),
    ):
        with pytest.raises(_StopClock):
            await goal_driver_loop(
                xnch=xnch, model_adapter=MagicMock(),
                policy_filter=MagicMock(), intent_interpreter=MagicMock(),
            )

    xnch.update_goal.assert_awaited_once()
    args, kwargs = xnch.update_goal.call_args
    assert args[0] == str(goal.goal_id)
    assert kwargs["status"] == "ACTIVE"


async def test_goal_driver_loop_survives_recovery_update_failure():
    """A failed recovery update_goal is logged, not propagated — loop keeps polling."""
    goal = _make_goal()
    xnch = _make_xnch()
    xnch.claim_next_goal = AsyncMock(side_effect=[goal, None])
    xnch.update_goal = AsyncMock(side_effect=RuntimeError("xnch down"))

    class _StopClock(Exception):
        pass

    sleep_calls = 0

    async def _fake_sleep(_interval: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 3:
            raise _StopClock()

    with (
        patch("nexi.goal.driver.asyncio.sleep", new=_fake_sleep),
        patch(
            "nexi.goal.driver._run_goal_step",
            new=AsyncMock(side_effect=RuntimeError("step boom")),
        ),
    ):
        with pytest.raises(_StopClock):
            await goal_driver_loop(
                xnch=xnch, model_adapter=MagicMock(),
                policy_filter=MagicMock(), intent_interpreter=MagicMock(),
            )

    # Recovery update_goal was attempted (and its failure was swallowed).
    xnch.update_goal.assert_awaited_once()
