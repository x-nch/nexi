"""P2-nexi: workflow executor loop + XnchClient step claim/outcome methods.

RED until nexi/workflow/executor.py exists, XnchClient gains
claim_workflow_step/post_step_outcome, and nexi config carries the
workflow executor flags.

The executor takes an injected ``execute_fn`` so tests run without the full
pipeline dependency tree; the default path imports run_pipeline_pass lazily.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from nexi.adapters.xnch_client import XnchClient


def _step_dict(**overrides) -> dict:
    base = {
        "step_uuid": str(uuid4()),
        "run_id": str(uuid4()),
        "idx": 0,
        "kind": "exec_tool",
        "summary": "Search highlights via web_search",
        "payload": {"target": "web_search", "args": {"query": "xnch"}},
        "requires_approval": True,
        "status": "APPROVED",
        "retry_count": 0,
        "max_retries": 3,
    }
    base.update(overrides)
    return base


def _resp(payload) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


# ----------------------------------------------------------------------
# Adapter methods
# ----------------------------------------------------------------------


async def test_claim_workflow_step_posts_and_parses():
    client = XnchClient()
    client._http = AsyncMock()
    step = _step_dict(status="CLAIMED")  # xnch returns the row post-claim
    client._http.post = AsyncMock(return_value=_resp(step))

    result = await client.claim_workflow_step("nexi-wf-executor", ttl_s=120)

    client._http.post.assert_awaited_once_with(
        "/workflows/steps/claim",
        json={"lease_owner": "nexi-wf-executor", "ttl_s": 120},
    )
    assert result is not None
    assert result["step_uuid"] == step["step_uuid"]
    assert result["status"] == "CLAIMED"


async def test_claim_workflow_step_returns_none_on_204():
    client = XnchClient()
    client._http = AsyncMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.status_code = 204
    resp.json = MagicMock(return_value=None)
    client._http.post = AsyncMock(return_value=resp)

    assert await client.claim_workflow_step("nexi-wf-executor") is None


async def test_post_step_outcome_posts_and_parses():
    client = XnchClient()
    client._http = AsyncMock()
    step_uuid = str(uuid4())
    done = _step_dict(step_uuid=step_uuid, status="DONE")
    client._http.post = AsyncMock(return_value=_resp(done))

    result = await client.post_step_outcome(step_uuid, outcome_status="SUCCESS")

    client._http.post.assert_awaited_once_with(
        f"/workflows/steps/{step_uuid}/outcome",
        json={"outcome_status": "SUCCESS"},
    )
    assert result["status"] == "DONE"


# ----------------------------------------------------------------------
# Executor loop (injected execute_fn — no pipeline imports needed)
# ----------------------------------------------------------------------


def _make_xnch_with_steps(steps):
    xnch = MagicMock()
    xnch.claim_workflow_step = AsyncMock(side_effect=[*steps, None])
    xnch.post_step_outcome = AsyncMock(return_value={"status": "DONE"})
    return xnch


class _StopClock(Exception):
    pass


def _stop_after(n):
    calls = {"n": 0}

    async def _fake_sleep(_interval: float) -> None:
        calls["n"] += 1
        if calls["n"] >= n:
            raise _StopClock()

    return _fake_sleep


async def test_loop_claims_executes_and_reports_success(monkeypatch):
    from nexi.workflow.executor import workflow_executor_loop

    step = _step_dict()
    xnch = _make_xnch_with_steps([step])
    execute_fn = AsyncMock(return_value={"status": "EXECUTING"})

    monkeypatch.setattr("nexi.workflow.executor.asyncio.sleep", _stop_after(2))
    with pytest.raises(_StopClock):
        await workflow_executor_loop(xnch=xnch, execute_fn=execute_fn)

    execute_fn.assert_awaited_once()
    assert execute_fn.await_args.kwargs["step"]["step_uuid"] == step["step_uuid"]
    xnch.post_step_outcome.assert_awaited_once_with(
        step["step_uuid"], outcome_status="SUCCESS"
    )


async def test_loop_reports_failure_when_execute_raises(monkeypatch):
    from nexi.workflow.executor import workflow_executor_loop

    step = _step_dict()
    xnch = _make_xnch_with_steps([step])

    async def boom(*, step, xnch):  # noqa: ANN001
        raise RuntimeError("tool exploded")

    monkeypatch.setattr("nexi.workflow.executor.asyncio.sleep", _stop_after(2))
    with pytest.raises(_StopClock):
        await workflow_executor_loop(xnch=xnch, execute_fn=boom)

    xnch.post_step_outcome.assert_awaited_once_with(
        step["step_uuid"], outcome_status="FAILURE"
    )


async def test_loop_survives_claim_errors_and_keeps_polling(monkeypatch):
    from nexi.workflow.executor import workflow_executor_loop

    xnch = MagicMock()
    xnch.claim_workflow_step = AsyncMock(side_effect=RuntimeError("net down"))
    xnch.post_step_outcome = AsyncMock()

    monkeypatch.setattr("nexi.workflow.executor.asyncio.sleep", _stop_after(3))
    with pytest.raises(_StopClock):
        await workflow_executor_loop(
            xnch=xnch, execute_fn=AsyncMock(), poll_interval_s=0.01
        )

    xnch.claim_workflow_step.await_count >= 3
    xnch.post_step_outcome.assert_not_awaited()


async def test_default_execute_uses_pipeline_pass(monkeypatch):
    """The default execution path routes through run_pipeline_pass with the
    step summary as raw_input."""
    from nexi.workflow import executor as ex

    called = {}

    class FakeResult:
        status = "EXECUTING"

    async def fake_run_pipeline_pass(**kwargs):
        called.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(
        "nexi.pipeline.run.run_pipeline_pass", fake_run_pipeline_pass
    )

    xnch = MagicMock()
    xnch.get_system_state = AsyncMock(
        return_value={"system_state_version": "v1", "policy_version": "p1"}
    )
    step = _step_dict()

    async def fake_session_factory(xnch_arg):
        return {"system_state_version": "v9", "policy_version": "p9"}

    await ex._default_execute_step(step, xnch=xnch, session_factory=fake_session_factory)

    assert called["raw_input"].startswith("[workflow]")
    assert called["session"].system_state_version == "v9"
