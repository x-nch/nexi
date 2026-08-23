"""Nexi Prometheus instrumentation: stage timing, policy gate, goal loop, callbacks, scrape gating."""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY


def _sample_value(name: str, labels: dict[str, str] | None = None) -> float | None:
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name != name:
                continue
            if labels is None or all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return None


# ---------------------------------------------------------------------------
# Stage timing
# ---------------------------------------------------------------------------


async def test_stage_timer_records_histogram() -> None:
    import asyncio

    from nexi.observability.metrics import stage_timer

    async with stage_timer("probe_stage"):
        await asyncio.sleep(0.01)

    assert (
        _sample_value("nexi_pipeline_stage_seconds_count", {"stage": "probe_stage"}) or 0.0
    ) >= 1.0


async def test_stage_timer_still_records_on_exception() -> None:
    from nexi.observability.metrics import stage_timer

    before = _sample_value("nexi_pipeline_stage_seconds_count", {"stage": "boom_stage"}) or 0.0
    with pytest.raises(RuntimeError):
        async with stage_timer("boom_stage"):
            raise RuntimeError("x")
    assert (_sample_value("nexi_pipeline_stage_seconds_count", {"stage": "boom_stage"}) or 0.0) == before + 1.0


def test_record_pass_outcome_counts_by_status() -> None:
    from nexi.observability.metrics import record_pass_outcome

    before = _sample_value("nexi_pipeline_pass_total", {"status": "ESCALATED"}) or 0.0
    record_pass_outcome("ESCALATED")
    assert (_sample_value("nexi_pipeline_pass_total", {"status": "ESCALATED"}) or 0.0) == before + 1.0


# ---------------------------------------------------------------------------
# PolicyFilter decisions
# ---------------------------------------------------------------------------


class _FakeXnch:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses

    async def check_policies_parallel(self, session: Any, options: list[Any]) -> list[Any]:
        return self._responses


async def test_policy_filter_counts_pass_and_block() -> None:
    from nexi.models import PolicyDryRunResponse, PlanOption, SessionContext
    from nexi.models.options import PolicyVerdict
    from nexi.pipeline.policy_filter import PolicyFilter

    def _opt(target: str) -> PlanOption:
        return PlanOption(
            option_id=uuid4(),
            action_type="noop",
            action_spec={"type": "noop", "target": target, "params": {}},
            stated_rationale="test",
            reversible=True,
            payload_hash="hash",
        )

    opt_a = _opt("x")
    opt_b = _opt("y")
    session = SessionContext(
        session_id=uuid4(),
        trace_id=uuid4(),
        actor={"id": "a", "role": "AGENT", "capability_set": []},
        system_state_version="v1",
        policy_version="v1",
        idempotency_key=uuid4(),
        raw_input="",
    )
    responses = [
        PolicyDryRunResponse(option_id=opt_a.option_id, session_id=session.session_id, verdict=PolicyVerdict.ALLOW, policy_refs=[]),
        PolicyDryRunResponse(option_id=opt_b.option_id, session_id=session.session_id, verdict=PolicyVerdict.BLOCK, policy_refs=[]),
    ]

    blocked_before = _sample_value("nexi_policy_options_total", {"verdict": "blocked"}) or 0.0
    passed_before = _sample_value("nexi_policy_options_total", {"verdict": "pass"}) or 0.0

    pf = PolicyFilter(_FakeXnch(responses))
    surviving = await pf.filter(session, [opt_a, opt_b])

    assert len(surviving) == 1
    assert (_sample_value("nexi_policy_options_total", {"verdict": "blocked"}) or 0.0) == blocked_before + 1.0
    assert (_sample_value("nexi_policy_options_total", {"verdict": "pass"}) or 0.0) == passed_before + 1.0


async def test_policy_filter_all_blocked_increments_escalation_counter() -> None:
    from nexi.models import PolicyDryRunResponse, PlanOption, SessionContext
    from nexi.models.options import PolicyVerdict
    from nexi.pipeline.policy_filter import AllOptionsBlocked, PolicyFilter

    opt = PlanOption(
        option_id=uuid4(),
        action_type="noop",
        action_spec={"type": "noop", "target": "x", "params": {}},
        stated_rationale="test",
        reversible=True,
        payload_hash="hash",
    )
    session = SessionContext(
        session_id=uuid4(),
        trace_id=uuid4(),
        actor={"id": "a", "role": "AGENT", "capability_set": []},
        system_state_version="v1",
        policy_version="v1",
        idempotency_key=uuid4(),
        raw_input="",
    )
    responses = [PolicyDryRunResponse(option_id=opt.option_id, session_id=session.session_id, verdict=PolicyVerdict.BLOCK, policy_refs=[])]

    esc_before = _sample_value("nexi_policy_all_blocked_total") or 0.0
    pf = PolicyFilter(_FakeXnch(responses))
    with pytest.raises(AllOptionsBlocked):
        await pf.filter(session, [opt])
    assert (_sample_value("nexi_policy_all_blocked_total") or 0.0) == esc_before + 1.0


# ---------------------------------------------------------------------------
# Goal driver claim/step outcomes
# ---------------------------------------------------------------------------


class _FakeGoalXnch:
    def __init__(self, goal: Any | None = None, claim_exc: Exception | None = None) -> None:
        self._goal = goal
        self._claim_exc = claim_exc
        self.updates: list[tuple[str, str]] = []

    async def get_system_state(self) -> dict[str, str]:
        return {"system_state_version": "v1", "policy_version": "p1"}

    async def claim_next_goal(self, owner: str) -> Any | None:
        if self._claim_exc:
            raise self._claim_exc
        return self._goal

    async def update_goal(self, goal_id: str, status: str, progress: str = "") -> None:
        self.updates.append((goal_id, status))


async def test_claim_once_counts_error_none_and_claimed() -> None:
    from nexi.goal.driver import _claim_once

    err_before = _sample_value("nexi_goal_claim_total", {"result": "error"}) or 0.0
    none_before = _sample_value("nexi_goal_claim_total", {"result": "none"}) or 0.0
    got_before = _sample_value("nexi_goal_claim_total", {"result": "claimed"}) or 0.0

    with pytest.raises(RuntimeError):
        await _claim_once(_FakeGoalXnch(claim_exc=RuntimeError("redis down")))
    assert (_sample_value("nexi_goal_claim_total", {"result": "error"}) or 0.0) == err_before + 1.0

    await _claim_once(_FakeGoalXnch(goal=None))
    assert (_sample_value("nexi_goal_claim_total", {"result": "none"}) or 0.0) == none_before + 1.0

    goal = await _claim_once(_FakeGoalXnch(goal="fake-goal"))
    assert goal == "fake-goal"
    assert (_sample_value("nexi_goal_claim_total", {"result": "claimed"}) or 0.0) == got_before + 1.0


async def test_run_goal_step_counts_executing_and_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import nexi.goal.driver as driver
    from nexi.pipeline.run import PipelinePassResult

    exec_before = _sample_value("nexi_goal_step_total", {"result": "executing"}) or 0.0
    blocked_before = _sample_value("nexi_goal_step_total", {"result": "blocked"}) or 0.0

    class _FakeGoal:
        goal_id = uuid4()

        def model_dump(self, mode: str = "python") -> dict[str, Any]:
            return {}

    async def fake_pass_executing(**kwargs: Any) -> PipelinePassResult:
        return PipelinePassResult(status="EXECUTING")

    async def fake_pass_escalated(**kwargs: Any) -> PipelinePassResult:
        return PipelinePassResult(status="ESCALATED", hold_id=uuid4())

    monkeypatch.setattr(driver, "run_pipeline_pass", fake_pass_executing)
    xnch = _FakeGoalXnch()
    await driver._run_goal_step(_FakeGoal(), xnch=xnch, model_adapter=None,
                                policy_filter=None, intent_interpreter=None)
    assert (_sample_value("nexi_goal_step_total", {"result": "executing"}) or 0.0) == exec_before + 1.0

    monkeypatch.setattr(driver, "run_pipeline_pass", fake_pass_escalated)
    await driver._run_goal_step(_FakeGoal(), xnch=xnch, model_adapter=None,
                                policy_filter=None, intent_interpreter=None)
    assert (_sample_value("nexi_goal_step_total", {"result": "blocked"}) or 0.0) == blocked_before + 1.0
    assert xnch.updates and xnch.updates[-1][1] == "BLOCKED"


# ---------------------------------------------------------------------------
# Outcome callback rate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outcome_callback_counts_skipped_and_write_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    skipped_before = _sample_value("nexi_outcome_callback_total", {"result": "skipped"}) or 0.0
    failed_before = _sample_value("nexi_outcome_callback_total", {"result": "write_failed"}) or 0.0

    from nexi.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post("/callback/outcome", json={"outcome_status": "SUCCESS"})
        assert r1.status_code == 200

        r2 = await client.post(
            "/callback/outcome",
            json={
                "outcome_status": "SUCCESS",
                "session_id": str(uuid4()),
                "episode_id": str(uuid4()),
                "trace_id": str(uuid4()),
            },
        )
        assert r2.status_code == 200

    assert (_sample_value("nexi_outcome_callback_total", {"result": "skipped"}) or 0.0) >= skipped_before + 1.0
    assert (
        _sample_value("nexi_outcome_callback_total", {"result": "write_failed"}) or 0.0
    ) >= failed_before + 1.0


# ---------------------------------------------------------------------------
# /metrics endpoint + HTTP middleware
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_endpoint_serves_request_series(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexi.config import settings

    monkeypatch.setattr(settings, "metrics_allow_cidrs", ["127.0.0.1", "::1"])
    from nexi.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        health = await client.get("/health")
        assert health.status_code == 200

        resp = await client.get("/metrics")
        assert resp.status_code == 200
        body = resp.text
    assert 'route="/health"' in body
    assert "nexi_http_requests_total" in body


@pytest.mark.asyncio
async def test_metrics_endpoint_denied_for_public_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from nexi.config import settings

    monkeypatch.setattr(settings, "metrics_allow_cidrs", ["10.99.0.0/16"])
    from nexi.main import app

    transport = ASGITransport(app=app, client=("203.0.113.9", 1234))
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.get("/metrics")
    assert resp.status_code == 403
