"""Tests for the shared single pipeline pass (nexi.pipeline.run.run_pipeline_pass).

Covers the extraction of session_start's steps 3-11 into a reusable coroutine,
with mocked XnchClient / ModelAdapter / PolicyFilter / IntentInterpreter.
"""
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexi.models import (
    Actor,
    ActorRole,
    ActionType,
    ContextManifest,
    GenerationPath,
    Intent,
    IntentClass,
    PlanOption,
    PolicyDryRunResponse,
    PolicyVerdict,
    SessionContext,
    Urgency,
    VerdictResponse,
    ExecutionDispatchPayload,
)
from nexi.models.options import ActionSpec
from nexi.pipeline.policy_filter import AllOptionsBlocked


def _make_session() -> SessionContext:
    return SessionContext(
        session_id=uuid4(),
        trace_id=uuid4(),
        actor=Actor(id="test-user", role=ActorRole.OPERATOR, capability_set=["DEPLOY"]),
        system_state_version="v1.0.0",
        policy_version="v1.0.0",
        idempotency_key=uuid4(),
        raw_input="deploy service myservice",
    )


def _make_intent() -> Intent:
    return Intent(
        session_id=uuid4(),
        intent_class=IntentClass.EXECUTION,
        action_type=ActionType.DEPLOY,
        target_entity_id="myservice",
        target_entity_class="SERVICE",
        urgency=Urgency.NORMAL,
        ambiguity_score=0.0,
        raw_input_hash="sha256:test",
        raw_input="deploy service myservice",
    )


def _make_manifest() -> ContextManifest:
    return ContextManifest(
        session_id=uuid4(),
        system_state_version="v1.0.0",
    )


def _make_option(option_id) -> PlanOption:
    return PlanOption(
        option_id=option_id,
        action_type="DEPLOY",
        action_spec=ActionSpec(type="DEPLOY", target="myservice", params={}),
        stated_rationale="safe deploy",
        reversible=True,
        payload_hash="sha256:payload",
    )


def _make_verdict() -> VerdictResponse:
    return VerdictResponse(
        request_id=uuid4(),
        verdict="ALLOW",
        verdict_reason="ok",
        policy_refs=[],
        execution_token="token-123",
        token_ttl_ms=30_000,
        audit_ref=uuid4(),
    )


@pytest.mark.asyncio
async def test_run_pipeline_pass_happy_path_dispatches_with_simulation_and_goal_id():
    """Canned happy path -> EXECUTING, dispatch called with simulation/goal_id."""
    from nexi.pipeline import run as run_mod

    session = _make_session()
    intent = _make_intent()
    manifest = _make_manifest()
    option_id = uuid4()
    option = _make_option(option_id)
    verdict = _make_verdict()
    dispatch_payload = ExecutionDispatchPayload(
        execution_ref=uuid4(),
        trace_id=session.trace_id,
        decision_id=uuid4(),
        action_spec={"type": "DEPLOY", "target": "myservice", "params": {}},
        execution_token="token-123",
        token_ttl_ms=30_000,
    )

    xnch = AsyncMock()
    xnch.get_weight_config = AsyncMock(return_value=None)
    xnch.submit_verdict = AsyncMock(return_value=verdict)

    model_adapter = MagicMock()

    policy_filter = MagicMock()
    policy_filter.filter = AsyncMock(return_value=[(
        option,
        PolicyDryRunResponse(
            option_id=option_id,
            session_id=session.session_id,
            verdict=PolicyVerdict.ALLOW,
            policy_refs=[],
        ),
    )])

    intent_interpreter = MagicMock()
    intent_interpreter.interpret = AsyncMock(return_value=intent)

    simulation = {"mode": "dry-run", "projected": "healthy"}
    goal_id = uuid4()

    with (
        patch.object(run_mod, "load_context", new=AsyncMock(return_value=manifest)),
        patch.object(
            run_mod,
            "generate_options",
            new=AsyncMock(return_value=([option], GenerationPath.MODEL)),
        ) as gen_options,
        patch.object(
            run_mod,
            "dispatch_execution",
            new=AsyncMock(return_value=dispatch_payload),
        ) as dispatch_fn,
    ):
        result = await run_mod.run_pipeline_pass(
            xnch=xnch,
            model_adapter=model_adapter,
            policy_filter=policy_filter,
            intent_interpreter=intent_interpreter,
            session=session,
            raw_input="deploy service myservice",
            simulation=simulation,
            goal_id=goal_id,
        )

    assert result.status == "EXECUTING"
    assert result.execution_ref == dispatch_payload.execution_ref
    assert result.audit_ref == verdict.audit_ref
    assert result.estimated_completion_ms == 30_000

    dispatch_fn.assert_awaited_once()
    assert dispatch_fn.call_args.kwargs["simulation"] == simulation
    assert dispatch_fn.call_args.kwargs["goal_id"] == goal_id
    assert gen_options.await_args.kwargs == {}
    xnch.submit_verdict.assert_awaited_once()
    assert xnch.submit_verdict.call_args.kwargs["goal_id"] == goal_id


@pytest.mark.asyncio
async def test_run_pipeline_pass_all_options_blocked_returns_escalated():
    """PolicyFilter.filter raising AllOptionsBlocked -> status=ESCALATED + hold_id."""
    from nexi.pipeline import run as run_mod

    session = _make_session()
    intent = _make_intent()
    manifest = _make_manifest()
    option_id = uuid4()
    option = _make_option(option_id)

    xnch = AsyncMock()
    xnch.get_weight_config = AsyncMock(return_value=None)

    model_adapter = MagicMock()

    policy_filter = MagicMock()
    policy_filter.filter = AsyncMock(side_effect=AllOptionsBlocked("all blocked"))

    intent_interpreter = MagicMock()
    intent_interpreter.interpret = AsyncMock(return_value=intent)

    with (
        patch.object(run_mod, "load_context", new=AsyncMock(return_value=manifest)),
        patch.object(
            run_mod,
            "generate_options",
            new=AsyncMock(return_value=([option], GenerationPath.MODEL)),
        ),
    ):
        result = await run_mod.run_pipeline_pass(
            xnch=xnch,
            model_adapter=model_adapter,
            policy_filter=policy_filter,
            intent_interpreter=intent_interpreter,
            session=session,
            raw_input="deploy service myservice",
        )

    assert result.status == "ESCALATED"
    assert result.hold_id is not None
    assert result.decision_id is None
    assert result.execution_ref is None
