"""Step 11 — dispatch_execution forwards simulation + goal_id.

Covers two contracts:
1. When ``simulation``/``goal_id`` are passed, the ``/execute`` POST body
   carries ``simulation`` and ``goal_id`` on the dispatch payload.
2. The ``_record_stub_outcome`` fallback (runner unreachable) forwards
   ``goal_id`` to ``/execution/outcome`` so a goal step's outcome still
   advances the goal.
"""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from nexi.models import (
    Actor,
    ActorRole,
    DecisionRecord,
    SelectionRationale,
    SessionContext,
    VerdictResponse,
)


def _make_session() -> SessionContext:
    return SessionContext(
        session_id=uuid4(),
        trace_id=uuid4(),
        actor=Actor(id="act-1", role=ActorRole.AGENT, capability_set=[]),
        system_state_version="v1.0.0",
        policy_version="v1.0.0",
        idempotency_key=uuid4(),
        raw_input="",
    )


def _make_decision(session: SessionContext) -> DecisionRecord:
    return DecisionRecord(
        session_id=session.session_id,
        intent_ref=uuid4(),
        context_manifest_ref=uuid4(),
        system_state_version="v1.0.0",
        options_generated=3,
        options_blocked=0,
        options_evaluated=[],
        selected_option_id=uuid4(),
        selection_rationale=SelectionRationale(
            score_breakdown={},
            weight_config_version="v1.0.0",
        ),
        confidence=0.8,
    )


def _make_verdict() -> VerdictResponse:
    return VerdictResponse(
        request_id=uuid4(),
        verdict="ALLOW",
        verdict_reason="ok",
        policy_refs=[],
        execution_token="tok-1",
        token_ttl_ms=30_000,
        audit_ref=uuid4(),
    )


@pytest.mark.asyncio
async def test_dispatch_posts_simulation_and_goal_id():
    """``/execute`` POST body must carry simulation + goal_id when supplied."""
    from nexi.pipeline.dispatch import dispatch_execution

    session = _make_session()
    decision = _make_decision(session)
    verdict = _make_verdict()
    simulation = {"predicted_outcome": "SUCCESS", "confidence": 0.9}
    goal_id = uuid4()

    with patch("httpx.AsyncClient") as mock_client:
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = AsyncMock(return_value=resp)
        mock_client.return_value = client

        payload = await dispatch_execution(
            session,
            decision,
            verdict,
            validated_action_spec={"type": "DEPLOY"},
            execution_runner_url="http://runner:8001/execution",
            simulation=simulation,
            goal_id=goal_id,
        )

    assert client.post.await_args.kwargs["json"]["simulation"] == simulation
    assert client.post.await_args.kwargs["json"]["goal_id"] == str(goal_id)
    assert payload.simulation == simulation
    assert payload.goal_id == goal_id


@pytest.mark.asyncio
async def test_stub_outcome_forwards_goal_id():
    """Runner-unavailable fallback must forward goal_id to /execution/outcome."""
    from nexi.pipeline.dispatch import dispatch_execution

    session = _make_session()
    decision = _make_decision(session)
    verdict = _make_verdict()
    goal_id = uuid4()

    runner_client = MagicMock()
    runner_client.__aenter__ = AsyncMock(return_value=runner_client)
    runner_client.__aexit__ = AsyncMock(return_value=None)
    runner_client.post = AsyncMock(side_effect=httpx.ConnectError("down"))

    stub_client = MagicMock()
    stub_client.__aenter__ = AsyncMock(return_value=stub_client)
    stub_client.__aexit__ = AsyncMock(return_value=None)
    stub_resp = MagicMock()
    stub_resp.raise_for_status = MagicMock()
    stub_client.post = AsyncMock(return_value=stub_resp)

    with patch("httpx.AsyncClient", side_effect=[runner_client, stub_client]):
        await dispatch_execution(
            session,
            decision,
            verdict,
            validated_action_spec={"type": "DEPLOY"},
            execution_runner_url="http://runner:8001/execution",
            goal_id=goal_id,
        )

    outcome_json = stub_client.post.await_args.kwargs["json"]
    assert outcome_json["goal_id"] == str(goal_id)
