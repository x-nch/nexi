"""XnchClient goal-tracking method tests: claim/update/system-state/verdict-goal_id."""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from nexi.adapters.xnch_client import XnchClient
from nexi.models import (
    Actor,
    ActorRole,
    DecisionRecord,
    GenerationPath,
    Goal,
    GoalStatus,
    SelectionRationale,
    SessionContext,
)


def _make_goal_dict(**overrides: object) -> dict[str, object]:
    """Build a JSON-serializable Goal payload (as xnch would return it)."""
    goal = Goal(owner_actor_id="agent", objective="deploy media service")
    data: dict[str, object] = goal.model_dump(mode="json")
    data.update(overrides)
    return data


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


def _make_decision() -> DecisionRecord:
    return DecisionRecord(
        session_id=uuid4(),
        intent_ref=uuid4(),
        context_manifest_ref=uuid4(),
        system_state_version="v1.0.0",
        options_generated=3,
        options_blocked=0,
        options_evaluated=[],
        selected_option_id=None,
        selection_rationale=SelectionRationale(
            score_breakdown={}, weight_config_version="v1"
        ),
        confidence=0.8,
        generation_path=GenerationPath.MODEL,
    )


def _make_verdict_response_dict() -> dict[str, object]:
    return {
        "request_id": str(uuid4()),
        "verdict": "ALLOW",
        "verdict_reason": "allowed",
        "policy_refs": [],
        "modified_action": None,
        "execution_token": None,
        "token_ttl_ms": 0,
        "audit_ref": str(uuid4()),
    }


async def test_claim_next_goal_posts_and_parses_goal():
    """claim_next_goal POSTs /goals/claim with lease_owner and returns a Goal."""
    client = XnchClient()
    client._http = AsyncMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=_make_goal_dict(goal_id=str(uuid4())))
    client._http.post = AsyncMock(return_value=resp)

    result = await client.claim_next_goal("nexi-agent-1")

    client._http.post.assert_awaited_once_with(
        "/goals/claim", json={"lease_owner": "nexi-agent-1"}
    )
    assert isinstance(result, Goal)


async def test_claim_next_goal_returns_none_on_null():
    """claim_next_goal returns None when xnch responds with a JSON null body."""
    client = XnchClient()
    client._http = AsyncMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=None)
    client._http.post = AsyncMock(return_value=resp)

    result = await client.claim_next_goal("nexi-agent-1")

    assert result is None


async def test_update_goal_posts_and_parses_goal():
    """update_goal POSTs /goals/{id}/update with status/progress and parses Goal."""
    client = XnchClient()
    client._http = AsyncMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=_make_goal_dict(status="RUNNING"))
    client._http.post = AsyncMock(return_value=resp)

    goal_id = str(uuid4())
    result = await client.update_goal(
        goal_id, status="RUNNING", progress="step 1 complete"
    )

    client._http.post.assert_awaited_once_with(
        f"/goals/{goal_id}/update",
        json={"status": "RUNNING", "progress": "step 1 complete"},
    )
    assert isinstance(result, Goal)
    assert result.status is GoalStatus.RUNNING


async def test_update_goal_allows_none_status_and_progress():
    """update_goal sends explicit None values for omitted status/progress."""
    client = XnchClient()
    client._http = AsyncMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=_make_goal_dict())
    client._http.post = AsyncMock(return_value=resp)

    goal_id = str(uuid4())
    await client.update_goal(goal_id)

    client._http.post.assert_awaited_once_with(
        f"/goals/{goal_id}/update",
        json={"status": None, "progress": None},
    )


async def test_get_system_state_returns_dict():
    """get_system_state GETs /system/state and returns the raw dict."""
    client = XnchClient()
    client._http = AsyncMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(
        return_value={"system_state_version": "v2.0.0", "policy_version": "p1.5.0"}
    )
    client._http.get = AsyncMock(return_value=resp)

    result = await client.get_system_state()

    client._http.get.assert_awaited_once_with("/system/state")
    assert result == {"system_state_version": "v2.0.0", "policy_version": "p1.5.0"}


async def test_submit_verdict_includes_goal_id_when_provided():
    """submit_verdict threads goal_id into the context dict when given."""
    client = XnchClient()
    client._http = AsyncMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=_make_verdict_response_dict())
    client._http.post = AsyncMock(return_value=resp)

    goal_id = uuid4()
    with patch("nexi.adapters.xnch_client.emit_event"):
        await client.submit_verdict(
            _make_session(),
            _make_decision(),
            {"type": "DEPLOY", "target": "svc", "params": {}},
            "hash-1",
            goal_id=goal_id,
        )

    body = client._http.post.call_args.kwargs["json"]
    assert body["context"]["goal_id"] == str(goal_id)


async def test_submit_verdict_omits_goal_id_when_not_provided():
    """submit_verdict leaves goal_id out of context when not provided."""
    client = XnchClient()
    client._http = AsyncMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=_make_verdict_response_dict())
    client._http.post = AsyncMock(return_value=resp)

    with patch("nexi.adapters.xnch_client.emit_event"):
        await client.submit_verdict(
            _make_session(),
            _make_decision(),
            {"type": "DEPLOY", "target": "svc", "params": {}},
            "hash-1",
        )

    body = client._http.post.call_args.kwargs["json"]
    assert "goal_id" not in body["context"]
