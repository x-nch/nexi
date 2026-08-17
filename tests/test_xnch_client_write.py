"""XnchClient write_prediction_update actor_role test."""
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_write_prediction_update_sends_lowercase_actor_role():
    """Prediction-update write must send actor_role so xnch capability check passes."""
    from nexi.adapters.xnch_client import XnchClient
    from nexi.models import SessionContext, Actor, ActorRole

    session = SessionContext(
        session_id=uuid4(),
        trace_id=uuid4(),
        actor=Actor(id="act-1", role=ActorRole.AGENT, capability_set=[]),
        system_state_version="v1.0.0",
        policy_version="v1.0.0",
        idempotency_key=uuid4(),
        raw_input="",
    )

    client = XnchClient()
    client._http = AsyncMock()
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    client._http.post = AsyncMock(return_value=resp)

    await client.write_prediction_update(session, uuid4(), 0.2, False)

    body = client._http.post.call_args.kwargs["json"]
    assert body["actor_role"] == "agent"
    assert body["write_type"] == "EPISODE_PREDICTION_UPDATE"
    assert "episode_id" in body["payload"]
