"""Reflection wiring in nexi /callback/outcome."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient, ASGITransport

from nexi.main import app


@pytest.mark.asyncio
async def test_callback_triggers_reflection_when_enabled():
    """Outcome callback should fire-and-forget a reflection for the context tuple."""
    payload = {
        "session_id": str(uuid4()),
        "episode_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "outcome_status": "FAILURE",
        "outcome_score_predicted": 0.8,
        "intent_class": "EXECUTION",
        "action_type": "DEPLOY",
        "entity_class": "SERVICE",
        "actor_role": "agent",
    }

    mock_reflector = MagicMock()
    mock_reflector.reflect = AsyncMock(return_value=None)

    mock_xnch = MagicMock()
    mock_xnch.write_prediction_update = AsyncMock(return_value=None)

    with patch("nexi.main.settings") as mock_settings, \
         patch("nexi.main._reflector", mock_reflector), \
         patch("nexi.main._xnch", mock_xnch):
        mock_settings.reflection_enabled = True

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/callback/outcome", json=payload)

        # Let the fire-and-forget task complete
        await asyncio.sleep(0.05)

    assert response.status_code == 200
    mock_reflector.reflect.assert_awaited_once()
    kwargs = mock_reflector.reflect.call_args.kwargs
    assert kwargs["intent_class"] == "EXECUTION"
    assert kwargs["action_type"] == "DEPLOY"
    assert kwargs["entity_class"] == "SERVICE"
    assert kwargs["actor_role"] == "agent"
    assert kwargs["outcome"] == "FAILURE"
    assert kwargs["prediction_delta"] == pytest.approx(0.8, abs=0.001)


@pytest.mark.asyncio
async def test_callback_skips_reflection_when_disabled():
    payload = {
        "session_id": str(uuid4()),
        "episode_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "outcome_status": "SUCCESS",
        "outcome_score_predicted": 0.9,
        "intent_class": "QUERY",
        "action_type": "LIST",
        "entity_class": "FILE",
        "actor_role": "viewer",
    }

    mock_reflector = MagicMock()
    mock_reflector.reflect = AsyncMock(return_value=None)

    with patch("nexi.main.settings") as mock_settings, \
         patch("nexi.main._reflector", mock_reflector):
        mock_settings.reflection_enabled = False

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/callback/outcome", json=payload)

        await asyncio.sleep(0.05)

    assert response.status_code == 200
    mock_reflector.reflect.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_reflection_skips_without_context_tuple():
    """Reflection must be skipped when xnch did not send context fields."""
    payload = {
        "session_id": str(uuid4()),
        "episode_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "outcome_status": "SUCCESS",
        "outcome_score_predicted": 0.9,
    }

    mock_reflector = MagicMock()
    mock_reflector.reflect = AsyncMock(return_value=None)

    with patch("nexi.main.settings") as mock_settings, \
         patch("nexi.main._reflector", mock_reflector):
        mock_settings.reflection_enabled = True

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/callback/outcome", json=payload)

        await asyncio.sleep(0.05)

    assert response.status_code == 200
    mock_reflector.reflect.assert_not_awaited()
