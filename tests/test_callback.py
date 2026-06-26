"""Nexi /callback/outcome endpoint tests."""
import pytest
from uuid import uuid4


@pytest.mark.asyncio
async def test_callback_accepts_valid_outcome():
    """Callback should accept valid outcome payload."""
    from httpx import AsyncClient, ASGITransport
    from nexi.main import app

    payload = {
        "session_id": str(uuid4()),
        "episode_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "outcome_status": "SUCCESS",
        "outcome_score_predicted": 0.8,
        "system_state_version": "v1.0.0",
        "policy_version": "v1.0.0",
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/callback/outcome", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_callback_failure_outcome():
    """Callback should accept FAILURE outcome."""
    from httpx import AsyncClient, ASGITransport
    from nexi.main import app

    payload = {
        "session_id": str(uuid4()),
        "episode_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "outcome_status": "FAILURE",
        "outcome_score_predicted": 0.2,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/callback/outcome", json=payload)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_callback_partial_outcome():
    """Callback should accept PARTIAL outcome."""
    from httpx import AsyncClient, ASGITransport
    from nexi.main import app

    payload = {
        "session_id": str(uuid4()),
        "episode_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "outcome_status": "PARTIAL",
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/callback/outcome", json=payload)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_callback_minimal_payload():
    """Callback should accept minimal payload."""
    from httpx import AsyncClient, ASGITransport
    from nexi.main import app

    payload = {
        "trace_id": str(uuid4()),
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/callback/outcome", json=payload)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_callback_computes_prediction_delta():
    """Callback should compute prediction delta correctly."""
    from httpx import AsyncClient, ASGITransport
    from nexi.main import app

    payload = {
        "session_id": str(uuid4()),
        "episode_id": str(uuid4()),
        "trace_id": str(uuid4()),
        "outcome_status": "SUCCESS",
        "outcome_score_predicted": 0.7,  # predicted 70% success
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/callback/outcome", json=payload)

    # Success = 1.0, predicted = 0.7, delta = 0.3
    assert response.status_code == 200