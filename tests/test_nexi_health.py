"""Nexi /health endpoint tests."""
import pytest


@pytest.mark.asyncio
async def test_health_returns_ok():
    """Health endpoint should return ok status."""
    from httpx import AsyncClient, ASGITransport
    from nexi.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_health_includes_version():
    """Health endpoint should include version field."""
    from httpx import AsyncClient, ASGITransport
    from nexi.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    data = response.json()
    assert data["version"] == "0.1.0"