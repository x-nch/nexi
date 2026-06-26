from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import fakeredis.aioredis
import pytest

from nexi.proactivity.engine import ProactivityEngine, ProactivityEvent


@pytest.fixture
def fake_redis():
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield r


@pytest.fixture
def engine(fake_redis):
    return ProactivityEngine(redis_client=fake_redis)


@pytest.mark.asyncio
async def test_queue_and_get_pending(engine):
    ev = ProactivityEvent(
        trigger="test",
        message="hello from proactivity",
        priority=3,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    await engine.queue_event(ev)
    pending = await engine.get_pending()
    assert len(pending) == 1
    assert pending[0].message == "hello from proactivity"
    assert pending[0].trigger == "test"
    assert pending[0].priority == 3


@pytest.mark.asyncio
async def test_get_pending_empty(engine):
    pending = await engine.get_pending()
    assert pending == []


@pytest.mark.asyncio
async def test_expired_event_skipped(engine, fake_redis):
    ev = ProactivityEvent(
        trigger="stale",
        message="this is old",
        priority=1,
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    await engine.queue_event(ev)
    pending = await engine.get_pending()
    assert pending == []


@pytest.mark.asyncio
async def test_multiple_events_priority_order(engine):
    for i, (msg, pri) in enumerate([
        ("low priority", 1),
        ("high priority", 10),
        ("medium priority", 5),
    ]):
        ev = ProactivityEvent(
            trigger="test",
            message=msg,
            priority=pri,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        await engine.queue_event(ev)
    pending = await engine.get_pending()
    assert pending[0].message == "high priority"
    assert pending[1].message == "medium priority"
    assert pending[2].message == "low priority"


@pytest.mark.asyncio
async def test_check_and_queue_inference_down(engine, fake_redis):
    mock_http = MagicMock()
    mock_http.get = AsyncMock(side_effect=Exception("connection refused"))
    engine._http = mock_http

    events = await engine.check_and_queue()
    assert len(events) >= 1
    assert any(e.trigger == "inference_down" for e in events)

    pending = await engine.get_pending()
    assert len(pending) >= 1


@pytest.mark.asyncio
async def test_proactivity_event_expired_property():
    fresh = ProactivityEvent(
        trigger="t", message="m", priority=1,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    assert not fresh.expired
    old = ProactivityEvent(
        trigger="t", message="m", priority=1,
        expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )
    assert old.expired


@pytest.mark.asyncio
async def test_proactivity_event_roundtrip():
    ev = ProactivityEvent(
        trigger="test_trigger",
        message="roundtrip message",
        priority=7,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
    )
    d = ev.to_dict()
    ev2 = ProactivityEvent.from_dict(d)
    assert ev2.trigger == ev.trigger
    assert ev2.message == ev.message
    assert ev2.priority == ev.priority


@pytest.mark.asyncio
async def test_check_and_queue_stale_pattern(engine, fake_redis):
    mock_pattern_store = MagicMock()
    mock_pattern_store.fetch_low_success = AsyncMock(
        return_value=[
            {"action_type": "deploy", "intent_class": "infra_change"},
            {"action_type": "scale", "intent_class": "capacity_mgmt"},
        ]
    )
    events = await engine.check_and_queue(pattern_store=mock_pattern_store)
    stale_events = [e for e in events if e.trigger == "stale_pattern"]
    assert len(stale_events) == 2
    assert "deploy" in stale_events[0].message or "scale" in stale_events[0].message
