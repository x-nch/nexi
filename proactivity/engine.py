from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx

from nexi.config import settings
from nexi.infra.discovery import build_snapshot, probe_services, service_priority

logger = logging.getLogger(__name__)


@dataclass
class ProactivityEvent:
    trigger: str
    message: str
    priority: int = 0
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=2))

    @property
    def expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> dict:
        return {
            "trigger": self.trigger,
            "message": self.message,
            "priority": self.priority,
            "expires_at": self.expires_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ProactivityEvent:
        return cls(
            trigger=d["trigger"],
            message=d["message"],
            priority=d.get("priority", 0),
            expires_at=datetime.fromisoformat(d["expires_at"]),
        )


class ProactivityEngine:
    def __init__(self, redis_client, http_client=None):
        self._redis = redis_client
        self._http = http_client or httpx.AsyncClient()

    async def check_and_queue(
        self,
        pattern_store=None,
        db_path=None,
    ) -> list[ProactivityEvent]:
        events: list[ProactivityEvent] = []
        now = datetime.now(timezone.utc)

        if pattern_store is not None:
            try:
                patterns = await pattern_store.fetch_low_success(max_success_rate=0.4, min_confidence=0.5)
                for p in patterns:
                    events.append(ProactivityEvent(
                        trigger="stale_pattern",
                        message=(
                            f"ck-san, the {p.get('action_type', 'unknown')} pattern for "
                            f"{p.get('intent_class', 'unknown')} is failing 60%+ of the time. "
                            f"Want me to draft a policy candidate fix?"
                        ),
                        priority=3,
                        expires_at=now + timedelta(hours=2),
                    ))
            except Exception as e:
                logger.warning("Proactivity Rule 1 (stale pattern) failed: %s", e)

        if db_path is not None:
            try:
                import aiosqlite
                async with aiosqlite.connect(db_path) as db:
                    async with db.execute(
                        "SELECT value FROM system_state WHERE key = 'last_consolidation_run'"
                    ) as cursor:
                        row = await cursor.fetchone()
                if row and row[0]:
                    last_run = datetime.fromisoformat(row[0])
                    if (now - last_run) > timedelta(hours=25):
                        events.append(ProactivityEvent(
                            trigger="consolidation_stale",
                            message="Consolidation has not run in 25h. Redis working memory may be stale.",
                            priority=2,
                            expires_at=now + timedelta(hours=4),
                        ))
            except Exception as e:
                logger.warning("Proactivity Rule 2 (consolidation) failed: %s", e)

        if self._http is not None:
            try:
                snapshot = build_snapshot()
                status = await probe_services(snapshot, http_client=self._http)
                for name in status["down"]:
                    if name == "vllm-ornith":
                        message = "Ornith (vLLM on Node B) is not responding."
                    else:
                        message = f"{name} is not responding."
                    events.append(ProactivityEvent(
                        trigger="inference_down" if name in {
                            "vllm-ornith", "vllm-qwen", "nexi"
                        } else "service_down",
                        message=message,
                        priority=service_priority(name),
                        expires_at=now + timedelta(hours=1),
                    ))
            except Exception as e:
                logger.warning("Infra probe (Rule 3) failed: %s", e)

        if db_path is not None:
            try:
                import aiosqlite
                async with aiosqlite.connect(db_path) as db:
                    async with db.execute(
                        "SELECT value FROM system_state WHERE key = 'last_extraction_run'"
                    ) as cursor:
                        row = await cursor.fetchone()
                if row and row[0]:
                    last_extract = datetime.fromisoformat(row[0])
                    if (now - last_extract) > timedelta(hours=7):
                        events.append(ProactivityEvent(
                            trigger="learning_loop_silence",
                            message="Pattern extractor has been silent for 7h. Check xnch-core logs.",
                            priority=4,
                            expires_at=now + timedelta(hours=2),
                        ))
            except Exception as e:
                logger.warning("Proactivity Rule 4 (learning silence) failed: %s", e)

        for ev in events:
            await self.queue_event(ev)

        return events

    async def queue_event(self, event: ProactivityEvent) -> None:
        key = f"proactivity:pending:{uuid4()}"
        ttl_seconds = max(60, int((event.expires_at - datetime.now(timezone.utc)).total_seconds()))
        await self._redis.set(key, json.dumps(event.to_dict()), ex=ttl_seconds)

    async def get_pending(self) -> list[ProactivityEvent]:
        events: list[ProactivityEvent] = []
        cursor = 0
        while True:
            cursor, keys = await self._redis.scan(cursor=cursor, match="proactivity:pending:*")
            for k in keys:
                raw = await self._redis.get(k)
                if raw is None:
                    continue
                try:
                    ev = ProactivityEvent.from_dict(json.loads(raw))
                    if not ev.expired:
                        events.append(ev)
                except Exception:
                    pass
            if cursor == 0:
                break
        events.sort(key=lambda e: e.priority, reverse=True)
        return events
