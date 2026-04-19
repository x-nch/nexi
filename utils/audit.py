import json
import os
import socket
import time
from pathlib import Path
from uuid import UUID

from ..config import settings


_audit_path = Path(settings.audit_events_path).expanduser()
_audit_path.parent.mkdir(parents=True, exist_ok=True)


def emit_event(
    trace_id: UUID | str,
    component: str,
    event_type: str,
    payload: dict | None = None,
) -> None:
    """Fire-and-forget event emission to the Event Log (async, non-blocking)."""
    event = {
        "trace_id": str(trace_id),
        "component": component,
        "event_type": event_type,
        "timestamp_ns": time.time_ns(),
        **(payload or {}),
    }
    line = json.dumps(event) + "\n"
    # Best-effort append — never raises; execution path must not depend on this
    try:
        with _audit_path.open("a") as fh:
            fh.write(line)
    except Exception:
        pass
