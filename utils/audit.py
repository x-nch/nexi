import json
import time
from uuid import UUID

from agentmemory import create_event, get_events


def emit_event(
    trace_id: UUID | str,
    component: str,
    event_type: str,
    payload: dict | None = None,
) -> None:
    """Fire-and-forget event emission via agentmemory."""
    try:
        text = f"{component}:{event_type}:{trace_id}"
        metadata = {
            "trace_id": str(trace_id),
            "component": component,
            "event_type": event_type,
            "timestamp_ns": time.time_ns(),
            **(payload or {}),
        }
        create_event(text, metadata=metadata)
    except Exception:
        pass
