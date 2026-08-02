from uuid import UUID

from xnch.memory.audit_store import emit_event as _pg_emit_event


def emit_event(
    trace_id: UUID | str,
    component: str,
    event_type: str,
    payload: dict | None = None,
) -> None:
    """Fire-and-forget audit event emission via Postgres `audit_events`."""
    _pg_emit_event(
        str(trace_id) if trace_id is not None else None,
        component,
        event_type,
        payload,
    )
