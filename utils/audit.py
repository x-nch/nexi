"""Fire-and-forget audit emission into xnch's Postgres audit_events.

xnch's storage deps (aiosqlite et al.) are intentionally absent from nexi's
isolated venv, so the xnch import is lazy + optional: when running standalone,
emission is a no-op. Alert delivery to xnch rides POST /admin/alerts over
HTTP regardless — this path only enriches events that originate in-process.
"""
from uuid import UUID


def emit_event(
    trace_id: UUID | str,
    component: str,
    event_type: str,
    payload: dict | None = None,
) -> None:
    try:
        from xnch.memory.audit_store import emit_event as _pg_emit_event
    except ImportError:
        return  # standalone nexi — no xnch storage stack available
    try:
        _pg_emit_event(
            str(trace_id) if trace_id is not None else None,
            component,
            event_type,
            payload,
        )
    except Exception:  # noqa: BLE001 — fire-and-forget by contract
        pass
