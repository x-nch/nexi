"""Step 4 — Context Manifest loading via xnch."""
from ..adapters.xnch_client import XnchClient
from ..models import SessionContext, Intent, ContextManifest
from ..utils.audit import emit_event


async def load_context(
    xnch: XnchClient,
    session: SessionContext,
    intent: Intent,
) -> ContextManifest:
    """Hard stop if xnch memory store is unavailable — no fallback to empty context."""
    emit_event(session.trace_id, "context_loader", "MANIFEST_REQUEST_START")

    manifest = await xnch.read_context(
        session,
        intent_class=intent.intent_class,
        target_entity_id=intent.target_entity_id,
        target_entity_class=intent.target_entity_class,
    )

    emit_event(session.trace_id, "context_loader", "MANIFEST_PINNED",
               {"manifest_id": str(manifest.manifest_id),
                "episodes": len(manifest.episodes),
                "patterns": len(manifest.patterns),
                "policies": len(manifest.policies)})
    return manifest
