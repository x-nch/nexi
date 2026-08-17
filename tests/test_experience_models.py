"""ExperienceRef model + ContextManifest.experiences round-trip tests."""
from datetime import datetime, timezone
from uuid import uuid4

from nexi.models import ContextManifest, ExperienceRef


async def test_experience_ref_parses():
    exp = ExperienceRef(
        experience_id=uuid4(),
        context_signature="sha256:abc",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="OPERATOR",
        outcome="FAILURE",
        lesson="Rollback first, then stage",
        insight="Staging directly caused outage",
        verdict="MODIFY",
        confidence=0.8,
        applicability="EXECUTION|DEPLOY|SERVICE",
        created_at=datetime.now(timezone.utc),
    )
    assert exp.lesson == "Rollback first, then stage"
    assert exp.confidence == 0.8


async def test_context_manifest_accepts_experiences():
    exp = ExperienceRef(
        experience_id=uuid4(),
        context_signature="sha256:abc",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="OPERATOR",
        outcome="FAILURE",
        lesson="Rollback first",
        insight="Staging directly caused outage",
        verdict="MODIFY",
        confidence=0.8,
        applicability="EXECUTION|DEPLOY|SERVICE",
        created_at=datetime.now(timezone.utc),
    )
    manifest = ContextManifest(
        session_id=uuid4(),
        system_state_version="v1.0.0",
        experiences=[exp],
    )
    assert len(manifest.experiences) == 1
    assert manifest.experiences[0].verdict == "MODIFY"


async def test_context_manifest_experiences_default_empty():
    manifest = ContextManifest(session_id=uuid4(), system_state_version="v1.0.0")
    assert manifest.experiences == []
