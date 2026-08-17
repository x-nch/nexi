"""Plan enrichment — option_generator surfaces experiential lessons in context."""
from datetime import datetime, timezone
from uuid import uuid4

from nexi.models import ContextManifest, ExperienceRef
from nexi.pipeline.option_generator import _build_context_summary


def _make_experience(lesson, verdict, confidence, intent_class="EXECUTION"):
    return ExperienceRef(
        experience_id=uuid4(),
        context_signature="sha256:abc",
        intent_class=intent_class,
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="operator",
        outcome="FAILURE",
        lesson=lesson,
        insight="insight",
        verdict=verdict,
        confidence=confidence,
        applicability="EXECUTION|DEPLOY|SERVICE",
        created_at=datetime.now(timezone.utc),
    )


def test_context_summary_includes_recent_lessons_when_present():
    manifest = ContextManifest(
        session_id=uuid4(),
        system_state_version="v1.0.0",
        experiences=[
            _make_experience("Rollback first, then stage", "MODIFY", 0.9),
            _make_experience("Backup before any deploy", "ALLOW_WITH_WARNINGS", 0.7),
        ],
    )

    summary = _build_context_summary(manifest)

    assert "recent_lessons" in summary
    lessons_text = " ".join(summary["recent_lessons"])
    assert "Rollback first, then stage" in lessons_text
    assert "MODIFY" in lessons_text


def test_context_summary_lessons_ranked_by_confidence():
    manifest = ContextManifest(
        session_id=uuid4(),
        system_state_version="v1.0.0",
        experiences=[
            _make_experience("low confidence lesson", "ALLOW", 0.5),
            _make_experience("high confidence lesson", "BLOCK", 0.95),
        ],
    )

    summary = _build_context_summary(manifest)

    lessons = summary["recent_lessons"]
    assert lessons.index("high confidence lesson (BLOCK)") < lessons.index("low confidence lesson (ALLOW)")


def test_context_summary_omits_lessons_when_empty():
    manifest = ContextManifest(
        session_id=uuid4(),
        system_state_version="v1.0.0",
        experiences=[],
    )

    summary = _build_context_summary(manifest)

    assert "recent_lessons" not in summary
