"""Steps 7–8 — Evaluator tests."""
from uuid import uuid4

from nexi.pipeline.evaluator import Evaluator
from nexi.models.intent import IntentClass, Urgency
from nexi.models.intent import Intent, ActionType
from nexi.models.options import PolicyVerdict, PolicyDryRunResponse
from nexi.models.outcomes import ContextManifest, PatternRef
from nexi.models.session import SessionContext, Actor, ActorRole
from nexi.adapters.model_adapter import _rule_based_options
from nexi.utils.context_signature import compute_context_signature


def _make_session() -> SessionContext:
    return SessionContext(
        session_id=uuid4(),
        trace_id=uuid4(),
        actor=Actor(id="test-user", role=ActorRole.OPERATOR, capability_set=["DEPLOY", "READ"]),
        system_state_version="v1.0.0",
        policy_version="v1.0.0",
        idempotency_key=uuid4(),
        raw_input="deploy service foo",
    )


def _make_intent(session: SessionContext) -> Intent:
    return Intent(
        session_id=session.session_id,
        intent_class=IntentClass.EXECUTION,
        action_type=ActionType.DEPLOY,
        target_entity_id="foo",
        target_entity_class="SERVICE",
        urgency=Urgency.NORMAL,
        ambiguity_score=0.1,
        raw_input_hash="sha256:abc",
    )


def _make_manifest(session: SessionContext, intent: Intent) -> ContextManifest:
    sig = compute_context_signature(
        intent.intent_class, intent.action_type,
        intent.target_entity_class, session.actor.role,
    )
    return ContextManifest(
        session_id=session.session_id,
        system_state_version=session.system_state_version,
        patterns=[PatternRef(
            pattern_id=uuid4(),
            context_signature=sig,
            success_rate=0.8,
            confidence=0.7,
            observation_count=15,
        )],
    )


def test_composite_scores_between_0_and_1():
    session = _make_session()
    intent = _make_intent(session)
    manifest = _make_manifest(session, intent)
    evaluator = Evaluator()

    options = _rule_based_options(IntentClass.EXECUTION, "foo")
    pairs = [
        (opt, PolicyDryRunResponse(
            option_id=opt.option_id,
            session_id=session.session_id,
            verdict=PolicyVerdict.ALLOW,
            policy_refs=[],
        ))
        for opt in options
    ]

    evaluated = evaluator.score(pairs, intent, manifest, session)
    for ev in evaluated:
        assert 0.0 <= ev.composite_score <= 1.0


def test_weights_sum_respected():
    evaluator = Evaluator()
    weights = evaluator._resolve_weights("EXECUTION")
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-9
