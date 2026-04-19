"""Contract 5 — rule-based option generator tests."""
from nexi.adapters.model_adapter import _rule_based_options
from nexi.models.intent import IntentClass


def test_always_produces_exactly_3():
    for cls in IntentClass:
        options = _rule_based_options(cls, "test-entity")
        assert len(options) == 3, f"Expected 3 options for {cls}, got {len(options)}"


def test_all_reversible():
    for cls in IntentClass:
        options = _rule_based_options(cls, "test-entity")
        assert all(opt.reversible for opt in options)


def test_no_forbidden_action_types():
    forbidden = {"RUN_COMMAND", "RUN_SCRIPT", "DEPLOY", "ROLLBACK", "DELETE_FILE", "MUTATE"}
    for cls in IntentClass:
        options = _rule_based_options(cls, "test-entity")
        for opt in options:
            assert opt.action_type not in forbidden, (
                f"Forbidden action_type {opt.action_type} in {cls} options"
            )


def test_payload_hash_is_sha256():
    options = _rule_based_options(IntentClass.EXECUTION, "test-entity")
    for opt in options:
        assert opt.payload_hash.startswith("sha256:")


def test_side_effects_at_most_one():
    for cls in IntentClass:
        options = _rule_based_options(cls, "test-entity")
        for opt in options:
            assert len(opt.estimated_side_effects) <= 1
