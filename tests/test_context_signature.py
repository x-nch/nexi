"""Contract 3 — context_signature determinism tests."""
from nexi.utils.context_signature import compute_context_signature


def test_deterministic():
    a = compute_context_signature("EXECUTION", "deploy", "ML_MODEL", "OPERATOR")
    b = compute_context_signature("EXECUTION", "deploy", "ML_MODEL", "OPERATOR")
    assert a == b


def test_case_insensitive():
    lower = compute_context_signature("execution", "deploy", "ml_model", "operator")
    upper = compute_context_signature("EXECUTION", "DEPLOY", "ML_MODEL", "OPERATOR")
    assert lower == upper


def test_sha256_prefix():
    sig = compute_context_signature("EXECUTION", "deploy", "ML_MODEL", "OPERATOR")
    assert sig.startswith("sha256:")


def test_field_order_matters():
    a = compute_context_signature("EXECUTION", "deploy", "ML_MODEL", "OPERATOR")
    b = compute_context_signature("deploy", "EXECUTION", "ML_MODEL", "OPERATOR")
    assert a != b


def test_known_value():
    import hashlib
    canonical = "execution|deploy|service|operator"
    expected = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
    result = compute_context_signature("EXECUTION", "deploy", "service", "OPERATOR")
    assert result == expected


def test_missing_field_raises():
    import pytest
    with pytest.raises(ValueError):
        compute_context_signature("EXECUTION", "", "ML_MODEL", "OPERATOR")
