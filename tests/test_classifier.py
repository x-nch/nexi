"""Unit tests for LiteLLM routing classifier (no LLM calls)."""
from unittest.mock import patch

import pytest

from xnch.routing.classifier import classify_request, ModelRoute


@pytest.fixture(autouse=True)
def no_routing_cache():
    with patch("xnch.routing.classifier._get_redis", return_value=None):
        yield


class TestClassifyRequest:
    def test_default_route(self):
        route = classify_request("list services", "VIEWER", {})
        assert route.model_name == "ornith"
        assert "default" in route.reason

    def test_privacy_sensitive_routes_to_local(self):
        route = classify_request("show user data", "OPERATOR", {"privacy_sensitive": True})
        assert route.model_name == "ornith"
        assert "privacy_sensitive" in route.reason

    def test_execution_routes_to_local(self):
        route = classify_request("deploy model", "OPERATOR", {"intent_class": "EXECUTION"})
        assert route.model_name == "ornith"
        assert "EXECUTION" in route.reason

    def test_decision_high_complexity_routes_to_ornith(self):
        route = classify_request("design architecture", "ADMIN", {
            "intent_class": "DECISION",
            "complexity_score": 0.85,
        })
        assert route.model_name == "ornith"
        assert "complexity" in route.reason

    def test_decision_low_complexity_routes_to_local(self):
        route = classify_request("simple query", "ADMIN", {
            "intent_class": "DECISION",
            "complexity_score": 0.3,
        })
        assert route.model_name == "ornith"

    def test_execution_overrides_privacy_sensitive(self):
        route = classify_request("delete database", "ADMIN", {
            "intent_class": "EXECUTION",
            "privacy_sensitive": True,
        })
        assert route.model_name == "ornith"
        assert "privacy_sensitive" in route.reason

    def test_unknown_intent_class_defaults_to_local(self):
        route = classify_request("random input", "VIEWER", {"intent_class": "UNKNOWN"})
        assert route.model_name == "ornith"

    @pytest.mark.parametrize("role", ["ADMIN", "OPERATOR", "VIEWER", "AGENT"])
    def test_actor_role_does_not_affect_routing(self, role):
        route = classify_request("list services", role, {"intent_class": "QUERY"})
        assert route.model_name == "ornith"

    def test_model_route_is_dataclass(self):
        route = classify_request("test", "ADMIN", {})
        assert isinstance(route, ModelRoute)
        assert hasattr(route, "model_name")
        assert hasattr(route, "reason")

    def test_redis_recall_used_when_exact_match(self):
        recalled = {
            "raw_input": "deploy model xyz",
            "model_name": "ornith",
            "reason": "recalled decision",
            "intent_class": "DECISION",
        }
        with patch("xnch.routing.classifier._cache_lookup", return_value=ModelRoute(
            model_name="ornith", reason="recalled: recalled decision",
        )):
            route = classify_request("deploy model xyz", "ADMIN", {"intent_class": "DECISION"})
            assert route.model_name == "ornith"
            assert "recalled" in route.reason
