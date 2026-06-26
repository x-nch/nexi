"""Unit tests for LiteLLM routing classifier (no LLM calls)."""
from unittest.mock import patch

import pytest

from xnch.routing.classifier import classify_request, ModelRoute


@pytest.fixture(autouse=True)
def mock_agentmemory():
    with patch("xnch.routing.classifier.search_memory", return_value=[]):
        with patch("xnch.routing.classifier.create_memory"):
            yield


class TestClassifyRequest:
    def test_default_route(self):
        route = classify_request("list services", "VIEWER", {})
        assert route.model_name == "gemma4-local"
        assert "default" in route.reason

    def test_privacy_sensitive_routes_to_local(self):
        route = classify_request("show user data", "OPERATOR", {"privacy_sensitive": True})
        assert route.model_name == "gemma4-local"
        assert "privacy_sensitive" in route.reason

    def test_execution_routes_to_local(self):
        route = classify_request("deploy model", "OPERATOR", {"intent_class": "EXECUTION"})
        assert route.model_name == "gemma4-local"
        assert "EXECUTION" in route.reason

    def test_decision_high_complexity_routes_to_claude(self):
        route = classify_request("design architecture", "ADMIN", {
            "intent_class": "DECISION",
            "complexity_score": 0.85,
        })
        assert route.model_name == "claude-judgment"
        assert "complexity" in route.reason

    def test_decision_low_complexity_routes_to_local(self):
        route = classify_request("simple query", "ADMIN", {
            "intent_class": "DECISION",
            "complexity_score": 0.3,
        })
        assert route.model_name == "gemma4-local"

    def test_execution_overrides_privacy_sensitive(self):
        route = classify_request("delete database", "ADMIN", {
            "intent_class": "EXECUTION",
            "privacy_sensitive": True,
        })
        assert route.model_name == "gemma4-local"
        assert "privacy_sensitive" in route.reason

    def test_unknown_intent_class_defaults_to_local(self):
        route = classify_request("random input", "VIEWER", {"intent_class": "UNKNOWN"})
        assert route.model_name == "gemma4-local"

    @pytest.mark.parametrize("role", ["ADMIN", "OPERATOR", "VIEWER", "AGENT"])
    def test_actor_role_does_not_affect_routing(self, role):
        route = classify_request("list services", role, {"intent_class": "QUERY"})
        assert route.model_name == "gemma4-local"

    def test_model_route_is_dataclass(self):
        route = classify_request("test", "ADMIN", {})
        assert isinstance(route, ModelRoute)
        assert hasattr(route, "model_name")
        assert hasattr(route, "reason")

    def test_agentmemory_recall_used_when_exact_match(self):
        recalled = [{
            "document": (
                '{"raw_input": "deploy model xyz", "model_name": "claude-judgment", '
                '"reason": "recalled decision", "intent_class": "DECISION"}'
            ),
            "metadata": {"model_name": "claude-judgment"},
        }]
        with patch("xnch.routing.classifier.search_memory", return_value=recalled):
            route = classify_request("deploy model xyz", "ADMIN", {"intent_class": "DECISION"})
            assert route.model_name == "claude-judgment"
            assert "recalled" in route.reason
