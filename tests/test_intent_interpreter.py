"""Contract 4 — intent classifier tests with mocked LiteLLM."""
import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from nexi.pipeline.intent_interpreter import IntentInterpreter, ClarificationRequired
from nexi.models.intent import Intent, IntentClass


_LLM_CLASSIFY_RESPONSE = {
    "choices": [{
        "message": {
            "content": (
                '{"intent_class": "EXECUTION", "action_type": "DEPLOY", '
                '"entity_class": "ML_MODEL", "urgency": "HIGH", '
                '"entity_id": "llama3-8b", "clarifications_needed": []}'
            )
        }
    }]
}

_LLM_AMBIGUOUS_RESPONSE = {
    "choices": [{
        "message": {
            "content": (
                '{"intent_class": "QUERY", "action_type": "ANALYZE", '
                '"entity_class": "RESOURCE", "urgency": "NORMAL", '
                '"entity_id": "production", '
                '"clarifications_needed": ["Do you want to read or modify?", '
                '"Which specific resource?"]}'
            )
        }
    }]
}


@pytest.fixture
def interpreter():
    return IntentInterpreter()


@pytest.fixture(autouse=True)
def _disable_intent_cache():
    """Intent interpreter caches classifications in Redis; keep unit tests isolated."""
    with patch("nexi.pipeline.intent_interpreter._get_redis", return_value=None):
        yield


@pytest.fixture
def mock_httpx_client():
    """Patch httpx.AsyncClient so all instances return a controlled mock."""
    with patch("nexi.pipeline.intent_interpreter.httpx.AsyncClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_cls.return_value.__aenter__.return_value = mock_instance
        yield mock_instance


@pytest.mark.asyncio
async def test_deploy_classifies_as_execution(interpreter):
    intent = await interpreter.interpret("deploy model llama3-8b to inference cluster", uuid4(), str(uuid4()))
    assert intent.intent_class == IntentClass.EXECUTION
    assert intent.action_type == "DEPLOY"
    assert intent.ambiguity_score == 0.0


@pytest.mark.asyncio
async def test_list_classifies_as_query(interpreter):
    intent = await interpreter.interpret("list all running services", uuid4(), str(uuid4()))
    assert intent.intent_class == IntentClass.QUERY


@pytest.mark.asyncio
async def test_escalate_classifies_as_escalation(interpreter):
    intent = await interpreter.interpret("escalate incident INS-42 to on-call", uuid4(), str(uuid4()))
    assert intent.intent_class == IntentClass.ESCALATION


@pytest.mark.asyncio
async def test_ambiguous_input_uses_llm(interpreter):
    with patch.object(interpreter, "_classify_with_llm") as mock_llm:
        mock_llm.return_value = Intent(
            session_id=uuid4(),
            intent_class=IntentClass.EXECUTION,
            action_type="DEPLOY",
            target_entity_id="llama3-8b",
            target_entity_class="ML_MODEL",
            urgency="HIGH",
            ambiguity_score=0.0,
            raw_input_hash="sha256:" + hashlib.sha256(b"xyzzyplex frobnicate").hexdigest(),
            raw_input="xyzzyplex frobnicate",
        )
        intent = await interpreter.interpret("xyzzyplex frobnicate", uuid4(), str(uuid4()))
        assert intent.intent_class == IntentClass.EXECUTION
        assert intent.action_type == "DEPLOY"
        mock_llm.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_ambiguous_raises_clarification(interpreter, mock_httpx_client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _LLM_AMBIGUOUS_RESPONSE
    mock_httpx_client.post.return_value = mock_resp

    with pytest.raises(ClarificationRequired) as exc_info:
        await interpreter._classify_with_llm(
            "something about production", uuid4(), str(uuid4()), "hash"
        )
    assert len(exc_info.value.questions) == 2
    assert "read or modify" in exc_info.value.questions[0]


@pytest.mark.asyncio
async def test_llm_classify_returns_intent(interpreter, mock_httpx_client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = _LLM_CLASSIFY_RESPONSE
    mock_httpx_client.post.return_value = mock_resp

    intent = await interpreter._classify_with_llm(
        "deploy llama to cluster", uuid4(), str(uuid4()), "hash"
    )
    assert intent.intent_class == IntentClass.EXECUTION
    assert intent.action_type == "DEPLOY"
    assert intent.ambiguity_score == 0.0


@pytest.mark.asyncio
async def test_llm_failure_falls_back(interpreter):
    async def fake_fail(*args, **kwargs):
        raise Exception("LLM unavailable")
    with patch.object(interpreter, "_classify_with_llm", fake_fail):
        intent = await interpreter.interpret(
            "something random to classify", uuid4(), str(uuid4())
        )
    assert intent.intent_class == IntentClass.QUERY
    assert intent.ambiguity_score == 0.5


@pytest.mark.asyncio
async def test_raw_input_hash_is_sha256(interpreter):
    raw = "deploy service foo"
    intent = await interpreter.interpret(raw, uuid4(), str(uuid4()))
    expected = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
    assert intent.raw_input_hash == expected


@pytest.mark.asyncio
async def test_rule_input_has_raw_input_and_no_clarifications(interpreter):
    intent = await interpreter.interpret("list all services", uuid4(), str(uuid4()))
    assert intent.raw_input == "list all services"
    assert intent.clarifications_needed == []


def test_clarification_required_has_questions():
    exc = ClarificationRequired(uuid4(), 0.85, questions=["What resource?"])
    assert exc.questions == ["What resource?"]
    assert "What resource?" in str(exc)


@pytest.mark.asyncio
async def test_rule_high_ambiguity_raises(interpreter):
    with patch("nexi.pipeline.intent_interpreter._rule_classify",
               return_value=(IntentClass.QUERY, "LIST", 0.2, 0.8)):
        with pytest.raises(ClarificationRequired) as exc_info:
            await interpreter.interpret("fuzzy input", uuid4(), str(uuid4()))
        assert exc_info.value.ambiguity_score == 0.8


@pytest.mark.asyncio
async def test_intent_cache_preserves_ambiguity_score():
    """Recalled intents must not reset ambiguity_score to 0.0."""
    from nexi.pipeline.intent_interpreter import _persist_intent, _recall_intent
    from nexi.models.intent import ActionType, Urgency

    raw = "something random to classify"
    intent = Intent(
        session_id=uuid4(),
        intent_class=IntentClass.QUERY,
        action_type=ActionType.ANALYZE,
        target_entity_id="unknown",
        target_entity_class="RESOURCE",
        urgency=Urgency.NORMAL,
        ambiguity_score=0.5,
        raw_input_hash="hash",
        raw_input=raw,
    )

    mock_redis = MagicMock()
    stored: dict[str, str] = {}

    def fake_set(key: str, value: str, ex: int) -> None:
        stored[key] = value

    mock_redis.get.side_effect = lambda key: stored.get(key)
    mock_redis.set.side_effect = fake_set

    with patch("nexi.pipeline.intent_interpreter._get_redis", return_value=mock_redis):
        _persist_intent(raw, intent)
        recalled = _recall_intent(raw)

    assert recalled is not None
    assert recalled.ambiguity_score == 0.5
