"""Reflector — Summary step producing structured experiential lessons."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexi.pipeline.reflector import Reflector, ReflectionRecord


async def test_reflector_builds_reflection_from_outcome():
    llm = AsyncMock(return_value={
        "verdict": "MODIFY",
        "lesson": "Rollback first, then stage",
        "insight": "Staging directly caused outage",
        "applicability": "EXECUTION|DEPLOY|SERVICE",
    })
    write = AsyncMock(return_value=None)
    reflector = Reflector(llm_fn=llm, write_fn=write)

    result = await reflector.reflect(
        session_id="sess-1",
        trace_id="trace-1",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="operator",
        outcome="FAILURE",
        prediction_delta=0.4,
        context_summary={"scores": {"composite_score": 0.6}},
    )

    assert isinstance(result, ReflectionRecord)
    assert result.verdict == "MODIFY"
    assert result.lesson == "Rollback first, then stage"

    # LLM was given outcome + delta context
    prompt_kwargs = llm.call_args.kwargs
    assert prompt_kwargs["outcome"] == "FAILURE"
    assert prompt_kwargs["prediction_delta"] == 0.4


async def test_reflector_persists_via_write_fn():
    llm = AsyncMock(return_value={
        "verdict": "ALLOW",
        "lesson": "Lesson text",
        "insight": "Insight text",
        "applicability": "QUERY|LIST|FILE",
    })
    write = AsyncMock(return_value=None)
    reflector = Reflector(llm_fn=llm, write_fn=write)

    await reflector.reflect(
        session_id="sess-1",
        trace_id="trace-1",
        intent_class="QUERY",
        action_type="LIST",
        entity_class="FILE",
        actor_role="viewer",
        outcome="SUCCESS",
        prediction_delta=0.1,
        context_summary={},
    )

    write.assert_awaited_once()
    call_kwargs = write.call_args.kwargs
    assert call_kwargs["context_signature"]
    assert call_kwargs["lesson"] == "Lesson text"
    assert call_kwargs["verdict"] == "ALLOW"


async def test_reflector_persists_with_default_context_signature():
    llm = AsyncMock(return_value={
        "verdict": "BLOCK",
        "lesson": "L",
        "insight": "I",
        "applicability": "EXECUTION|DEPLOY|SERVICE",
    })
    write = AsyncMock(return_value=None)
    reflector = Reflector(llm_fn=llm, write_fn=write)

    await reflector.reflect(
        session_id="sess-1",
        trace_id="trace-1",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="operator",
        outcome="FAILURE",
        prediction_delta=0.5,
        context_summary={},
    )

    sig = write.call_args.kwargs["context_signature"]
    assert sig.startswith("sha256:")


async def test_build_reflector_wires_xnch_write_and_llm_proxy():
    """Production Reflector must call litellm proxy + XnchClient.write_experience."""
    from nexi.adapters.xnch_client import XnchClient
    from nexi.pipeline.reflector import build_reflector

    xnch = MagicMock(spec=XnchClient)
    reflector = build_reflector(xnch)

    assert isinstance(reflector, Reflector)
    assert reflector._write is xnch.write_experience


async def test_reflector_handles_llm_failure_without_raising():
    llm = AsyncMock(side_effect=RuntimeError("llm down"))
    write = AsyncMock(return_value=None)
    reflector = Reflector(llm_fn=llm, write_fn=write)

    result = await reflector.reflect(
        session_id="sess-1",
        trace_id="trace-1",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="operator",
        outcome="FAILURE",
        prediction_delta=0.5,
        context_summary={},
    )

    assert result is None
    write.assert_not_awaited()


async def test_reflector_ignores_missing_verdict_from_llm():
    llm = AsyncMock(return_value={"lesson": "no verdict"})
    write = AsyncMock(return_value=None)
    reflector = Reflector(llm_fn=llm, write_fn=write)

    result = await reflector.reflect(
        session_id="sess-1",
        trace_id="trace-1",
        intent_class="EXECUTION",
        action_type="DEPLOY",
        entity_class="SERVICE",
        actor_role="operator",
        outcome="FAILURE",
        prediction_delta=0.5,
        context_summary={},
    )

    assert result is None
    write.assert_not_awaited()
