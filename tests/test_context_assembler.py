from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexi.pipeline.context_assembler import (
    AssembledContext,
    _extract_entity_mentions,
    assemble_context,
    _dedupe_lines,
    _episode_line,
)


@pytest.fixture
def mock_stores():
    wm = MagicMock()
    wm.get_turns = AsyncMock(
        return_value=[
            {"role": "user", "content": "deploy the new service", "timestamp": "1"},
            {"role": "assistant", "content": "checking policies", "timestamp": "2"},
        ]
    )

    pg = MagicMock()
    pg.retrieve_similar = AsyncMock(
        return_value=[
            {
                "id": "ep-1",
                "summary": "previous deployment episode",
                "raw_text": "deployed service foo",
                "similarity": 0.85,
            }
        ]
    )
    pg.fetch_by_type = AsyncMock(return_value=[])
    pg.bump_recall = AsyncMock()

    gs = MagicMock()
    gs.get_entity_by_name = MagicMock(
        return_value={"metadata": {"entity_id": "ent-1", "name": "Gemma"}}
    )
    gs.query_entity_connections = MagicMock(
        return_value=[{"connected_name": "RTX 3090", "rel_type": "runs_on"}]
    )

    rs = MagicMock()
    rs.get_relationships = AsyncMock(return_value=[])

    sb = MagicMock()
    sb.read_recent = AsyncMock(
        return_value=[{"data": "voice command heard", "source": "voice"}]
    )

    return wm, pg, gs, rs, sb


@pytest.mark.asyncio
async def test_assemble_context_basic(mock_stores):
    wm, pg, gs, rs, sb = mock_stores
    ctx = await assemble_context(
        session_id="test-sess",
        raw_input="deploy Gemma 4 model",
        working_memory=wm,
        pg_episodic=pg,
        graph_store=gs,
        relationship_store=rs,
        sensory_buffer=sb,
    )
    assert isinstance(ctx, AssembledContext)
    assert len(ctx.recent_turns) == 2
    assert len(ctx.relevant_episodes) == 1
    assert len(ctx.perception_snippets) == 1
    assert "deploy Gemma 4 model" in ctx.to_messages("deploy Gemma 4 model")[-1]["content"]


@pytest.mark.asyncio
async def test_assemble_context_to_messages(mock_stores):
    wm, pg, gs, rs, sb = mock_stores
    ctx = await assemble_context(
        session_id="test-sess",
        raw_input="hello",
        working_memory=wm,
        pg_episodic=pg,
        graph_store=gs,
        relationship_store=rs,
        sensory_buffer=sb,
    )
    msgs = ctx.to_messages("hello")
    assert len(msgs) >= 2
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "hello"}
    assert "Nexi" in msgs[0]["content"]


@pytest.mark.asyncio
async def test_assemble_context_with_proactivity(mock_stores):
    wm, pg, gs, rs, sb = mock_stores
    pe = MagicMock()
    pe.get_pending = AsyncMock(
        return_value=[
            MagicMock(
                message="Gemma 4 on i9 is not responding.",
                trigger="inference_down",
                priority=5,
            )
        ]
    )
    ctx = await assemble_context(
        session_id="test-sess",
        raw_input="status check",
        working_memory=wm,
        pg_episodic=pg,
        graph_store=gs,
        relationship_store=rs,
        sensory_buffer=sb,
        proactivity_engine=pe,
    )
    assert "Pending Observations" in ctx.system_prompt
    assert "Gemma 4 on i9 is not responding" in ctx.system_prompt


@pytest.mark.asyncio
async def test_assemble_context_empty_entities(mock_stores):
    wm, pg, gs, rs, sb = mock_stores
    gs.get_entity_by_name = MagicMock(return_value=None)
    ctx = await assemble_context(
        session_id="test-sess",
        raw_input="do something",
        working_memory=wm,
        pg_episodic=pg,
        graph_store=gs,
        relationship_store=rs,
        sensory_buffer=sb,
    )
    assert isinstance(ctx, AssembledContext)
    assert ctx.entity_context == []


@pytest.mark.asyncio
async def test_assemble_context_empty_turns(mock_stores):
    wm, pg, gs, rs, sb = mock_stores
    wm.get_turns = AsyncMock(return_value=[])
    ctx = await assemble_context(
        session_id="new-sess",
        raw_input="first message",
        working_memory=wm,
        pg_episodic=pg,
        graph_store=gs,
        relationship_store=rs,
        sensory_buffer=sb,
    )
    assert ctx.recent_turns == []


def test_extract_entity_mentions():
    mentions = _extract_entity_mentions("Deploy Gemma 4 on RTX 3090")
    assert "Gemma" in mentions or "Deploy Gemma" in mentions
    mentions2 = _extract_entity_mentions("hello world")
    assert mentions2 == []


@pytest.mark.asyncio
async def test_identity_facts_injected_from_store(mock_stores):
    wm, pg, gs, rs, sb = mock_stores
    pg.fetch_by_type = AsyncMock(
        return_value=[
            {"raw_text": "XNCH is the private AI orchestration platform", "summary": ""},
            {"raw_text": "vLLM Ornith-1.0-35B on port 8082", "summary": ""},
        ]
    )
    ctx = await assemble_context(
        session_id="s1", raw_input="what should we build next",
        working_memory=wm, pg_episodic=pg, graph_store=gs,
        relationship_store=rs, sensory_buffer=sb,
    )
    assert "## Identity" in ctx.system_prompt
    assert "XNCH is the private AI orchestration platform" in ctx.system_prompt
    assert "vLLM Ornith-1.0-35B on port 8082" in ctx.system_prompt


@pytest.mark.asyncio
async def test_identity_facts_fallback_to_seeder(mock_stores):
    wm, pg, gs, rs, sb = mock_stores
    ctx = await assemble_context(
        session_id="s1", raw_input="hello",
        working_memory=wm, pg_episodic=pg, graph_store=gs,
        relationship_store=rs, sensory_buffer=sb,
    )
    assert "## Identity" in ctx.system_prompt
    assert "XNCH is the private AI orchestration platform" in ctx.system_prompt


@pytest.mark.asyncio
async def test_recall_min_score_passed_to_retrieve_similar(mock_stores, monkeypatch):
    wm, pg, gs, rs, sb = mock_stores
    captured: dict = {}

    async def _retrieve(**kwargs):
        captured.update(kwargs)
        return [{"id": "ep-1", "summary": "x", "raw_text": "y", "similarity": 0.8}]

    pg.retrieve_similar = AsyncMock(side_effect=_retrieve)
    monkeypatch.setenv("XNCH_MEMORY_RECALL_MIN_SCORE", "0.5")
    await assemble_context(
        session_id="s1", raw_input="deploy the new service",
        working_memory=wm, pg_episodic=pg, graph_store=gs,
        relationship_store=rs, sensory_buffer=sb,
    )
    assert captured["min_score"] == 0.5


@pytest.mark.asyncio
async def test_recall_min_score_default(mock_stores, monkeypatch):
    wm, pg, gs, rs, sb = mock_stores
    monkeypatch.delenv("XNCH_MEMORY_RECALL_MIN_SCORE", raising=False)
    captured: dict = {}

    async def _retrieve(**kwargs):
        captured.update(kwargs)
        return [{"id": "ep-1", "summary": "x", "raw_text": "y", "similarity": 0.8}]

    pg.retrieve_similar = AsyncMock(side_effect=_retrieve)
    await assemble_context(
        session_id="s1", raw_input="deploy the new service",
        working_memory=wm, pg_episodic=pg, graph_store=gs,
        relationship_store=rs, sensory_buffer=sb,
    )
    assert captured["min_score"] == 0.35


@pytest.mark.asyncio
async def test_openclaw_summary_uses_raw_text_and_dedupes(mock_stores):
    wm, pg, gs, rs, sb = mock_stores
    pg.retrieve_similar = AsyncMock(
        return_value=[
            {
                "id": "ep-1",
                "summary": "OpenClaw chat: Build something",
                "raw_text": "Build something\nSure, what should we build?",
                "similarity": 0.9,
            },
            {
                "id": "ep-2",
                "summary": "OpenClaw chat: Build something",
                "raw_text": "Build something\nSure, what should we build?",
                "similarity": 0.8,
            },
        ]
    )
    ctx = await assemble_context(
        session_id="s1", raw_input="what should we build next",
        working_memory=wm, pg_episodic=pg, graph_store=gs,
        relationship_store=rs, sensory_buffer=sb,
    )
    assert len(ctx.relevant_episodes) == 1
    assert "Build something\nSure, what should we build?" in ctx.relevant_episodes[0]


def test_episode_line_plain_summary():
    line = _episode_line({"summary": "deployed foo", "raw_text": "full raw text"})
    assert line == "deployed foo"


def test_episode_line_openclaw_prefers_raw_text():
    line = _episode_line({
        "summary": "OpenClaw chat: Build something",
        "raw_text": "Build something\nSure, what should we build?",
    })
    assert line.startswith("Build something\n")


def test_episode_line_truncates_long_raw():
    line = _episode_line({"summary": "OpenClaw chat: x", "raw_text": "z" * 500})
    assert len(line) <= 300


def test_dedupe_lines_keeps_first():
    assert _dedupe_lines(["a", "b", "a", "", "c"]) == ["a", "b", "c"]


def test_assembled_context_defaults():
    ctx = AssembledContext()
    assert ctx.system_prompt == ""
    assert ctx.recent_turns == []
    assert ctx.relevant_episodes == []
    assert ctx.entity_context == []
    assert ctx.relationship_context == []
    assert ctx.perception_snippets == []
