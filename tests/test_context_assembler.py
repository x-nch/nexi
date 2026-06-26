from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from nexi.pipeline.context_assembler import (
    AssembledContext,
    _extract_entity_mentions,
    assemble_context,
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


def test_assembled_context_defaults():
    ctx = AssembledContext()
    assert ctx.system_prompt == ""
    assert ctx.recent_turns == []
    assert ctx.relevant_episodes == []
    assert ctx.entity_context == []
    assert ctx.relationship_context == []
    assert ctx.perception_snippets == []
