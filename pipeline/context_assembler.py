from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nexi.character.prompt_loader import get_identity_fact_texts
from nexi.character.prompt_loader import build_system_prompt

DEFAULT_RECALL_MIN_SCORE = 0.35
_RECALL_MIN_SCORE_ENV = "XNCH_MEMORY_RECALL_MIN_SCORE"


def _recall_min_score() -> float:
    raw = os.environ.get(_RECALL_MIN_SCORE_ENV)
    if raw is None:
        return DEFAULT_RECALL_MIN_SCORE
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_RECALL_MIN_SCORE


def _episode_line(episode: dict[str, Any]) -> str:
    """Render one episode for the system prompt.

    Old episodes were stored with `OpenClaw chat: {user}` summaries; prefer the
    (truncated) raw_text for those so the model sees real content, not the junk
    summary. Other episodes use summary when present.
    """
    summary = (episode.get("summary") or "").strip()
    raw_text = (episode.get("raw_text") or "").strip()
    if summary.startswith("OpenClaw chat:"):
        return (raw_text or summary)[:300]
    return (summary or raw_text)[:300]


def _dedupe_lines(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        if line and line not in seen:
            seen.add(line)
            result.append(line)
    return result


async def _load_identity_facts(pg_episodic) -> list[str]:
    """Identity facts from the episodic store, falling back to the seeder list."""
    try:
        if pg_episodic is not None and hasattr(pg_episodic, "fetch_by_type"):
            episodes = await pg_episodic.fetch_by_type("identity", limit=20)
            facts = [
                (ep.get("raw_text") or ep.get("summary") or "").strip()
                for ep in episodes or []
            ]
            facts = _dedupe_lines([f for f in facts if f])
            if facts:
                return facts
    except Exception:
        pass
    return get_identity_fact_texts()


@dataclass
class AssembledContext:
    system_prompt: str = ""
    recent_turns: list[dict] = field(default_factory=list)
    relevant_episodes: list[str] = field(default_factory=list)
    entity_context: list[dict] = field(default_factory=list)
    relationship_context: list[dict] = field(default_factory=list)
    perception_snippets: list[str] = field(default_factory=list)

    def to_messages(self, raw_input: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system_prompt}]
        for t in self.recent_turns:
            msgs.append({"role": t.get("role", "user"), "content": t.get("content", "")})
        msgs.append({"role": "user", "content": raw_input})
        return msgs


def _extract_entity_mentions(text: str) -> list[str]:
    import re
    matches = re.findall(r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b', text)
    seen = set()
    entities = []
    for m in matches:
        if m.lower() not in seen and len(m) > 2:
            seen.add(m.lower())
            entities.append(m)
    return entities


async def assemble_context(
    session_id: str,
    raw_input: str,
    working_memory,
    pg_episodic,
    graph_store,
    relationship_store,
    sensory_buffer,
    proactivity_engine=None,
    recall_query: str | None = None,
    min_score: float | None = None,
) -> AssembledContext:
    ctx = AssembledContext()
    min_score = _recall_min_score() if min_score is None else min_score

    recent_turns = await working_memory.get_turns(session_id, last_n=20)
    ctx.recent_turns = recent_turns

    retrieve_text = recall_query or raw_input
    relevant = await pg_episodic.retrieve_similar(
        query_text=retrieve_text, top_k=5, min_score=min_score
    )
    ctx.relevant_episodes = _dedupe_lines(
        _episode_line(r) for r in relevant
    )

    entities = _extract_entity_mentions(raw_input)
    if entities:
        for ent in entities:
            entity_node = graph_store.get_entity_by_name(ent)
            if entity_node:
                eid = entity_node["metadata"].get("entity_id", ent)
                connections = graph_store.query_entity_connections(eid)
                ctx.entity_context.extend(connections)
                rels = await relationship_store.get_relationships(eid)
                ctx.relationship_context.extend(
                    {"entity_a": r.entity_a_id, "entity_b": r.entity_b_id,
                     "type": r.relationship_type, "strength": r.strength}
                    for r in rels
                )

    for r in relevant:
        rid = r.get("id")
        if rid:
            await pg_episodic.bump_recall(rid)

    recent_perceptions = await sensory_buffer.read_recent("voice", limit=3)
    ctx.perception_snippets = [p.get("data", "") for p in recent_perceptions]

    session_memories = [{"summary": s} for s in ctx.relevant_episodes[:5]]
    session_entities = [f"{c.get('connected_name', '')} ({c.get('rel_type', '')})" for c in ctx.entity_context[:5]]
    identity_facts = await _load_identity_facts(pg_episodic)
    ctx.system_prompt = build_system_prompt(
        session_memory=session_memories,
        recent_entities=session_entities,
        identity_facts=identity_facts,
    )

    if proactivity_engine is not None:
        pending = await proactivity_engine.get_pending()
        if pending:
            obs_lines = "\n".join(e.message for e in pending)
            ctx.system_prompt += f"\n\n## Pending Observations\n{obs_lines}"

    ts = datetime.now(timezone.utc).isoformat()
    ctx.system_prompt += f"\n\nContext assembled at {ts}"

    return ctx
