from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from nexi.character.prompt_loader import build_system_prompt


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
) -> AssembledContext:
    ctx = AssembledContext()

    recent_turns = await working_memory.get_turns(session_id, last_n=20)
    ctx.recent_turns = recent_turns

    relevant = await pg_episodic.retrieve_similar(
        query_text=raw_input, top_k=5
    )
    ctx.relevant_episodes = [
        r.get("summary") or r.get("raw_text", "") for r in relevant
    ]

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

    session_memories = ctx.relevant_episodes[:5]
    session_entities = [f"{c.get('connected_name', '')} ({c.get('rel_type', '')})" for c in ctx.entity_context[:5]]
    ctx.system_prompt = build_system_prompt(
        session_memory=[{"summary": s} for s in session_memories],
        recent_entities=session_entities,
    )

    if proactivity_engine is not None:
        pending = await proactivity_engine.get_pending()
        if pending:
            obs_lines = "\n".join(e.message for e in pending)
            ctx.system_prompt += f"\n\n## Pending Observations\n{obs_lines}"

    ts = datetime.now(timezone.utc).isoformat()
    ctx.system_prompt += f"\n\nContext assembled at {ts}"

    return ctx
