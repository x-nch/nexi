from __future__ import annotations

import logging

from nexi.character.prompt_loader import get_identity_fact_records

logger = logging.getLogger(__name__)

# Backward-compatible export for tests and imports.
IDENTITY_FACTS = get_identity_fact_records()


async def sync_identity_memories(episodic_store) -> int:
    """Insert identity episodes from identity_facts.yaml that are not already stored."""
    from xnch.memory.pg_episodic_store import PgEpisodicStore

    if not isinstance(episodic_store, PgEpisodicStore):
        return 0

    facts = get_identity_fact_records()
    if not facts:
        return 0

    existing = await episodic_store.fetch_by_type("identity", limit=200)
    existing_texts = {(e.get("raw_text") or "").strip() for e in existing}

    added = 0
    for fact in facts:
        text = fact["raw_text"].strip()
        if not text or text in existing_texts:
            continue
        await episodic_store.store_episode(
            type_=fact["type"],
            raw_text=text,
            importance=fact.get("importance", 2.0),
        )
        existing_texts.add(text)
        added += 1

    if added:
        logger.info("Synced %d new identity fact(s) from identity_facts.yaml", added)
    return added


async def seed_identity_memories(episodic_store) -> int:
    """First-boot and ongoing sync — adds missing YAML identity facts only."""
    return await sync_identity_memories(episodic_store)
