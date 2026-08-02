from __future__ import annotations

IDENTITY_FACTS = [
    {
        "type": "identity",
        "raw_text": "ck-san is a Senior Platform Engineer at Rakuten India in Bengaluru",
        "importance": 2.0,
    },
    {
        "type": "identity",
        "raw_text": "ck-san is pivoting to AI Infrastructure and FDE roles",
        "importance": 2.0,
    },
    {
        "type": "identity",
        "raw_text": "XNCH is the private AI orchestration platform; Nexi is the product interface layer",
        "importance": 2.0,
    },
    {
        "type": "identity",
        "raw_text": "Primary local model: Gemma 4 26B on RTX 3090 at ~135 tok/s",
        "importance": 2.0,
    },
    {
        "type": "identity",
        "raw_text": "ck-san prefers Firefox; never suggest Chrome-based tooling",
        "importance": 2.0,
    },
    {
        "type": "identity",
        "raw_text": "ck-san works solo on XNCH and Nexi — no team, sole ownership",
        "importance": 2.0,
    },
    {
        "type": "identity",
        "raw_text": "Chitradurga relocation is a long-term consideration for remote work lifestyle",
        "importance": 2.0,
    },
]


async def seed_identity_memories(episodic_store) -> int:
    from xnch.memory.pg_episodic_store import PgEpisodicStore

    if not isinstance(episodic_store, PgEpisodicStore):
        return 0

    if await episodic_store.has_episode_of_type("identity"):
        return 0

    count = 0
    for fact in IDENTITY_FACTS:
        await episodic_store.store_episode(
            type_=fact["type"],
            raw_text=fact["raw_text"],
            importance=fact["importance"],
        )
        count += 1
    return count
