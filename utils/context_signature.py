import hashlib


def compute_context_signature(
    intent_class: str,
    action_type: str,
    entity_class: str,
    actor_role: str,
) -> str:
    """Contract 3: deterministic context tuple identifier."""
    if not all([intent_class, action_type, entity_class, actor_role]):
        raise ValueError("All four fields are required for context signature computation")

    canonical = "|".join([
        intent_class.lower(),
        action_type.lower(),
        entity_class.lower(),
        actor_role.lower(),
    ])
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
