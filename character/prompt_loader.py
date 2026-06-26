from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml


_CHARACTER_PATH = Path(__file__).parent / "nexi_character.yaml"


def load_character() -> dict:
    with open(_CHARACTER_PATH) as f:
        return yaml.safe_load(f)


def build_system_prompt(
    session_memory: list[dict] | None = None,
    recent_entities: list[str] | None = None,
) -> str:
    char = load_character()
    identity = char["identity"]
    style = char["communication_style"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts = [f"You are {identity['name']}.", identity["persona"]]
    parts.append("")
    parts.append(f"You address the user as {identity['address_user_as']}.")
    parts.append(f"Communication: {style['verbosity']}, {style['tone']}.")
    parts.append(f"Current time: {now}")
    parts.append("")

    if session_memory:
        parts.append("## Session Context")
        for mem in session_memory[-5:]:
            summary = mem.get("summary", mem.get("raw_text", ""))
            parts.append(f"- {summary}")
        parts.append("")

    if recent_entities:
        parts.append("## Known Entities")
        for ent in recent_entities:
            parts.append(f"- {ent}")
        parts.append("")

    return "\n".join(parts)


def get_nexi_system_prompt() -> str:
    return build_system_prompt()
