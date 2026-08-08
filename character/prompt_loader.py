from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_PERSONA_PATH = Path(__file__).parent / "persona.yaml"
_CAPABILITIES_PATH = Path(__file__).parent / "capabilities.yaml"
_FACTS_PATH = Path(__file__).parent / "identity_facts.yaml"
_DEFAULT_IMPORTANCE = 2.0


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_persona() -> dict[str, Any]:
    """Identity and communication style — the stable persona signal."""
    return _load_yaml(_PERSONA_PATH)


def load_capabilities() -> dict[str, Any]:
    """Operational capabilities: hosts, filesystem, tools, tool routing."""
    return _load_yaml(_CAPABILITIES_PATH)


def get_identity_fact_records() -> list[dict[str, Any]]:
    """Return identity facts for episodic seeding: {raw_text, importance, type}."""
    facts = _load_yaml(_FACTS_PATH)
    records: list[dict[str, Any]] = []
    for item in facts.get("identity_facts") or []:
        if isinstance(item, str):
            text = item.strip()
            importance = _DEFAULT_IMPORTANCE
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            importance = float(item.get("importance", _DEFAULT_IMPORTANCE))
        else:
            continue
        if text:
            records.append(
                {"type": "identity", "raw_text": text, "importance": importance}
            )
    return records


def get_identity_fact_texts() -> list[str]:
    return [r["raw_text"] for r in get_identity_fact_records()]


def _format_capabilities(cap: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    summary = (cap.get("summary") or "").strip()
    if summary:
        lines.append(summary)

    hosts = cap.get("hosts") or {}
    if hosts:
        lines.append("Hosts:")
        for key, desc in hosts.items():
            lines.append(f"- {key}: {desc}")

    fs = cap.get("filesystem") or {}
    if fs:
        lines.append(
            f"Filesystem: read-only under {fs.get('path_root', '/home/x-nch')}; "
            f"use prefix {fs.get('path_prefix', '')} for repo files."
        )
        for ex in fs.get("example_paths") or []:
            lines.append(f"  e.g. {ex}")

    tools = cap.get("tools") or {}
    if tools:
        lines.append("Tools (invoke when you need ground truth):")
        for group, items in tools.items():
            if not items:
                continue
            lines.append(f"- {group}:")
            for item in items:
                lines.append(f"  - {item}")

    return lines


def build_system_prompt(
    session_memory: list[dict] | None = None,
    recent_entities: list[str] | None = None,
    identity_facts: list[str] | None = None,
    include_capabilities: bool = False,
) -> str:
    persona = load_persona()
    identity = persona["identity"]
    style = persona["communication_style"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts = [f"You are {identity['name']}.", identity["persona"]]
    parts.append("")
    parts.append(f"You address the user as {identity['address_user_as']}.")
    parts.append(f"Communication: {style['verbosity']}, {style['tone']}.")
    parts.append(f"Current time: {now}")
    parts.append("")

    if include_capabilities:
        cap_lines = _format_capabilities(load_capabilities())
        if cap_lines:
            parts.append("## Capabilities")
            parts.extend(cap_lines)
            parts.append("")

    never_do = style.get("never_do") or []
    if never_do:
        parts.append("## Rules (never do)")
        for rule in never_do:
            parts.append(f"- {rule}")
        parts.append("")

    facts = identity_facts if identity_facts is not None else get_identity_fact_texts()
    if facts:
        parts.append("## Identity")
        for fact in facts:
            parts.append(f"- {fact}")
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
    return build_system_prompt(include_capabilities=True)
