from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from nexi.config import settings

logger = logging.getLogger(__name__)

_PERSONA_PATH = Path(__file__).parent / "persona.yaml"
_CAPABILITIES_PATH = Path(__file__).parent / "capabilities.yaml"
_FACTS_PATH = Path(__file__).parent / "identity_facts.yaml"
_DEFAULT_IMPORTANCE = 2.0

# Keys the auto-refresh generator owns. When a generated overlay is present these
# override the hand-maintained base; everything else (summary, voice, …) stays
# authoritative in capabilities.yaml.
_GENERATED_KEYS = ("hosts", "tools", "tool_routing", "filesystem", "status")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_persona() -> dict[str, Any]:
    """Identity and communication style — the stable persona signal."""
    return _load_yaml(_PERSONA_PATH)


def get_generated_overlay_path() -> Path:
    """Primary path the capability builder writes to (env-overridable)."""
    return Path(settings.capabilities_generated_path).expanduser()


def _load_generated_overlay() -> dict[str, Any] | None:
    """Best-effort read of the auto-generated overlay; falls back to repo copy."""
    candidates = [
        get_generated_overlay_path(),
        Path(__file__).parent / "capabilities.generated.yaml",
    ]
    for raw in candidates:
        path = Path(raw).expanduser()
        if not path.is_file():
            continue
        try:
            return _load_yaml(path)
        except Exception as exc:
            logger.warning("Failed to load generated capabilities overlay %s: %s", path, exc)
            return None
    return None


def load_capabilities() -> dict[str, Any]:
    """Operational capabilities: static base merged with the generated overlay."""
    base = _load_yaml(_CAPABILITIES_PATH)
    overlay = _load_generated_overlay()
    if overlay:
        for key in _GENERATED_KEYS:
            if key not in overlay:
                continue
            value = overlay[key]
            # Never let a partial/empty generated section clobber the richer base.
            if isinstance(value, dict) and not value:
                continue
            if value in (None, ""):
                continue
            base[key] = value
    return base


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

    routing = (cap.get("tool_routing") or "").strip()
    if routing:
        lines.append("Tool routing:")
        for line in routing.splitlines():
            line = line.strip()
            if line:
                lines.append(f"- {line}")

    status = cap.get("status") or {}
    if status:
        lines.append("Status (live infra probes):")
        lines.append(f"- checked_at: {status.get('checked_at', 'unknown')}")
        if status.get("healthy"):
            lines.append(f"- healthy: {', '.join(status['healthy'])}")
        if status.get("down"):
            lines.append(f"- DOWN: {', '.join(status['down'])}")

    voice = cap.get("voice") or {}
    if voice:
        lines.append("Voice (TTS — you CAN reply aloud):")
        lines.append(
            f"- push-to-talk; STT {voice.get('stt', 'faster-whisper')}; "
            f"TTS {voice.get('tts', 'piper')}"
        )
        for ep in voice.get("endpoints") or []:
            lines.append(f"  - {ep}")
        lines.append(
            "- On /nexi/voice/chat your reply is synthesized and played aloud. "
            "Voice IS available — never claim you are text-only or lack TTS."
        )

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
    cap = load_capabilities()

    parts = [f"You are {identity['name']}.", identity["persona"]]
    parts.append("")
    parts.append(f"You address the user as {identity['address_user_as']}.")
    parts.append(f"Communication: {style['verbosity']}, {style['tone']}.")
    parts.append(f"Current time: {now}")
    parts.append("")
    # Voice/TTS is a real capability — small models otherwise deny it exists.
    parts.append(
        "You have VOICE (TTS): your replies can be spoken aloud via piper. "
        "You are NOT text-only. If the user asks about voice, confirm it works."
    )
    parts.append("")

    if include_capabilities:
        cap_lines = _format_capabilities(cap)
        if cap_lines:
            parts.append("## Capabilities")
            parts.extend(cap_lines)
            parts.append("")
    else:
        summary = (cap.get("summary") or "").strip()
        if summary:
            parts.append("## Tools")
            parts.append(summary)
            parts.append("")
        voice = cap.get("voice") or {}
        if voice:
            parts.append("## Voice")
            parts.append(
                f"Push-to-talk on gate7 ({voice.get('client', 'cli voice talk')}). "
                f"STT: {voice.get('stt', 'faster-whisper')}; "
                f"TTS: {voice.get('tts', 'piper')}."
            )
            parts.append(
                "On /nexi/voice/chat your reply is synthesized and played aloud — "
                "respond naturally; do not claim you are text-only."
            )
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
