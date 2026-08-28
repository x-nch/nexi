from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from nexi.config import settings
from nexi.character.persona_builder import SelfModel, render_persona_template

logger = logging.getLogger(__name__)

_PERSONA_PATH = Path(__file__).parent / "persona.yaml"
_CAPABILITIES_PATH = Path(__file__).parent / "capabilities.yaml"
_FACTS_PATH = Path(__file__).parent / "identity_facts.yaml"
_DEFAULT_IMPORTANCE = 2.0

# Keys the auto-refresh generator owns. When a generated overlay is present these
# override the hand-maintained base; everything else (summary, voice, …) stays
# authoritative in capabilities.yaml.
_GENERATED_KEYS = ("hosts", "tools", "tool_routing", "filesystem", "status")

# Keys the persona generator owns (the live self-model). Merged into load_persona().
_PERSONA_GENERATED_KEYS = ("self_model",)


@dataclass
class PromptSegments:
    """Stable-prefix / dynamic-suffix split, modeled on elizaOS promptSegments.

    ``stable`` holds the persona, identity, capabilities and rules — content that
    changes only when the character config changes. ``dynamic`` holds
    session-scoped context (memory, entities). Keeping the stable prefix byte
    identical across calls lets LLM providers (Anthropic cache_control, OpenAI /
    Gemini stable-prefix reordering) cache the expensive fixed preamble.
    """

    stable: str = ""
    dynamic: str = ""

    def to_messages(self, raw_input: str) -> list[dict]:
        return [
            {"role": "system", "content": self.stable + self.dynamic},
            {"role": "user", "content": raw_input},
        ]


# Cache of rendered stable cores keyed by a hash of the stable inputs. Avoids
# re-parsing/re-assembling the persona/capabilities preamble on every assembly.
_STABLE_CACHE: dict[str, str] = {}


def _config_fingerprint() -> str:
    """Hash of the persona/capabilities inputs so the stable core is rebuilt
    when any of them changes (including a regenerated overlay).

    The rendered stable prefix depends on persona.yaml + capabilities.yaml merged
    with their auto-generated overlays. The overlay files change on every persona
    refresh, so their content must be part of the cache key — otherwise a
    long-lived process keeps serving a stale persona after a refresh (bug: the
    key previously ignored the overlays entirely).
    """
    wanted = [
        _PERSONA_PATH,
        Path(__file__).parent / "capabilities.yaml",
        Path(settings.persona_generated_path).expanduser(),
        get_generated_overlay_path(),
    ]
    h = hashlib.sha256()
    for path in wanted:
        h.update(b"\x00")
        h.update(path.as_posix().encode())
        try:
            st = path.stat()
        except OSError:
            continue
        h.update(str(st.st_size).encode())
        h.update(str(st.st_mtime_ns).encode())
    return h.hexdigest()


def _stable_cache_key(
    identity_facts: list[str] | None,
    include_capabilities: bool,
) -> str:
    payload = hashlib.sha256()
    if identity_facts is None:
        payload.update(b"<auto>")
    else:
        payload.update(("\x1f".join(identity_facts)).encode())
    payload.update(("\x1e" + str(include_capabilities)).encode())
    payload.update(("\x1f" + _config_fingerprint()).encode())
    return payload.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_persona() -> dict[str, Any]:
    """Identity and communication style — merged with the live persona overlay."""
    persona = _load_yaml(_PERSONA_PATH)
    overlay = _load_generated_persona_overlay()
    if overlay:
        for key in _PERSONA_GENERATED_KEYS:
            if key in overlay and overlay[key] not in (None, ""):
                persona[key] = overlay[key]
    return persona


def _load_generated_persona_overlay() -> dict[str, Any] | None:
    """Best-effort read of the auto-generated persona overlay; falls back to repo copy."""
    candidates = [
        Path(settings.persona_generated_path).expanduser(),
        Path(__file__).parent / "persona.generated.yaml",
    ]
    for raw in candidates:
        path = Path(raw).expanduser()
        if not path.is_file():
            continue
        try:
            return _load_yaml(path)
        except Exception as exc:
            logger.warning("Failed to load generated persona overlay %s: %s", path, exc)
            return None
    return None


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


def _render_stable_core(
    include_capabilities: bool = False,
    identity_facts: list[str] | None = None,
) -> list[str]:
    """Render the stable prompt preamble (persona / capabilities / rules)."""
    persona = load_persona()
    identity = persona["identity"]
    style = persona["communication_style"]
    cap = load_capabilities()

    persona_text = identity["persona"]
    self_model = persona.get("self_model")
    if self_model:
        try:
            persona_text = render_persona_template(persona_text, SelfModel(**self_model))
        except Exception as exc:
            logger.warning("Persona template render failed; falling back to raw: %s", exc)
    parts = [f"You are {identity['name']}.", persona_text]
    parts.append("")
    parts.append(f"You address the user as {identity['address_user_as']}.")
    parts.append(f"Communication: {style['verbosity']}, {style['tone']}.")
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

    return parts


def build_prompt_segments(
    session_memory: list[dict] | None = None,
    recent_entities: list[str] | None = None,
    identity_facts: list[str] | None = None,
    include_capabilities: bool = False,
) -> PromptSegments:
    """Split a system prompt into stable (cacheable) + dynamic (session) segments.

    The stable prefix is cached keyed on its inputs; repeated assemblies with an
    unchanged character config reuse the rendered preamble instead of re-parsing
    YAML and re-assembling — the same stable-prefix idea elizaOS's promptSegments
    exposes for provider-level prompt caching.
    """
    key = _stable_cache_key(identity_facts, include_capabilities)
    stable = _STABLE_CACHE.get(key)
    if stable is None:
        stable = "\n".join(_render_stable_core(include_capabilities, identity_facts))
        _STABLE_CACHE[key] = stable

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    dyn_parts: list[str] = [f"Current time: {now}", ""]

    if session_memory:
        dyn_parts.append("## Session Context")
        for mem in session_memory[-5:]:
            summary = mem.get("summary", mem.get("raw_text", ""))
            dyn_parts.append(f"- {summary}")
        dyn_parts.append("")

    if recent_entities:
        dyn_parts.append("## Known Entities")
        for ent in recent_entities:
            dyn_parts.append(f"- {ent}")
        dyn_parts.append("")

    return PromptSegments(stable=stable, dynamic="\n".join(dyn_parts))


def build_system_prompt(
    session_memory: list[dict] | None = None,
    recent_entities: list[str] | None = None,
    identity_facts: list[str] | None = None,
    include_capabilities: bool = False,
) -> str:
    segs = build_prompt_segments(
        session_memory=session_memory,
        recent_entities=recent_entities,
        identity_facts=identity_facts,
        include_capabilities=include_capabilities,
    )
    return segs.stable + segs.dynamic


def get_nexi_system_prompt() -> str:
    return build_system_prompt(include_capabilities=True)
