from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nexi.character import prompt_loader
from nexi.character.prompt_loader import (
    PromptSegments,
    build_prompt_segments,
    build_system_prompt,
    get_identity_fact_records,
    get_identity_fact_texts,
    get_nexi_system_prompt,
    load_capabilities,
    load_persona,
)


@pytest.fixture(autouse=True)
def _clear_stable_cache():
    from nexi.character.prompt_loader import _STABLE_CACHE
    _STABLE_CACHE.clear()
    yield
    _STABLE_CACHE.clear()


def test_load_persona():
    persona = load_persona()
    assert "identity" in persona
    assert persona["identity"]["name"] == "Nexi"
    assert persona["identity"]["address_user_as"] == "ck-san"
    assert "persona" in persona["identity"]
    assert "communication_style" in persona
    assert "capabilities" not in persona
    assert "identity_facts" not in persona


def test_load_capabilities():
    caps = load_capabilities()
    assert "hosts" in caps
    assert "tools" in caps
    assert "tool_routing" in caps
    assert "node-a" in caps["hosts"]
    assert "xnch_fs_read" in str(caps["tools"])
    assert "identity" not in caps


def test_load_capabilities_merges_generated_overlay(monkeypatch, tmp_path):
    overlay = tmp_path / "generated.yaml"
    overlay.write_text(
        """
generated_at: "2026-08-14T00:00:00Z"
hosts:
  node-a:
    role: control-plane
    services: {xnch: "192.168.50.1:8001"}
tools:
  filesystem: ["xnch_fs_read"]
tool_routing: "File contents on disk? → xnch_fs_*"
filesystem: {read_only: true, roots: ["/home/x-nch"]}
status: {healthy: ["xnch"], down: [], checked_at: "t"}
"""
    )
    monkeypatch.setattr("nexi.character.prompt_loader.settings.capabilities_generated_path", str(overlay))

    caps = load_capabilities()
    # generated keys win
    assert caps["hosts"]["node-a"]["role"] == "control-plane"
    assert caps["tools"]["filesystem"] == ["xnch_fs_read"]
    assert caps["tool_routing"] == "File contents on disk? → xnch_fs_*"
    assert caps["status"]["healthy"] == ["xnch"]
    # base-only keys are preserved
    assert "voice" in caps
    assert "summary" in caps


def test_load_capabilities_empty_generated_section_keeps_base(monkeypatch, tmp_path):
    overlay = tmp_path / "generated.yaml"
    overlay.write_text(
        """
generated_at: "2026-08-14T00:00:00Z"
hosts:
  node-a:
    role: control-plane
    services: {xnch: "192.168.50.1:8001"}
tools: {}
tool_routing: ""
"""
    )
    monkeypatch.setattr("nexi.character.prompt_loader.settings.capabilities_generated_path", str(overlay))

    caps = load_capabilities()
    # empty generated tools/tool_routing must not clobber the richer base
    assert "xnch_fs_read" in str(caps["tools"])
    assert "crg_*" in caps["tool_routing"]
    # non-empty generated hosts still applies
    assert caps["hosts"]["node-a"]["role"] == "control-plane"


def test_load_capabilities_missing_overlay_uses_base(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "nexi.character.prompt_loader.settings.capabilities_generated_path",
        str(tmp_path / "does-not-exist.yaml"),
    )
    caps = load_capabilities()
    assert "hosts" in caps
    assert "node-a" in caps["hosts"]
    assert "voice" in caps


def test_load_capabilities_bad_overlay_falls_back(monkeypatch, tmp_path):
    overlay = tmp_path / "generated.yaml"
    overlay.write_text("::: not valid yaml")
    monkeypatch.setattr("nexi.character.prompt_loader.settings.capabilities_generated_path", str(overlay))
    caps = load_capabilities()
    assert "hosts" in caps
    assert "node-a" in caps["hosts"]


def test_identity_facts_from_yaml():
    records = get_identity_fact_records()
    assert len(records) >= 10
    texts = {r["raw_text"] for r in records}
    assert any("gate7" in t or "node-a" in t for t in texts)
    assert any("xnch_fs" in t or "filesystem" in t.lower() for t in texts)


def test_build_system_prompt_cold_start():
    prompt = build_system_prompt()
    assert "Nexi" in prompt
    assert "ck-san" in prompt
    assert "UTC" in prompt
    assert "## Capabilities" not in prompt
    assert "## Tools" in prompt
    assert "GET /nexi/capabilities" in prompt
    assert "## Voice" in prompt
    assert "TTS" in prompt
    assert "## Rules (never do)" in prompt
    assert "## Identity" in prompt
    assert "xnch_fs_read" in prompt
    assert "## Session Context" not in prompt


def test_build_system_prompt_with_capabilities():
    prompt = build_system_prompt(include_capabilities=True)
    assert "## Capabilities" in prompt
    assert "xnch_fs_read" in prompt
    assert "node-a" in prompt
    assert "Tool routing:" in prompt
    assert "crg_*" in prompt


def test_build_system_prompt_with_memory():
    session_memory = [
        {"summary": "deployed new policy filter"},
        {"summary": "fixed Kuzu query bug"},
    ]
    recent_entities = ["Gemma 4", "RTX 3090"]
    prompt = build_system_prompt(
        session_memory=session_memory,
        recent_entities=recent_entities,
    )
    assert "## Session Context" in prompt
    assert "deployed new policy filter" in prompt
    assert "## Known Entities" in prompt
    assert "Gemma 4" in prompt
    assert "RTX 3090" in prompt


def test_get_nexi_system_prompt():
    prompt = get_nexi_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 50
    assert "Nexi" in prompt
    assert "## Capabilities" in prompt


def test_build_system_prompt_includes_style():
    prompt = build_system_prompt()
    assert "concise" in prompt
    assert "direct_technical" in prompt


def test_build_system_prompt_never_do():
    prompt = build_system_prompt()
    assert "xnch_fs_read" in prompt
    assert "invent file contents" in prompt


def test_get_identity_fact_texts():
    texts = get_identity_fact_texts()
    assert isinstance(texts, list)
    assert all(isinstance(t, str) for t in texts)
    assert len(texts) >= 10


def test_character_yamls_valid():
    base = Path(prompt_loader.__file__).parent
    persona = yaml.safe_load((base / "persona.yaml").read_text())
    assert persona["identity"]["name"] == "Nexi"
    assert any("invent file contents" in item for item in persona["communication_style"]["never_do"])

    caps = yaml.safe_load((base / "capabilities.yaml").read_text())
    assert "hosts" in caps and "tools" in caps

    facts = yaml.safe_load((base / "identity_facts.yaml").read_text())
    assert len(facts["identity_facts"]) >= 10
    assert all(f["importance"] <= 2.0 for f in facts["identity_facts"])


def test_prompt_segments_splits_stable_and_dynamic():
    segs = build_prompt_segments(
        session_memory=[{"summary": "deployed foo"}],
        recent_entities=["Gemma 4"],
    )
    assert isinstance(segs, PromptSegments)
    # Stable: identity, persona, capabilities(tools path), rules
    assert "Nexi" in segs.stable
    assert "## Rules (never do)" in segs.stable
    assert "## Tools" in segs.stable
    assert "## Identity" in segs.stable
    # Dynamic: session context and entities ONLY in dynamic
    assert "## Session Context" in segs.dynamic
    assert "deployed foo" in segs.dynamic
    assert "## Known Entities" in segs.dynamic
    assert "Gemma 4" in segs.dynamic
    # Stable must NOT contain session-specific content
    assert "deployed foo" not in segs.stable
    assert "Gemma 4" not in segs.stable


def test_prompt_segments_stable_cache_hit_no_rerender(monkeypatch):
    renders = {"n": 0}

    def _fake_render():
        renders["n"] += 1
        return ["Nexi", "## Capabilities", "## Rules (never do)"]

    monkeypatch.setattr(
        "nexi.character.prompt_loader._render_stable_core",
        lambda *a, **k: _fake_render(),
    )

    build_prompt_segments(session_memory=[], recent_entities=[])
    build_prompt_segments(session_memory=[], recent_entities=[])
    # Same stable config -> second call served from cache, no re-render
    assert renders["n"] == 1


def test_prompt_segments_different_stable_invalidates_cache(monkeypatch):
    calls: list[list] = []

    def _fake_render(include_capabilities=True, identity_facts=None):
        calls.append(identity_facts)
        return ["Nexi", "## Capabilities"]

    monkeypatch.setattr(
        "nexi.character.prompt_loader._render_stable_core",
        _fake_render,
    )

    build_prompt_segments(session_memory=[], recent_entities=[])
    build_prompt_segments(session_memory=[], recent_entities=[], identity_facts=["different"])
    assert len(calls) == 2


def test_prompt_segments_stable_frozen_across_calls():
    s1 = build_prompt_segments(session_memory=[{"summary": "a"}], recent_entities=["E1"]).stable
    s2 = build_prompt_segments(session_memory=[{"summary": "b"}], recent_entities=["E2"]).stable
    # Stable prefix is byte-identical even though session context differs
    assert s1 == s2


def test_prompt_segments_concat_equals_build_system_prompt(monkeypatch):
    monkeypatch.setattr("nexi.character.prompt_loader.load_capabilities", lambda: {
        "summary": "cap summary",
        "hosts": {"node-a": {"role": "control-plane"}},
        "voice": {"stt": "faster-whisper", "tts": "piper"},
        "tools": {},
        "tool_routing": "",
    })
    monkeypatch.setattr("nexi.character.prompt_loader.get_identity_fact_texts", lambda: ["fake fact"])
    segs = build_prompt_segments(
        session_memory=[{"summary": "s"}],
        recent_entities=["E1"],
        include_capabilities=True,
    )
    full = build_system_prompt(
        session_memory=[{"summary": "s"}],
        recent_entities=["E1"],
        identity_facts=["fake fact"],
        include_capabilities=True,
    )
    # The concatenation reproduces the full system prompt modulo the frozen timestamp.
    assert segs.stable + segs.dynamic == full


def test_prompt_segments_cache_cleared():
    from nexi.character.prompt_loader import _STABLE_CACHE
    build_prompt_segments()
    assert len(_STABLE_CACHE) >= 1
