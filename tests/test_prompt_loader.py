from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from nexi.character import prompt_loader
from nexi.character.prompt_loader import (
    build_system_prompt,
    get_identity_fact_records,
    get_identity_fact_texts,
    get_nexi_system_prompt,
    load_character,
)


def test_load_character():
    char = load_character()
    assert "identity" in char
    assert char["identity"]["name"] == "Nexi"
    assert char["identity"]["address_user_as"] == "ck-san"
    assert "persona" in char["identity"]
    assert "communication_style" in char
    assert "identity_facts" in char
    assert "capabilities" in char


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
    assert "## Capabilities" in prompt
    assert "## Rules (never do)" in prompt
    assert "## Identity" in prompt
    assert "xnch_fs_read" in prompt
    assert "## Session Context" not in prompt


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


def test_load_character_yaml_valid():
    raw = (Path(prompt_loader.__file__).parent / "nexi_character.yaml").read_text()
    parsed = yaml.safe_load(raw)
    assert parsed["identity"]["name"] == "Nexi"
    assert any("invent file contents" in item for item in parsed["communication_style"]["never_do"])
