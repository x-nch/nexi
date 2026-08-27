"""Tests for the dynamic persona builder (live self-model generation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nexi.character import persona_builder as pb
from nexi.character.persona_builder import (
    PersonaGenerationError,
    SelfModel,
    introspect_backend,
    introspect_opencode,
    introspect_repo,
    introspect_skills,
    render_persona_template,
)
from nexi.config import settings


def _fake_inventory(*names: str):
    from nexi.character.capability_builder import ToolInventory

    inv = ToolInventory()
    inv.native = list(names)
    return inv


def test_introspect_opencode_union_of_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(pb, "_REPO_ROOT", tmp_path)
    # JSONC with comments + trailing commas + // inside URL
    (tmp_path / "opencode.jsonc").write_text(
        """{
  // comment
  "mcpServers": {
    "code-review-graph": {"command": "uvx"},
    "xnch": {"command": "python",},
  },
}"""
    )
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"superpowers": {}, "code-review-graph": {}}})
    )
    (tmp_path / ".opencode.json").write_text(
        json.dumps({"mcpServers": {"agentmemory": {}}})
    )
    servers = introspect_opencode(tmp_path)
    assert set(servers) == {"code-review-graph", "xnch", "superpowers", "agentmemory"}


def test_introspect_skills_reads_frontmatter(tmp_path, monkeypatch):
    monkeypatch.setattr(pb, "_REPO_ROOT", tmp_path)
    claude = tmp_path / ".claude" / "skills"
    claude.mkdir(parents=True)
    (claude / "debug-issue.md").write_text(
        "---\nname: debug-issue\ndescription: Repro bugs\n---\nbody\n"
    )
    sup = tmp_path / "skills" / "superpowers" / "skills" / "brainstorming"
    sup.mkdir(parents=True)
    (sup / "SKILL.md").write_text(
        "---\nname: brainstorming\ndescription: Turn ideas into designs\n---\nbody\n"
    )
    skills = introspect_skills(tmp_path)
    names = {s["name"] for s in skills}
    assert names == {"debug-issue", "brainstorming"}
    assert any(s["description"] == "Repro bugs" for s in skills)


def test_introspect_repo_lists_packages(tmp_path, monkeypatch):
    monkeypatch.setattr(pb, "_REPO_ROOT", tmp_path)
    (tmp_path / "xnch").mkdir(); (tmp_path / "xnch" / "__init__.py").write_text("")
    (tmp_path / "nexi").mkdir(); (tmp_path / "nexi" / "pyproject.toml").write_text("")
    (tmp_path / ".git").mkdir()
    (tmp_path / "README.md").write_text("")
    packages = introspect_repo(tmp_path)
    assert "xnch" in packages and "nexi" in packages
    assert ".git" not in packages
    assert "README.md" not in packages


async def test_introspect_backend_no_network(monkeypatch):
    monkeypatch.setattr(settings, "opencode_go_api_url", "")
    monkeypatch.setattr(settings, "vllm_primary_url", "")
    monkeypatch.setattr(settings, "vllm_health_url", "")
    monkeypatch.setattr(settings, "model_id", "deepseek-v4-pro")
    backend, model, health = await introspect_backend()
    assert backend == "deepseek-v4-pro"
    assert model == "deepseek-v4-pro"
    assert health == {}


async def test_introspect_backend_prefers_opencode_go(monkeypatch):
    monkeypatch.setattr(settings, "opencode_go_api_url", "https://opencode.ai/zen/go/v1")
    monkeypatch.setattr(settings, "vllm_primary_url", "")
    monkeypatch.setattr(settings, "model_id", "deepseek-v4-pro")
    monkeypatch.setattr(settings, "vllm_health_url", "")
    backend, model, _ = await introspect_backend()
    assert backend == "opencode-go (hosted deepseek-v4-pro)"


def test_render_persona_template_fills_placeholders():
    sm = SelfModel(
        inference_backend="opencode-go (hosted deepseek-v4-pro)",
        active_model="deepseek-v4-pro",
        hosts={"node-a": "gate7 (192.168.50.1)"},
        tool_count=37,
        mcp_servers=["code-review-graph", "xnch"],
        skills=[{"name": "debug-issue"}, {"name": "brainstorming"}],
    )
    sm.backend_health = {}
    sm.services = []
    rendered = render_persona_template(pb._FALLBACK_TEMPLATE, sm)
    assert "opencode-go (hosted deepseek-v4-pro)" in rendered
    assert "deepseek-v4-pro" in rendered
    assert "37 live tools" in rendered
    assert "code-review-graph, xnch" in rendered
    assert "debug-issue, brainstorming" in rendered
    assert "{" not in rendered  # no unresolved placeholders


def test_render_persona_template_unknown_placeholder_raises():
    sm = SelfModel()
    with pytest.raises(PersonaGenerationError):
        render_persona_template("backend={bogus_placeholder}", sm)


def test_build_self_model_assembles(monkeypatch):
    from nexi.infra.discovery import InfraSnapshot

    snap = InfraSnapshot(hosts={"node-a": {"label": "gate7", "ip": "192.168.50.1"}},
                         services=[], policies={},
                         status={"healthy": ["xnch"], "down": ["vllm-ornith"]})
    sm = pb.build_self_model(
        snapshot=snap,
        tool_count=12,
        mcp_servers=["xnch", "code-review-graph"],
        skills=[{"name": "debug-issue"}],
        backend=("opencode-go (hosted deepseek-v4-pro)", "deepseek-v4-pro", {"opencode-go": True}),
        repo_packages=["xnch", "nexi"],
    )
    assert sm.inference_backend == "opencode-go (hosted deepseek-v4-pro)"
    assert sm.hosts["node-a"] == "gate7 (192.168.50.1)"
    assert sm.tool_count == 12
    # live status reflects down service + backend health
    status = pb._build_live_status(sm)
    assert "vllm-ornith DOWN" in status
    assert "opencode-go up" in status


async def test_build_persona_writes_overlay(tmp_path, monkeypatch):
    monkeypatch.setattr(pb, "_REPO_ROOT", tmp_path)
    out = tmp_path / "persona.generated.yaml"
    monkeypatch.setattr(settings, "persona_generated_path", str(out))
    monkeypatch.setattr(settings, "opencode_go_api_url", "")
    monkeypatch.setattr(settings, "vllm_primary_url", "")
    monkeypatch.setattr(settings, "model_id", "deepseek-v4-pro")
    monkeypatch.setattr(settings, "vllm_health_url", "")

    # Stub live sources so the test never touches the network.
    class _Snap:
        hosts = {"node-a": {"label": "gate7", "ip": "192.168.50.1"}}
        services = []
        policies = {}
        status = {"healthy": ["xnch"], "down": []}
    monkeypatch.setattr(pb, "build_snapshot", lambda *a, **k: _Snap())
    async def _fake_probe(snap, *a, **k):
        return {"healthy": ["xnch"], "down": []}
    monkeypatch.setattr(pb, "probe_services", _fake_probe)
    async def _fake_inv(*a, **k):
        return _fake_inventory("xnch_fs_read", "crg_query")
    monkeypatch.setattr("nexi.character.capability_builder.fetch_tool_inventory", _fake_inv)

    # skills/opencode config absent in tmp -> empty but safe
    result = await pb.build_persona()
    assert result.error is None, result.error
    assert out.is_file()
    assert result.self_model.tool_count == 2
    # strip header then parse yaml
    body = "\n".join(out.read_text().splitlines()[1:])
    import yaml
    loaded = yaml.safe_load(body)
    assert "self_model" in loaded
    assert loaded["self_model"]["active_model"] == "deepseek-v4-pro"


async def test_build_persona_no_write_returns_model(tmp_path, monkeypatch):
    monkeypatch.setattr(pb, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(settings, "opencode_go_api_url", "")
    monkeypatch.setattr(settings, "model_id", "deepseek-v4-pro")
    monkeypatch.setattr(settings, "vllm_primary_url", "")
    monkeypatch.setattr(settings, "vllm_health_url", "")

    class _Snap:
        hosts = {"node-a": {"label": "gate7", "ip": "1.2.3.4"}}
        services = []
        policies = {}
        status = {}
    monkeypatch.setattr(pb, "build_snapshot", lambda *a, **k: _Snap())
    async def _fake_probe(snap, *a, **k):
        return {"healthy": [], "down": []}
    monkeypatch.setattr(pb, "probe_services", _fake_probe)
    async def _fake_inv(*a, **k):
        return _fake_inventory("xnch_fs_read")
    monkeypatch.setattr("nexi.character.capability_builder.fetch_tool_inventory", _fake_inv)
    result = await pb.build_persona(write=False)
    assert result.error is None
    assert result.self_model.tool_count == 1
    assert result.changed is False
