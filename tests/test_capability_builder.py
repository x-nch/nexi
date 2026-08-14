"""Tests for the auto-generated capabilities overlay builder."""

from __future__ import annotations

import re

import pytest
import yaml

from nexi.character.capability_builder import (
    BuildResult,
    ToolInventory,
    _build_tool_routing,
    _group_tools,
    build_capabilities,
    fetch_tool_inventory,
    render_overlay,
    write_overlay,
)
from nexi.infra.discovery import build_snapshot


def _inventory() -> ToolInventory:
    return ToolInventory(
        native=["xnch_fs_read", "xnch_fs_list", "xnch_health", "xnch_exec_run", "xnch_web_search"],
        bridged={
            "code-review-graph": {
                "prefix": "crg_",
                "server": "code-review-graph",
                "connected": True,
                "tools": ["crg_query_graph_tool", "crg_detect_changes_tool"],
            },
            "agentmemory": {
                "prefix": "am_",
                "server": "agentmemory",
                "connected": True,
                "tools": ["am_memory_recall", "am_memory_save"],
            },
        },
    )


def _snapshot(tmp_path) -> "object":
    (tmp_path / "node-a" / "systemd").mkdir(parents=True)
    (tmp_path / "node-b" / "systemd").mkdir(parents=True)
    (tmp_path / "node-a" / "systemd" / "xnch.service").write_text("--port 8001\n")
    (tmp_path / "node-b" / "systemd" / "nexi.service").write_text("--port 8000\n")
    (tmp_path / "exec-policy.yaml").write_text("defaults: {timeout_seconds: 60}\nhosts: {}\n")
    (tmp_path / "fs-policy.yaml").write_text("hosts: {}\n")
    return build_snapshot(
        tmp_path,
        exec_path=str(tmp_path / "exec-policy.yaml"),
        fs_path=str(tmp_path / "fs-policy.yaml"),
    )


def test_group_tools():
    grouped = _group_tools(_inventory())
    assert "filesystem" in grouped
    assert "code_graph" in grouped
    assert "crg_query_graph_tool" in grouped["code_graph"]
    assert "am_memory_recall" in grouped["agent_memory"]
    assert "web_search" in grouped


def test_build_tool_routing():
    routing = _build_tool_routing(_group_tools(_inventory()))
    assert "crg_*" in routing
    assert "xnch_fs_*" in routing
    assert "am_memory_*" in routing
    assert "xnch_web_search" in routing


def test_build_capabilities_structure(tmp_path):
    snapshot = _snapshot(tmp_path)
    snapshot.status = {"healthy": ["xnch"], "down": ["nexi"], "checked_at": "t"}
    caps = build_capabilities(snapshot, _inventory())

    assert "hosts" in caps
    assert "node-a" in caps["hosts"]
    assert "node-b" in caps["hosts"]
    assert caps["hosts"]["node-a"]["services"]["xnch"].endswith(":8001")
    # tools are base-shaped: group -> list of entries, base descriptions preserved
    assert any(str(i).startswith("xnch_fs_read —") for i in caps["tools"]["filesystem"])
    assert any(str(i).startswith("xnch_fs_list —") for i in caps["tools"]["filesystem"])
    # discovered bridged tools are appended
    assert any(str(i).startswith("crg_query_graph_tool") for i in caps["tools"]["code_graph"])
    assert caps["bridge"]["servers"]["code-review-graph"]["connected"] is True
    assert caps["status"]["down"] == ["nexi"]
    assert caps["filesystem"]["read_only"] is True
    assert "generated_at" in caps


def test_render_and_atomic_write(tmp_path):
    caps = {"generated_at": "t", "hosts": {"node-a": {"role": "control-plane"}}}
    target = tmp_path / "overlay.yaml"
    first = write_overlay(target, render_overlay(caps))
    second = write_overlay(target, render_overlay(caps))
    assert first is True
    assert second is False
    parsed = yaml.safe_load(target.read_text())
    assert parsed["hosts"]["node-a"]["role"] == "control-plane"


async def test_fetch_tool_inventory_from_xnch(monkeypatch):
    payload = {
        "tools": [
            {"type": "function", "function": {"name": "xnch_fs_read", "description": "read"}},
            {"type": "function", "function": {"name": "crg_query_graph_tool", "description": "q"}},
        ],
        "bridge": {
            "active": True,
            "servers": [
                {"server_id": "code-review-graph", "tool_prefix": "crg_", "connected": True},
            ],
        },
    }

    class _FakeClient:
        async def get(self, url, **kwargs):
            return type("R", (), {"json": lambda self: payload, "raise_for_status": lambda self: None})()

    inventory = await fetch_tool_inventory(
        xnch_base_url="http://xnch.test:8001", http_client=_FakeClient()
    )
    assert "xnch_fs_read" in inventory.native
    assert "code-review-graph" in inventory.bridged
    assert inventory.bridged["code-review-graph"]["connected"] is True


async def test_fetch_tool_inventory_fallback_to_local_config(monkeypatch, tmp_path):
    cfg = tmp_path / "mcp.yaml"
    cfg.write_text(
        """
servers:
  docs-test:
    enabled: true
    tool_prefix: doc_
    allow_tools:
      - resolve-library-id
      - query-docs
  context7:
    enabled: false
    tool_prefix: c7_
"""
    )
    monkeypatch.setattr("nexi.character.capability_builder.settings.mcp_servers_path", str(cfg))

    class _Broken:
        async def get(self, url, **kwargs):
            raise RuntimeError("xnch down")

    inventory = await fetch_tool_inventory(
        xnch_base_url="http://xnch.test:8001", http_client=_Broken()
    )
    assert "docs-test" in inventory.bridged
    assert inventory.bridged["docs-test"]["tools"] == ["query-docs", "resolve-library-id"]
    assert "context7" not in inventory.bridged


async def test_refresh_writes_overlay(tmp_path, monkeypatch):
    from nexi.character import capability_builder as cb

    monkeypatch.setattr(cb.settings, "capabilities_generated_path", str(tmp_path / "out" / "gen.yaml"))
    monkeypatch.setattr(cb.settings, "infra_manifests_path", str(tmp_path / "infra"))

    (tmp_path / "infra" / "node-a" / "systemd").mkdir(parents=True)
    (tmp_path / "infra" / "node-b" / "systemd").mkdir(parents=True)
    (tmp_path / "infra" / "node-a" / "systemd" / "xnch.service").write_text("--port 8001\n")
    (tmp_path / "infra" / "node-b" / "systemd" / "nexi.service").write_text("--port 8000\n")

    class _FakeHttp:
        async def get(self, url, **kwargs):
            if "tools" in url:
                return type("R", (), {
                    "json": lambda self: {"tools": [], "bridge": {"servers": []}},
                    "raise_for_status": lambda self: None,
                })()
            if url.endswith("/"):
                return type("R", (), {"status_code": 200})()
            return type("R", (), {"status_code": 404})()

    result: BuildResult = await cb.refresh(
        manifest_dir=str(tmp_path / "infra"),
        xnch_base_url="http://xnch.test:8001",
        http_client=_FakeHttp(),
    )
    assert result.changed is True
    out = tmp_path / "out" / "gen.yaml"
    assert out.is_file()
    text = out.read_text()
    assert text.startswith("# AUTO-GENERATED")
    assert "node-a" in text


def test_render_overlay_is_valid_yaml():
    text = render_overlay({"status": {"healthy": ["a"]}, "generated_at": "t"})
    assert re.match(r"^# AUTO-GENERATED", text)
    body = text.split("\n", 1)[1]
    parsed = yaml.safe_load(body)
    assert parsed["status"]["healthy"] == ["a"]
