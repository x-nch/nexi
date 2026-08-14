"""Tests for infra discovery: service topology, policies, and health probes."""

from __future__ import annotations

from types import SimpleNamespace

from nexi.infra.discovery import (
    ServiceSpec,
    build_snapshot,
    discover_policies,
    discover_services,
    probe_services,
)


def _make_manifest_tree(tmp_path):
    node_a = tmp_path / "node-a"
    node_b = tmp_path / "node-b"
    (node_a / "systemd").mkdir(parents=True)
    (node_b / "systemd").mkdir(parents=True)

    (node_a / "systemd" / "xnch.service").write_text(
        "ExecStart=/usr/bin/uvicorn xnch.main:app --host 0.0.0.0 --port 8001\n"
    )
    (node_a / "systemd" / "vault-indexer.service").write_text(
        "ExecStart=/usr/bin/python -m scripts.vault_indexer\n"
    )
    (node_b / "systemd" / "nexi.service").write_text(
        "ExecStart=/usr/bin/uvicorn nexi.main:app --port 8000\n"
    )
    (node_b / "systemd" / "vllm-ornith.service").write_text(
        "ExecStart=/usr/bin/vllm serve --port 8082\n"
    )
    return tmp_path


def test_discover_services_from_manifests(tmp_path):
    manifest_dir = _make_manifest_tree(tmp_path)
    services = discover_services(manifest_dir)

    by_name = {s.name: s for s in services}
    assert by_name["xnch"].port == 8001
    assert by_name["xnch"].host == "node-a"
    assert by_name["xnch"].source == "systemd"
    assert by_name["nexi"].port == 8000
    assert by_name["nexi"].host == "node-b"
    assert by_name["vllm-ornith"].port == 8082
    # unit without a port flag is not added to the topology
    assert "vault-indexer" not in by_name


def test_discover_services_compose(tmp_path):
    node_a = tmp_path / "node-a"
    node_a.mkdir(parents=True)
    (node_a / "docker-compose.yml").write_text(
        """
services:
  litellm:
    image: litellm/litellm
    ports:
      - "4000:4000"
  redis:
    image: redis
    ports:
      - "6379:6379"
"""
    )
    services = discover_services(tmp_path)
    by_name = {s.name: s for s in services}
    assert by_name["litellm"].port == 4000
    assert by_name["litellm"].source == "compose"
    assert by_name["redis"].port == 6379
    assert by_name["redis"].source == "compose"


def test_discover_services_defaults_when_no_manifests(tmp_path):
    services = discover_services(tmp_path)
    by_name = {s.name: s for s in services}
    assert by_name["xnch"].port == 8001
    assert by_name["xnch"].source == "default"
    assert by_name["nexi"].port == 8000
    assert by_name["media-gateway"].port == 8090


def test_discover_policies(tmp_path):
    (tmp_path / "exec-policy.yaml").write_text(
        """
defaults:
  timeout_seconds: 60
denied_substrings: ["sudo ", "kubectl apply"]
hosts:
  node-a:
    allowed_prefixes:
      - systemctl status
      - journalctl
"""
    )
    (tmp_path / "fs-policy.yaml").write_text(
        """
hosts:
  node-a:
    roots: ["/home/x-nch"]
  node-b:
    roots: ["/home/x-nch"]
deny_globs: ["**/.env"]
"""
    )
    policies = discover_policies(
        exec_path=str(tmp_path / "exec-policy.yaml"),
        fs_path=str(tmp_path / "fs-policy.yaml"),
    )
    assert "sudo " in policies["exec"]["denied_substrings"]
    assert policies["exec"]["hosts"]["node-a"] == ["journalctl", "systemctl status"]
    assert policies["filesystem"]["read_only"] is True
    assert "/home/x-nch" in policies["filesystem"]["roots"]
    assert "**/.env" in policies["filesystem"]["deny_globs"]


def test_build_snapshot_assembles_topology_and_policies(tmp_path):
    manifest_dir = _make_manifest_tree(tmp_path)
    (tmp_path / "exec-policy.yaml").write_text("defaults: {timeout_seconds: 30}\nhosts: {}\n")
    (tmp_path / "fs-policy.yaml").write_text("hosts: {}\n")

    snapshot = build_snapshot(
        manifest_dir,
        exec_path=str(tmp_path / "exec-policy.yaml"),
        fs_path=str(tmp_path / "fs-policy.yaml"),
    )
    assert "node-a" in snapshot.hosts
    assert "node-b" in snapshot.hosts
    assert any(s.name == "nexi" for s in snapshot.services_on("node-b"))
    assert snapshot.policies["exec"]["timeout_seconds"] == 30


class _FakeHttp:
    def __init__(self, responses: dict[str, int]):
        self.responses = responses

    async def get(self, url, headers=None):
        code = self.responses.get(url, 200)
        if code == -1:
            raise RuntimeError("connection refused")
        return SimpleNamespace(status_code=code)


async def test_probe_services_classifies_up_and_down(tmp_path):
    manifest_dir = _make_manifest_tree(tmp_path)
    snapshot = build_snapshot(manifest_dir)

    # node-a xnch up; node-b nexi/vllm unreachable
    fake = _FakeHttp({"http://192.168.50.1:8001/": 200, "http://192.168.50.1:4000/": -1})
    status = await probe_services(snapshot, http_client=fake)
    assert "xnch" in status["healthy"]
    assert "litellm" in status["down"]
    assert "checked_at" in status


def test_service_priority_prefers_inference():
    from nexi.infra.discovery import service_priority

    assert service_priority("vllm-ornith") > service_priority("media-gateway")
    assert service_priority("nexi") >= 3
