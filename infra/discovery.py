"""Infra discovery — service topology, policies, and live health probes.

Reads only — never writes infra state. Sources of truth:
- ``infra/no-k3s/node-{a,b}/systemd/*.service``  → services on each host
- ``infra/no-k3s/node-{a,b}/docker-compose.yml`` → container services
- ``~/.xnch/mcp-servers.yaml``                  → MCP bridge server inventory
- ``~/.xnch/exec-policy.yaml`` + ``fs-policy.yaml`` → governed capability summary
- live HTTP probes                             → realtime status
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml

from nexi.config import settings

logger = logging.getLogger(__name__)

# Baseline topology (node hostnames/roles). Ports are derived from manifests when
# possible; these defaults keep the snapshot sane when a manifest is missing.
DEFAULT_HOSTS: dict[str, dict[str, Any]] = {
    "node-a": {"ip": "192.168.50.1", "role": "control-plane", "label": "gate7"},
    "node-b": {"ip": "192.168.50.2", "role": "inference", "label": "xnch-core"},
}

# Baseline service -> (host, port). Overridden per-service when a systemd unit
# carries a `--port` flag or a compose service maps a host port.
DEFAULT_SERVICES: dict[str, tuple[str, int]] = {
    "xnch": ("node-a", 8001),
    "litellm": ("node-a", 4000),
    "postgres": ("node-a", 5432),
    "redis": ("node-a", 6379),
    "searxng": ("node-a", 8888),
    "nexi": ("node-b", 8000),
    "vllm-ornith": ("node-b", 8082),
    "fs-read-agent": ("node-b", 8003),
    "exec-agent": ("node-b", 8004),
}

_PORT_RE = re.compile(r"(?:--?port[=\s]+(\d{2,5}))|(?::(\d{2,5})(?:/|\s|$))")
_INFERENCE_PRIORITY = {"vllm-ornith": 5, "nexi": 3}


@dataclass
class ServiceSpec:
    name: str
    host: str
    port: int
    source: str = "default"  # "default" | "systemd" | "compose"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class InfraSnapshot:
    hosts: dict[str, dict[str, Any]]
    services: list[ServiceSpec]
    policies: dict[str, Any]
    status: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: _now_iso())

    def service_names(self) -> list[str]:
        return sorted({s.name for s in self.services})

    def services_on(self, host: str) -> list[ServiceSpec]:
        return sorted((s for s in self.services if s.host == host), key=lambda s: s.name)

    def down_services(self) -> list[str]:
        return list(self.status.get("down", []))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_yaml_optional(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.is_file():
        return {}
    try:
        data = yaml.safe_load(p.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("Failed to load %s: %s", p, exc)
        return {}


def _manifest_dir(path: str | Path | None = None) -> Path:
    return Path(path or settings.infra_manifests_path).expanduser()


def _service_name_from_unit(filename: str) -> str:
    """'vllm-ornith.service' → 'vllm-ornith'; 'nexi.service' → 'nexi'."""
    return Path(filename).stem


def _ports_in_text(text: str) -> list[int]:
    return [int(a or b) for a, b in _PORT_RE.findall(text)]


def _discover_systemd_services(manifest_dir: Path) -> dict[str, ServiceSpec]:
    """Parse systemd unit files for service/port pairs per host."""
    found: dict[str, ServiceSpec] = {}
    for host in DEFAULT_HOSTS:
        units_dir = manifest_dir / f"node-{host.split('-')[1]}" / "systemd"
        if not units_dir.is_dir():
            units_dir = manifest_dir / f"node-{host}" / "systemd"
        if not units_dir.is_dir():
            continue
        for unit in sorted(units_dir.glob("*.service")):
            name = _service_name_from_unit(unit.name)
            try:
                text = unit.read_text()
            except OSError:
                continue
            ports = _ports_in_text(text)
            if not ports:
                continue
            base_host, base_port = DEFAULT_SERVICES.get(name, (host, 0))
            found[name] = ServiceSpec(
                name=name,
                host=base_host,
                port=ports[0],
                source="systemd",
            )
    return found


def _discover_compose_services(manifest_dir: Path) -> dict[str, ServiceSpec]:
    """Parse docker-compose port mappings into ServiceSpec entries."""
    found: dict[str, ServiceSpec] = {}
    for host in DEFAULT_HOSTS:
        host_dir = manifest_dir / f"node-{host.split('-')[1]}"
        compose = host_dir / "docker-compose.yml"
        if not compose.is_file():
            compose = host_dir / "docker-compose.yaml"
        if not compose.is_file():
            continue
        data = _load_yaml_optional(compose)
        services = data.get("services") or {}
        for name, spec in services.items():
            if not isinstance(spec, dict):
                continue
            ports = spec.get("ports") or []
            host_port: int | None = None
            for raw in ports:
                text = str(raw)
                m = re.match(r"^\s*(\d{2,5}):\d{2,5}", text)
                if m:
                    host_port = int(m.group(1))
                    break
            if host_port is None:
                continue
            base_host, _ = DEFAULT_SERVICES.get(name, (host, 0))
            found[name] = ServiceSpec(
                name=name, host=base_host, port=host_port, source="compose"
            )
    return found


def discover_services(manifest_dir: str | Path | None = None) -> list[ServiceSpec]:
    """Build the service topology: defaults overridden by systemd/compose manifests."""
    base = _manifest_dir(manifest_dir)
    merged: dict[str, ServiceSpec] = {}
    for name, (host, port) in DEFAULT_SERVICES.items():
        merged[name] = ServiceSpec(name=name, host=host, port=port, source="default")
    merged.update(_discover_systemd_services(base))
    merged.update(_discover_compose_services(base))
    return sorted(merged.values(), key=lambda s: s.name)


def discover_policies(
    exec_path: str | Path | None = None,
    fs_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize governed exec/fs policy for capability rendering."""
    exec_policy = _load_yaml_optional(exec_path or settings.exec_policy_path)
    fs_policy = _load_yaml_optional(fs_path or settings.fs_policy_path)

    exec_blocks = exec_policy.get("hosts") or {}
    exec_summary = {
        host: sorted(prefixes.get("allowed_prefixes", []))
        for host, prefixes in sorted(exec_blocks.items())
        if isinstance(prefixes, dict)
    }
    denied = exec_policy.get("denied_substrings") or []
    fs_roots = fs_policy.get("hosts") or {}
    deny_globs = fs_policy.get("deny_globs") or []

    return {
        "exec": {
            "hosts": exec_summary,
            "denied_substrings": list(denied),
            "timeout_seconds": exec_policy.get("defaults", {}).get("timeout_seconds", 60),
        },
        "filesystem": {
            "read_only": True,
            "roots": sorted(
                {root for host_cfg in fs_roots.values() if isinstance(host_cfg, dict)
                 for root in host_cfg.get("roots", [])}
            ),
            "deny_globs": list(deny_globs),
        },
    }


def build_snapshot(
    manifest_dir: str | Path | None = None,
    exec_path: str | Path | None = None,
    fs_path: str | Path | None = None,
) -> InfraSnapshot:
    """Topology + policy snapshot (no network)."""
    hosts = {host: dict(cfg) for host, cfg in DEFAULT_HOSTS.items()}
    services = discover_services(manifest_dir)
    for svc in services:
        hosts.setdefault(svc.host, dict(DEFAULT_HOSTS.get(svc.host, {"role": "service"})))
        hosts[svc.host].setdefault("ip", DEFAULT_HOSTS.get(svc.host, {}).get("ip", ""))
    policies = discover_policies(exec_path, fs_path)
    return InfraSnapshot(hosts=hosts, services=services, policies=policies)


def service_priority(name: str) -> int:
    return _INFERENCE_PRIORITY.get(name, 1)


async def probe_services(
    snapshot: InfraSnapshot,
    timeout: float | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Probe every discovered service; any HTTP response <500 counts as up."""
    timeout = timeout or settings.probe_timeout_s
    healthy: set[str] = set()
    down: set[str] = set()

    async def _probe(svc: ServiceSpec) -> None:
        host_cfg = snapshot.hosts.get(svc.host, {})
        ip = host_cfg.get("ip") or svc.host
        url = f"http://{ip}:{svc.port}/"
        try:
            client = http_client
            if client is None:
                async with httpx.AsyncClient(timeout=timeout) as c:
                    resp = await c.get(url, headers={"User-Agent": "nexi-capability-probe"})
            else:
                resp = await client.get(url, headers={"User-Agent": "nexi-capability-probe"})
            if resp.status_code < 500:
                healthy.add(svc.name)
            else:
                down.add(svc.name)
        except Exception:
            down.add(svc.name)

    for svc in snapshot.services:
        await _probe(svc)

    return {
        "healthy": sorted(healthy),
        "down": sorted(down),
        "checked_at": _now_iso(),
    }
