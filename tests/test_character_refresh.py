"""Tests for the standalone persona + capabilities refresh entry point.

``python -m nexi.character.refresh`` is used to (re)generate the dynamic persona
and capabilities overlays in environments where nexi runs on-demand (no persistent
FastAPI service) so the lifespan-driven refresh loop never fires.
"""

from __future__ import annotations

import asyncio

import pytest

from nexi.character import refresh as refresh_mod
from nexi.character import persona_builder
from nexi.character.capability_builder import BuildResult


@pytest.fixture(autouse=True)
def _isolation(monkeypatch, tmp_path):
    """Point both overlay outputs and infra at temp dirs so no real files change."""
    out = tmp_path / "out"
    persona_path = str(out / "persona.generated.yaml")
    caps_path = str(out / "caps.generated.yaml")
    # Patch the paths where each builder's settings resolve their outputs.
    monkeypatch.setattr(
        persona_builder.settings, "persona_generated_path", persona_path
    )
    from nexi.character import capability_builder as cb

    monkeypatch.setattr(cb.settings, "capabilities_generated_path", caps_path)
    monkeypatch.setattr(cb.settings, "infra_manifests_path", str(tmp_path / "infra"))
    return out


def test_arun_returns_zero_and_writes_both_overlays(_isolation):
    code = asyncio.run(refresh_mod.arun())
    assert code == 0
    persona = _isolation / "persona.generated.yaml"
    caps = _isolation / "caps.generated.yaml"
    assert persona.is_file()
    assert caps.is_file()
    assert persona.read_text().startswith("# AUTO-GENERATED")
    assert caps.read_text().startswith("# AUTO-GENERATED")


def test_arun_reports_error_via_exit_code(monkeypatch, _isolation):
    async def _boom():
        return BuildResult(capabilities={}, error="boom")

    monkeypatch.setattr(
        "nexi.character.refresh.refresh_capabilities", lambda: _boom(),
    )
    code = asyncio.run(refresh_mod.arun())
    assert code == 1
