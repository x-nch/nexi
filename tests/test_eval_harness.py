"""Tests for Nexi loop-2 eval harness (deterministic graders)."""
from __future__ import annotations

import pytest

from nexi.eval import EvalHarness, grade_output, load_cases
from nexi.eval.cases import EvalCase


def test_load_frozen_cases() -> None:
    cases = load_cases()
    assert len(cases) >= 5
    assert cases[0].id == "greeting-direct"


def test_grade_must_contain_pass() -> None:
    case = EvalCase(id="t", prompt="x", must_contain=["Nexi"])
    result = grade_output(case, "I am Nexi on gate7.")
    assert result.passed
    assert result.score == 1.0


def test_grade_forbidden_and_tool() -> None:
    case = EvalCase(
        id="t",
        prompt="x",
        must_not_contain=["Great question"],
        required_tools=["xnch_fs_read"],
    )
    bad = grade_output(case, "Great question! here is a guess.", tools_invoked=[])
    assert not bad.passed
    assert any("forbidden" in f for f in bad.failures)
    assert any("required tool" in f for f in bad.failures)

    good = grade_output(case, "Reading via tool.", tools_invoked=["xnch_fs_read"])
    assert good.passed


@pytest.mark.asyncio
async def test_harness_mean_score() -> None:
    cases = [
        EvalCase(id="a", prompt="p", must_contain=["ok"]),
        EvalCase(id="b", prompt="p", must_contain=["ok"]),
    ]
    harness = EvalHarness(cases=cases)

    async def generate(case: EvalCase) -> tuple[str, list[str]]:
        if case.id == "a":
            return "ok done", []
        return "fail", []

    result = await harness.run(generate)
    assert result.mean_score == 0.5
    assert result.pass_rate == 0.5
