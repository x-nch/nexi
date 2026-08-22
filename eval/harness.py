"""Eval harness runner."""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .cases import EvalCase, load_cases
from .grader import GradeResult, grade_output, llm_judge


GenerateFn = Callable[[EvalCase], Awaitable[tuple[str, list[str]]]]


class EvalRunResult(BaseModel):
    run_id: str
    results: list[GradeResult] = Field(default_factory=list)
    mean_score: float = 0.0
    pass_rate: float = 0.0


class EvalHarness:
    def __init__(
        self,
        cases: list[EvalCase] | None = None,
        *,
        cases_path: Path | None = None,
        use_llm_judge: bool = False,
    ) -> None:
        self.cases = cases if cases is not None else load_cases(cases_path)
        self.use_llm_judge = use_llm_judge

    async def run(self, generate: GenerateFn) -> EvalRunResult:
        run_id = str(uuid4())
        graded: list[GradeResult] = []
        for case in self.cases:
            output, tools = await generate(case)
            result = grade_output(case, output, tools_invoked=tools)
            if self.use_llm_judge and case.judge_rubric:
                ok, notes = await llm_judge(case, output)
                result.judge_notes = notes
                result.checks["llm_judge"] = ok
                if not ok:
                    result.failures.append("llm_judge failed")
                    result.passed = False
                # Blend judge into score (equal weight with prior mean)
                n = len(result.checks)
                result.score = sum(1 for v in result.checks.values() if v) / n
            graded.append(result)

        mean = sum(r.score for r in graded) / len(graded) if graded else 0.0
        pass_rate = sum(1 for r in graded if r.passed) / len(graded) if graded else 0.0
        return EvalRunResult(
            run_id=run_id,
            results=graded,
            mean_score=mean,
            pass_rate=pass_rate,
        )
