"""Nexi output eval harness — loop-2 verification (not pre-selection scoring).

Frozen cases + deterministic graders (+ optional LLM-judge). Scores can be
persisted into episodic PG via PgEpisodicStore.store_eval_run.
"""
from __future__ import annotations

from .grader import GradeResult, grade_output
from .cases import EvalCase, load_cases
from .harness import EvalHarness, EvalRunResult
from .store import persist_eval_run

__all__ = [
    "EvalCase",
    "EvalHarness",
    "EvalRunResult",
    "GradeResult",
    "grade_output",
    "load_cases",
    "persist_eval_run",
]
