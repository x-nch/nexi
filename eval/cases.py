"""Eval case models and loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    prompt: str
    intent_class: str = "QUERY"
    # Deterministic expectations
    must_contain: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    max_chars: int | None = None
    # Optional LLM-judge rubric (used only when judge enabled)
    judge_rubric: str | None = None


def load_cases(path: Path | None = None) -> list[EvalCase]:
    cases_path = path or Path(__file__).with_name("cases.yaml")
    raw = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
    items: list[Any] = raw.get("cases", raw) if isinstance(raw, dict) else raw
    return [EvalCase.model_validate(item) for item in items]
