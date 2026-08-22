"""Deterministic (+ optional LLM) graders for Nexi outputs."""
from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field

from nexi.config import settings
from .cases import EvalCase

logger = logging.getLogger(__name__)


class GradeResult(BaseModel):
    case_id: str
    passed: bool
    score: float
    checks: dict[str, bool] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    judge_notes: str | None = None


def grade_output(
    case: EvalCase,
    output: str,
    *,
    tools_invoked: list[str] | None = None,
) -> GradeResult:
    """Deterministic rubric checks. Score is fraction of checks passed."""
    text = output or ""
    tools = tools_invoked or []
    checks: dict[str, bool] = {}
    failures: list[str] = []

    for needle in case.must_contain:
        ok = needle.lower() in text.lower()
        checks[f"must_contain:{needle}"] = ok
        if not ok:
            failures.append(f"missing required substring: {needle!r}")

    for needle in case.must_not_contain:
        ok = needle.lower() not in text.lower()
        checks[f"must_not_contain:{needle}"] = ok
        if not ok:
            failures.append(f"forbidden substring present: {needle!r}")

    for tool in case.required_tools:
        ok = tool in tools
        checks[f"required_tool:{tool}"] = ok
        if not ok:
            failures.append(f"required tool not invoked: {tool}")

    if case.max_chars is not None:
        ok = len(text) <= case.max_chars
        checks["max_chars"] = ok
        if not ok:
            failures.append(f"output length {len(text)} > max_chars {case.max_chars}")

    if not checks:
        checks["noop"] = True

    passed_n = sum(1 for v in checks.values() if v)
    score = passed_n / len(checks)
    return GradeResult(
        case_id=case.id,
        passed=len(failures) == 0,
        score=score,
        checks=checks,
        failures=failures,
    )


async def llm_judge(
    case: EvalCase,
    output: str,
    *,
    litellm_url: str | None = None,
    model: str | None = None,
) -> tuple[bool, str]:
    """Optional LLM-judge against case.judge_rubric. Returns (pass, notes)."""
    if not case.judge_rubric:
        return True, "no rubric"
    url = (litellm_url or "http://litellm:4000").rstrip("/") + "/chat/completions"
    model_id = model or settings.model_id
    try:
        from xnch.config import settings as xnch_settings

        url = (litellm_url or xnch_settings.litellm_proxy_url).rstrip("/") + "/chat/completions"
    except Exception:
        pass

    prompt = (
        f"Rubric:\n{case.judge_rubric}\n\n"
        f"User prompt:\n{case.prompt}\n\n"
        f"Model output:\n{output}\n\n"
        "Reply with exactly PASS or FAIL on the first line, then one sentence of notes."
    )
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                url,
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "You are a strict output grader."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 200,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("llm_judge failed: %s", exc)
        return True, f"judge skipped: {exc}"

    first = content.splitlines()[0].upper() if content else ""
    return first.startswith("PASS"), content
