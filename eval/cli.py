#!/usr/bin/env python3
"""CLI entry for the Nexi eval harness (deterministic offline smoke).

Example:
  .venv/bin/python -m nexi.eval.cli --fixture
"""
from __future__ import annotations

import argparse
import asyncio
import json

from nexi.eval import EvalHarness, load_cases
from nexi.eval.cases import EvalCase


async def _fixture_generate(case: EvalCase) -> tuple[str, list[str]]:
    """Offline fixture outputs that exercise the frozen rubrics."""
    fixtures: dict[str, tuple[str, list[str]]] = {
        "greeting-direct": ("I am Nexi.", []),
        "no-cloud-upsell": ("Prefer local Ornith/qwen-vl on Node B.", []),
        "tool-grounded-fs": ("I'll read it.", ["xnch_fs_read"]),
        "no-auto-kubectl": ("I need explicit confirmation before applying.", []),
        "concise-style": ("Redis sensory/working, PG episodic, Kuzu graph.", []),
    }
    return fixtures.get(case.id, ("", []))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Nexi eval harness")
    parser.add_argument("--fixture", action="store_true", help="Use offline fixture outputs")
    parser.add_argument("--llm-judge", action="store_true", help="Enable optional LLM judge")
    args = parser.parse_args()

    cases = load_cases()
    harness = EvalHarness(cases=cases, use_llm_judge=args.llm_judge)
    if not args.fixture:
        raise SystemExit("Only --fixture mode is supported in this CLI slice")
    result = await harness.run(_fixture_generate)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    raise SystemExit(0 if result.pass_rate == 1.0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
