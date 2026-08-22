"""Persist eval harness scores into episodic PG (loop-4 signal path)."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xnch.memory.pg_episodic_store import PgEpisodicStore

    from .harness import EvalRunResult


async def persist_eval_run(
    store: PgEpisodicStore,
    result: EvalRunResult,
    *,
    model_id: str = "qwen2.5-vl-7b",
) -> str:
    """Write one eval_runs row (+ summary episode). Returns run_id."""
    await store.store_eval_run(
        run_id=result.run_id,
        model_id=model_id,
        mean_score=result.mean_score,
        pass_rate=result.pass_rate,
        results_json=[r.model_dump(mode="json") for r in result.results],
    )
    summary = (
        f"eval_run {result.run_id}: mean_score={result.mean_score:.3f} "
        f"pass_rate={result.pass_rate:.3f} n={len(result.results)}"
    )
    await store.store_episode(
        type_="eval_run",
        raw_text=json.dumps({"run_id": result.run_id, "model_id": model_id}),
        summary=summary,
        importance=0.8,
    )
    return result.run_id
