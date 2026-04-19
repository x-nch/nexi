"""Step 10a — Action spec validation before verdict submission."""
from typing import Any

from ..models.options import ActionSpec, PlanOption


_KNOWN_ACTION_TYPES = {
    "READ_FILE", "WRITE_FILE", "DELETE_FILE", "LIST",
    "RUN_COMMAND", "RUN_SCRIPT", "DEPLOY", "ROLLBACK",
    "STAGE", "MUTATE", "BACKUP", "RESTORE", "PLAN",
    "ANALYZE", "ESCALATE", "QUERY",
}


class PlanCompilationError(Exception):
    pass


def compile_action_spec(opt: PlanOption) -> dict[str, Any]:
    """Validates the selected option's action_spec. Raises PlanCompilationError on failure."""
    spec = opt.action_spec

    if not spec.type:
        raise PlanCompilationError("action_spec.type is required")
    if not spec.target:
        raise PlanCompilationError("action_spec.target is required")
    if spec.params is None:
        raise PlanCompilationError("action_spec.params must not be null")

    if spec.type.upper() not in _KNOWN_ACTION_TYPES:
        # Unknown types are preserved (per Contract 4) but logged
        pass

    return {
        "type": spec.type.upper(),
        "target": spec.target,
        "params": spec.params,
    }
