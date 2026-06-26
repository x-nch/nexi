"""Step 10a — Action spec validation before verdict submission.

Converts a PlanOption into a single-node CompiledDAG for execution dispatch.
"""
from ..models.dag import DAGNode, CompiledDAG
from ..models.options import ActionSpec, PlanOption


_KNOWN_ACTION_TYPES = {
    "READ_FILE", "WRITE_FILE", "DELETE_FILE", "LIST",
    "RUN_COMMAND", "RUN_SCRIPT", "DEPLOY", "ROLLBACK",
    "STAGE", "MUTATE", "BACKUP", "RESTORE", "PLAN",
    "ANALYZE", "ESCALATE", "QUERY",
}


class PlanCompilationError(Exception):
    pass


def compile_action_spec(opt: PlanOption) -> CompiledDAG:
    """Validates the selected option and returns a single-node CompiledDAG."""
    spec = opt.action_spec

    if not spec.type:
        raise PlanCompilationError("action_spec.type is required")
    if not spec.target:
        raise PlanCompilationError("action_spec.target is required")
    if spec.params is None:
        raise PlanCompilationError("action_spec.params must not be null")

    if spec.type.upper() not in _KNOWN_ACTION_TYPES:
        pass

    node = DAGNode(
        node_id=str(opt.option_id),
        action_type=spec.type.upper(),
        target=spec.target,
        params=spec.params,
        depends_on=[],
    )
    return CompiledDAG(
        nodes=[node],
        edges=[],
        entry_node=node.node_id,
    )
