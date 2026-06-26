from typing import Any

from pydantic import BaseModel


class DAGNode(BaseModel):
    node_id: str
    action_type: str
    target: str
    params: dict[str, Any]
    depends_on: list[str]


class CompiledDAG(BaseModel):
    nodes: list[DAGNode]
    edges: list[tuple[str, str]]
    entry_node: str
