"""ExecutionDispatchPayload simulation + goal_id field tests."""
from uuid import uuid4

from nexi.models.outcomes import ExecutionDispatchPayload


async def test_dispatch_payload_carries_simulation_and_goal_id():
    gid = uuid4()
    p = ExecutionDispatchPayload(
        trace_id=uuid4(), decision_id=uuid4(),
        action_spec={"type": "DEPLOY", "target": "x", "params": {}},
        execution_token="tok", token_ttl_ms=30000,
        simulation={"outcome": "fail", "detail": "x", "next_plan_hint": "y"},
        goal_id=gid,
    )
    assert p.simulation == {"outcome": "fail", "detail": "x", "next_plan_hint": "y"}
    assert p.goal_id == gid


async def test_dispatch_payload_defaults():
    p = ExecutionDispatchPayload(
        trace_id=uuid4(), decision_id=uuid4(),
        action_spec={}, execution_token="t", token_ttl_ms=1,
    )
    assert p.simulation is None and p.goal_id is None
