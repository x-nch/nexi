"""Goal model defaults + JSON serialization tests."""
from uuid import UUID

from nexi.models import Goal, GoalStatus


def _make_goal(**overrides: object) -> Goal:
    kwargs: dict[str, object] = {
        "owner_actor_id": "agent",
        "objective": "deploy media service",
    }
    kwargs.update(overrides)
    return Goal(**kwargs)


async def test_goal_defaults():
    goal = _make_goal()
    assert goal.max_steps == 10
    assert goal.failure_threshold == 3
    assert goal.status is GoalStatus.PENDING
    assert goal.steps_completed == 0
    assert goal.consecutive_failures == 0
    assert goal.progress == ""
    assert goal.simulation_plan == []
    assert goal.last_step_outcome is None
    assert goal.next_due_at is None
    assert goal.lease_owner is None
    assert goal.lease_expires_at is None


async def test_goal_goal_id_is_uuid_by_default():
    goal = _make_goal()
    assert isinstance(goal.goal_id, UUID)


async def test_goal_json_serialization():
    goal = _make_goal()
    dumped = goal.model_dump(mode="json")
    assert isinstance(dumped["status"], str)
    assert dumped["status"] == "PENDING"
    assert isinstance(dumped["goal_id"], str)
    UUID(dumped["goal_id"])
