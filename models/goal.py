import time
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class GoalStatus(StrEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Goal(BaseModel):
    goal_id: UUID = Field(default_factory=uuid4)
    owner_actor_id: str
    objective: str
    status: GoalStatus = GoalStatus.PENDING
    progress: str = ""
    steps_completed: int = 0
    max_steps: int = 10
    consecutive_failures: int = 0
    failure_threshold: int = 3
    last_step_outcome: str | None = None
    next_due_at: float | None = None
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    simulation_plan: list[dict[str, Any]] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
