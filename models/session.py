from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel


class ActorRole(StrEnum):
    ADMIN = "ADMIN"
    OPERATOR = "OPERATOR"
    VIEWER = "VIEWER"
    AGENT = "AGENT"


class Actor(BaseModel):
    id: str
    role: ActorRole
    capability_set: list[str]


class SessionContext(BaseModel):
    session_id: UUID
    trace_id: UUID
    actor: Actor
    system_state_version: str
    policy_version: str
    idempotency_key: UUID
    raw_input: str
    priority: str = "NORMAL"
