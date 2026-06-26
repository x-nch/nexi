from enum import StrEnum
from typing import Annotated
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class IntentClass(StrEnum):
    QUERY = "QUERY"
    DECISION = "DECISION"
    EXECUTION = "EXECUTION"
    ESCALATION = "ESCALATION"


class ActionType(StrEnum):
    READ_FILE = "READ_FILE"
    WRITE_FILE = "WRITE_FILE"
    DELETE_FILE = "DELETE_FILE"
    LIST = "LIST"
    RUN_COMMAND = "RUN_COMMAND"
    RUN_SCRIPT = "RUN_SCRIPT"
    DEPLOY = "DEPLOY"
    ROLLBACK = "ROLLBACK"
    STAGE = "STAGE"
    MUTATE = "MUTATE"
    BACKUP = "BACKUP"
    RESTORE = "RESTORE"
    PLAN = "PLAN"
    ANALYZE = "ANALYZE"
    ESCALATE = "ESCALATE"
    QUERY = "QUERY"


class Urgency(StrEnum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Intent(BaseModel):
    intent_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    intent_class: IntentClass
    action_type: ActionType
    target_entity_id: str
    target_entity_class: str
    constraints_declared: list[str] = Field(default_factory=list)
    urgency: Urgency = Urgency.NORMAL
    ambiguity_score: Annotated[float, Field(ge=0.0, le=1.0)]
    raw_input_hash: str
    raw_input: str = ""
    clarifications_needed: list[str] = Field(default_factory=list)
