from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class OutcomeStatus(StrEnum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILURE = "FAILURE"
    ROLLED_BACK = "ROLLED_BACK"


class VerdictResponse(BaseModel):
    request_id: UUID
    verdict: str
    verdict_reason: str
    policy_refs: list[str]
    modified_action: dict[str, Any] | None = None
    execution_token: str | None = None
    token_ttl_ms: int
    audit_ref: UUID


class ExecutionDispatchPayload(BaseModel):
    execution_ref: UUID = Field(default_factory=uuid4)
    trace_id: UUID
    decision_id: UUID
    action_spec: dict[str, Any]
    execution_token: str
    token_ttl_ms: int


class ExecutionOutcome(BaseModel):
    execution_ref: UUID
    decision_id: UUID
    execution_token_ref: str
    outcome_status: OutcomeStatus
    observed_state_delta: dict[str, Any] = Field(default_factory=dict)
    side_effects_observed: list[str] = Field(default_factory=list)
    duration_ms: int
    anomalies: list[str] = Field(default_factory=list)


class Episode(BaseModel):
    episode_id: UUID = Field(default_factory=uuid4)
    decision_id: UUID
    intent_class: str
    action_type: str
    entity_class: str
    actor_role: str
    outcome: OutcomeStatus | None = None
    prediction_delta: float | None = None
    early_reextraction_flag: bool | None = None
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class EpisodeRef(BaseModel):
    episode_id: UUID
    action_type: str
    entity_class: str
    outcome: str
    created_at: datetime
    duration_ms: int | None = None


class PatternRef(BaseModel):
    pattern_id: UUID
    context_signature: str
    success_rate: float
    confidence: float
    observation_count: int


class PolicyRef(BaseModel):
    policy_id: str
    rule_expression: str
    enforcement_level: str


class ExperienceRef(BaseModel):
    experience_id: UUID
    context_signature: str
    intent_class: str
    action_type: str
    entity_class: str
    actor_role: str
    outcome: str
    lesson: str
    insight: str
    verdict: str
    confidence: float
    applicability: str
    created_at: datetime


class ContextManifest(BaseModel):
    manifest_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    system_state_version: str
    pinned_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    episodes: list[EpisodeRef] = Field(default_factory=list)
    patterns: list[PatternRef] = Field(default_factory=list)
    policies: list[PolicyRef] = Field(default_factory=list)
    experiences: list[ExperienceRef] = Field(default_factory=list)
