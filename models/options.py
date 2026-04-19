from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PolicyVerdict(StrEnum):
    ALLOW = "ALLOW"
    ALLOW_WITH_WARNINGS = "ALLOW_WITH_WARNINGS"
    MODIFY = "MODIFY"
    DEFER = "DEFER"
    BLOCK = "BLOCK"


class GenerationPath(StrEnum):
    MODEL = "MODEL"
    RULE_BASED = "RULE_BASED"


class ActionSpec(BaseModel):
    type: str
    target: str
    params: dict[str, Any] = Field(default_factory=dict)


class PlanOption(BaseModel):
    option_id: UUID = Field(default_factory=uuid4)
    action_type: str
    action_spec: ActionSpec
    stated_rationale: str
    estimated_side_effects: list[str] = Field(default_factory=list)
    reversible: bool
    payload_hash: str


class PolicyDryRunResponse(BaseModel):
    option_id: UUID
    session_id: UUID
    verdict: PolicyVerdict
    policy_refs: list[str]
    warnings: list[str] = Field(default_factory=list)
    modified_action_spec: ActionSpec | None = None
    requires_actor: str | None = None


class Scores(BaseModel):
    policy_score: Annotated[float, Field(ge=0.0, le=1.0)]
    outcome_score: Annotated[float, Field(ge=0.0, le=1.0)]
    risk_score: Annotated[float, Field(ge=0.0, le=1.0)]
    context_fit_score: Annotated[float, Field(ge=0.0, le=1.0)]


class EvaluatedOption(BaseModel):
    option_id: UUID
    policy_verdict: PolicyVerdict
    scores: Scores
    composite_score: Annotated[float, Field(ge=0.0, le=1.0)]
    weight_config_version: str
    simulation_required: bool


class SelectionRationale(BaseModel):
    score_breakdown: dict[str, Any]
    weight_config_version: str


class DecisionRecord(BaseModel):
    decision_id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    intent_ref: UUID
    context_manifest_ref: UUID
    system_state_version: str
    options_generated: int
    options_blocked: int
    options_evaluated: list[EvaluatedOption]
    selected_option_id: UUID | None
    selection_rationale: SelectionRationale
    confidence: float
    escalation_triggered: bool = False
    generation_path: GenerationPath = GenerationPath.MODEL
