"""Steps 7 & 8 — Scoring and conditional outcome simulation."""
from typing import Any
from uuid import UUID

from ..models import (
    SessionContext, Intent, ContextManifest,
    PlanOption, PolicyDryRunResponse, EvaluatedOption,
)
from ..models.options import PolicyVerdict, Scores
from ..models.session import ActorRole
from ..utils.context_signature import compute_context_signature
from ..utils.audit import emit_event


# Default weights per intent class (loaded from xnch in production via weight_config)
_DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "EXECUTION":  {"policy": 0.25, "outcome": 0.30, "risk": 0.35, "context_fit": 0.10},
    "QUERY":      {"policy": 0.20, "outcome": 0.30, "risk": 0.20, "context_fit": 0.30},
    "DECISION":   {"policy": 0.25, "outcome": 0.35, "risk": 0.25, "context_fit": 0.15},
    "ESCALATION": {"policy": 0.30, "outcome": 0.25, "risk": 0.30, "context_fit": 0.15},
}

_POLICY_SCORE_MAP = {
    PolicyVerdict.ALLOW: 1.0,
    PolicyVerdict.ALLOW_WITH_WARNINGS: 0.7,
    PolicyVerdict.MODIFY: 0.5,
    PolicyVerdict.DEFER: 0.3,
}

_ENTITY_SENSITIVITY: dict[str, float] = {
    "ML_MODEL": 0.6, "DATABASE": 0.8, "SCHEMA": 0.9,
    "SERVICE": 0.5, "CLUSTER": 0.7, "FILE": 0.2,
}


def _outcome_score(
    manifest: ContextManifest,
    intent_class: str,
    action_type: str,
    entity_class: str,
    actor_role: str,
) -> float:
    sig = compute_context_signature(intent_class, action_type, entity_class, actor_role)
    for pattern in manifest.patterns:
        if pattern.context_signature == sig:
            return pattern.success_rate * pattern.confidence
    # No matching pattern — neutral score
    return 0.5


def _risk_score(
    opt: PlanOption,
    actor_role: str,
    entity_class: str,
) -> float:
    score = 0.0
    if not opt.reversible:
        score += 0.3
    score += _ENTITY_SENSITIVITY.get(entity_class, 0.3)
    score += min(len(opt.estimated_side_effects) * 0.05, 0.2)
    if actor_role == ActorRole.AGENT:
        score += 0.1
    return min(score, 1.0)


def _context_fit_score(opt: PlanOption, intent: Intent) -> float:
    if not intent.constraints_declared:
        return 1.0
    spec_fields = set(opt.action_spec.params.keys())
    constraint_fields = set(intent.constraints_declared)
    if not constraint_fields:
        return 1.0
    return len(spec_fields & constraint_fields) / len(constraint_fields)


class Evaluator:
    def __init__(self, weight_config: dict[str, Any] | None = None) -> None:
        self._weight_config = weight_config
        self._weight_config_version = (
            weight_config.get("version", "default") if weight_config else "default-v0"
        )

    def score(
        self,
        options: list[tuple[PlanOption, PolicyDryRunResponse]],
        intent: Intent,
        manifest: ContextManifest,
        session: SessionContext,
    ) -> list[EvaluatedOption]:
        weights = self._resolve_weights(intent.intent_class)
        evaluated = []

        for opt, dry_run in options:
            p_score = _POLICY_SCORE_MAP.get(dry_run.verdict, 0.0)
            o_score = _outcome_score(
                manifest, intent.intent_class, opt.action_type,
                intent.target_entity_class, session.actor.role,
            )
            r_score = _risk_score(opt, session.actor.role, intent.target_entity_class)
            c_score = _context_fit_score(opt, intent)

            composite = (
                p_score * weights["policy"]
                + o_score * weights["outcome"]
                + r_score * weights["risk"]
                + c_score * weights["context_fit"]
            )

            simulation_required = (
                r_score > 0.6
                or not opt.reversible
                or session.actor.role == ActorRole.AGENT
            )

            evaluated.append(EvaluatedOption(
                option_id=opt.option_id,
                policy_verdict=dry_run.verdict,
                scores=Scores(
                    policy_score=p_score,
                    outcome_score=o_score,
                    risk_score=r_score,
                    context_fit_score=c_score,
                ),
                composite_score=round(composite, 4),
                weight_config_version=self._weight_config_version,
                simulation_required=simulation_required,
            ))

        emit_event(session.trace_id, "evaluator", "SCORING_COMPLETE",
                   {"options_scored": len(evaluated)})
        return evaluated

    def simulate_and_rescore(
        self,
        evaluated: list[EvaluatedOption],
        options: list[tuple[PlanOption, PolicyDryRunResponse]],
        manifest: ContextManifest,
        intent: Intent,
        session: SessionContext,
    ) -> list[EvaluatedOption]:
        """Step 8: forward state projection for top-2 options requiring simulation."""
        needs_sim = [e for e in evaluated if e.simulation_required]
        if not needs_sim:
            return evaluated

        # Take top 2 by composite score
        top2 = sorted(needs_sim, key=lambda e: e.composite_score, reverse=True)[:2]
        opt_map = {opt.option_id: opt for opt, _ in options}

        rescored = {e.option_id: e for e in evaluated}

        for ev in top2:
            opt = opt_map.get(ev.option_id)
            if not opt:
                continue

            violation = self._project_state(opt, manifest)
            if violation:
                new_risk = min(ev.scores.risk_score + 0.3, 1.0)
                weights = self._resolve_weights(intent.intent_class)
                new_composite = round(
                    ev.scores.policy_score * weights["policy"]
                    + ev.scores.outcome_score * weights["outcome"]
                    + new_risk * weights["risk"]
                    + ev.scores.context_fit_score * weights["context_fit"],
                    4,
                )
                updated_scores = ev.scores.model_copy(update={"risk_score": new_risk})
                rescored[ev.option_id] = ev.model_copy(
                    update={"scores": updated_scores, "composite_score": new_composite}
                )
                emit_event(session.trace_id, "evaluator", "SIMULATION_VIOLATION",
                           {"option_id": str(ev.option_id), "new_composite": new_composite})

        return list(rescored.values())

    def _project_state(self, opt: PlanOption, manifest: ContextManifest) -> bool:
        # v0 stub: no real forward projection; returns False (no violation) by default.
        # Production implementation queries system state + outcome_delta patterns.
        return False

    def _resolve_weights(self, intent_class: str) -> dict[str, float]:
        if self._weight_config and "weights" in self._weight_config:
            w = self._weight_config["weights"]
            return {
                "policy": w["policy_score"],
                "outcome": w["outcome_score"],
                "risk": w["risk_score"],
                "context_fit": w["context_fit_score"],
            }
        return _DEFAULT_WEIGHTS.get(intent_class, _DEFAULT_WEIGHTS["EXECUTION"])
