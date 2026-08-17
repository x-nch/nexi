"""HTTP client for all xnch-server interactions."""
import asyncio
from typing import Any
from uuid import UUID

import httpx

from ..config import settings
from ..models import (
    SessionContext,
    ContextManifest,
    PolicyDryRunResponse,
    DecisionRecord,
    VerdictResponse,
    Goal,
)
from ..utils.audit import emit_event


class XnchClient:
    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.xnch_base_url,
            timeout=10.0,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Step 4: memory/read — context manifest
    # ------------------------------------------------------------------

    async def read_context(
        self,
        session: SessionContext,
        intent_class: str,
        target_entity_id: str,
        target_entity_class: str,
    ) -> ContextManifest:
        body = {
            "session_id": str(session.session_id),
            "actor_id": session.actor.id,
            "actor_role": session.actor.role,
            "query": {
                "intent_class": intent_class,
                "target_entity_id": target_entity_id,
                "target_entity_class": target_entity_class,
                "lookback_window_days": 30,
                "max_episodes": 20,
                "max_patterns": 10,
            },
        }
        resp = await self._http.post("/memory/read", json=body)
        resp.raise_for_status()
        manifest = ContextManifest.model_validate(resp.json())
        emit_event(session.trace_id, "xnch_client", "CONTEXT_MANIFEST_RECEIVED",
                   {"manifest_id": str(manifest.manifest_id)})
        return manifest

    # ------------------------------------------------------------------
    # Step 6: policy/check — dry-run per option (called in parallel)
    # ------------------------------------------------------------------

    async def check_policy(
        self,
        session: SessionContext,
        option_id: UUID,
        action_type: str,
        action_spec: dict[str, Any],
        payload_hash: str,
    ) -> PolicyDryRunResponse:
        body = {
            "session_id": str(session.session_id),
            "system_state_version": session.system_state_version,
            "actor_role": session.actor.role,
            "option_id": str(option_id),
            "action": {
                "type": action_type,
                "target": action_spec.get("target", ""),
                "spec": action_spec.get("params", {}),
                "payload_hash": payload_hash,
            },
        }
        resp = await self._http.post("/policy/check", json=body)
        resp.raise_for_status()
        return PolicyDryRunResponse.model_validate(resp.json())

    async def check_policies_parallel(
        self,
        session: SessionContext,
        options: list[Any],
    ) -> list[PolicyDryRunResponse]:
        tasks = [
            self.check_policy(
                session,
                opt.option_id,
                opt.action_type,
                opt.action_spec.model_dump(),
                opt.payload_hash,
            )
            for opt in options
        ]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Step 10: verdict submission
    # ------------------------------------------------------------------

    async def submit_verdict(
        self,
        session: SessionContext,
        decision: DecisionRecord,
        selected_action_spec: dict[str, Any],
        payload_hash: str,
        intent_class: str = "",
        entity_class: str = "",
        outcome_score_predicted: float = 0.5,
        goal_id: UUID | None = None,
    ) -> VerdictResponse:
        context: dict[str, Any] = {
            "session_id": str(session.session_id),
            "nexi_reasoning_ref": str(decision.decision_id),
            "system_state_version": session.system_state_version,
            "outcome_score_predicted": outcome_score_predicted,
        }
        if goal_id is not None:
            context["goal_id"] = str(goal_id)
        body = {
            "request_id": str(decision.decision_id),
            "actor": {
                "id": session.actor.id,
                "claimed_role": session.actor.role,
            },
            "action": {
                "type": selected_action_spec.get("type", ""),
                "target": selected_action_spec.get("target", ""),
                "payload_hash": payload_hash,
                "payload": selected_action_spec.get("params", {}),
                "intent_class": intent_class,
                "entity_class": entity_class,
            },
            "context": context,
        }
        resp = await self._http.post("/verdict", json=body)
        resp.raise_for_status()
        verdict = VerdictResponse.model_validate(resp.json())
        emit_event(session.trace_id, "xnch_client", "VERDICT_RECEIVED",
                   {"verdict": verdict.verdict, "audit_ref": str(verdict.audit_ref)})
        return verdict

    # ------------------------------------------------------------------
    # Step 14: memory/write — prediction delta update
    # ------------------------------------------------------------------

    async def write_prediction_update(
        self,
        session: SessionContext,
        episode_id: UUID,
        prediction_delta: float,
        early_reextraction_flag: bool,
    ) -> None:
        body = {
            "session_id": str(session.session_id),
            "actor_id": session.actor.id,
            "actor_role": session.actor.role.lower(),
            "write_type": "EPISODE_PREDICTION_UPDATE",
            "payload": {
                "episode_id": str(episode_id),
                "prediction_delta": prediction_delta,
                "early_reextraction_flag": early_reextraction_flag,
            },
        }
        resp = await self._http.post("/memory/write", json=body)
        resp.raise_for_status()

    # ------------------------------------------------------------------
    # Step 14: memory/write — experiential reflection (Summary output)
    # ------------------------------------------------------------------

    async def write_experience(
        self,
        context_signature: str,
        intent_class: str,
        action_type: str,
        entity_class: str,
        actor_role: str,
        outcome: str,
        lesson: str,
        insight: str,
        verdict: str,
        applicability: str,
    ) -> None:
        body = {
            "session_id": "system",
            "actor_id": "nexi",
            "actor_role": "nexi",
            "write_type": "EXPERIENCE_REFLECTION",
            "payload": {
                "context_signature": context_signature,
                "intent_class": intent_class,
                "action_type": action_type,
                "entity_class": entity_class,
                "actor_role": actor_role,
                "outcome": outcome,
                "lesson": lesson,
                "insight": insight,
                "verdict": verdict,
                "applicability": applicability,
            },
        }
        resp = await self._http.post("/memory/write", json=body)
        resp.raise_for_status()

    # ------------------------------------------------------------------
    # Governance: weight config retrieval
    # ------------------------------------------------------------------

    async def get_weight_config(self, intent_class: str) -> dict[str, Any]:
        resp = await self._http.get("/governance/weights", params={"intent_class": intent_class})
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Goal tracking: claim / update / system state
    # ------------------------------------------------------------------

    async def claim_next_goal(self, lease_owner: str) -> Goal | None:
        resp = await self._http.post("/goals/claim", json={"lease_owner": lease_owner})
        resp.raise_for_status()
        data = resp.json()
        return Goal.model_validate(data) if data else None

    async def update_goal(
        self,
        goal_id: str,
        *,
        status: str | None = None,
        progress: str | None = None,
    ) -> Goal:
        resp = await self._http.post(
            f"/goals/{goal_id}/update",
            json={"status": status, "progress": progress},
        )
        resp.raise_for_status()
        return Goal.model_validate(resp.json())

    async def get_system_state(self) -> dict[str, Any]:
        resp = await self._http.get("/system/state")
        resp.raise_for_status()
        return resp.json()
