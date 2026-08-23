"""Step 6 — Parallel policy dry-run via xnch."""
from ..adapters.xnch_client import XnchClient
from ..models import SessionContext, PlanOption, PolicyDryRunResponse
from ..models.options import PolicyVerdict
from ..observability.metrics import POLICY_ALL_BLOCKED, POLICY_OPTIONS
from ..utils.audit import emit_event


class AllOptionsBlocked(Exception):
    pass


class PolicyFilter:
    def __init__(self, xnch: XnchClient) -> None:
        self._xnch = xnch

    async def filter(
        self,
        session: SessionContext,
        options: list[PlanOption],
    ) -> list[tuple[PlanOption, PolicyDryRunResponse]]:
        emit_event(session.trace_id, "policy_filter", "DRY_RUN_START",
                   {"option_count": len(options)})

        responses = await self._xnch.check_policies_parallel(session, options)
        option_map = {opt.option_id: opt for opt in options}

        surviving: list[tuple[PlanOption, PolicyDryRunResponse]] = []
        blocked_count = 0

        for resp in responses:
            opt = option_map[resp.option_id]
            if resp.verdict == PolicyVerdict.BLOCK:
                blocked_count += 1
                POLICY_OPTIONS.labels(verdict="blocked").inc()
                emit_event(session.trace_id, "policy_filter", "OPTION_BLOCKED",
                           {"option_id": str(resp.option_id), "policy_refs": resp.policy_refs})
                continue

            if resp.verdict == PolicyVerdict.MODIFY and resp.modified_action_spec:
                # Replace action spec with xnch's modified version
                POLICY_OPTIONS.labels(verdict="modified").inc()
                opt = opt.model_copy(update={"action_spec": resp.modified_action_spec})

            POLICY_OPTIONS.labels(verdict="pass").inc()
            surviving.append((opt, resp))

        emit_event(session.trace_id, "policy_filter", "DRY_RUN_COMPLETE",
                   {"surviving": len(surviving), "blocked": blocked_count})

        if not surviving:
            POLICY_ALL_BLOCKED.inc()
            raise AllOptionsBlocked(f"All {len(options)} options blocked by policy")

        return surviving
