"""Goal step planner — pure synthesis of step input + simulation override."""


def build_step_input(goal: dict) -> str:
    base = goal.get("objective", "")
    progress = (goal.get("progress") or "").strip()
    parts = [base]
    if progress:
        parts.append(f"[progress]\n{progress}")
    return "\n".join(parts)


def build_simulation(goal: dict) -> dict | None:
    plan = goal.get("simulation_plan") or []
    idx = goal.get("steps_completed", 0)
    return plan[idx] if idx < len(plan) else None
