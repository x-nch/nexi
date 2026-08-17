"""Goal step planner — pure synthesis of step input + simulation override."""

from nexi.goal.planner import build_simulation, build_step_input


def _make_goal(**overrides: object) -> dict:
    base: dict[str, object] = {
        "objective": "deploy media service",
        "progress": "",
        "steps_completed": 0,
        "simulation_plan": [],
    }
    base.update(overrides)
    return base


async def test_build_step_input_includes_objective():
    goal = _make_goal(objective="ship the api")
    assert build_step_input(goal) == "ship the api"


async def test_build_step_input_folds_progress():
    goal = _make_goal(objective="ship the api", progress="step 1 done")
    assert build_step_input(goal) == "ship the api\n[progress]\nstep 1 done"


async def test_build_step_input_omits_progress_when_empty():
    goal = _make_goal(objective="ship the api", progress="")
    assert build_step_input(goal) == "ship the api"


async def test_build_step_input_omits_progress_when_whitespace():
    goal = _make_goal(objective="ship the api", progress="   ")
    assert build_step_input(goal) == "ship the api"


async def test_build_step_input_omits_progress_when_none():
    goal = _make_goal(objective="ship the api", progress=None)
    assert build_step_input(goal) == "ship the api"


async def test_build_simulation_returns_plan_entry_at_step_index():
    plan = [{"kind": "sim", "name": "s0"}, {"kind": "sim", "name": "s1"}]
    goal = _make_goal(simulation_plan=plan, steps_completed=1)
    assert build_simulation(goal) == {"kind": "sim", "name": "s1"}


async def test_build_simulation_returns_none_when_plan_empty():
    goal = _make_goal(simulation_plan=[], steps_completed=0)
    assert build_simulation(goal) is None


async def test_build_simulation_returns_none_when_plan_absent():
    goal = _make_goal()
    goal.pop("simulation_plan")
    assert build_simulation(goal) is None


async def test_build_simulation_returns_none_when_index_past_end():
    goal = _make_goal(simulation_plan=[{"kind": "sim"}], steps_completed=1)
    assert build_simulation(goal) is None
