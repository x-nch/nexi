"""xnch-executor FastAPI application - sandboxed action execution."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Per-run workspace root
WORKSPACE_ROOT = Path("/var/lib/xnch/workspaces")
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


class ActionSpec(BaseModel):
    type: str
    target: str
    params: dict[str, Any] = Field(default_factory=dict)


class SimulationOverride(BaseModel):
    outcome: str | None = None


class ExecuteRequest(BaseModel):
    execution_ref: str
    decision_id: str
    execution_token: str = ""
    action_spec: ActionSpec
    simulation: SimulationOverride = Field(default_factory=SimulationOverride)
    goal_id: str = ""
    workspace_id: str | None = None


class ExecuteResponse(BaseModel):
    execution_ref: str
    decision_id: str
    outcome_status: str
    observed_state_delta: dict[str, Any] = Field(default_factory=dict)
    side_effects_observed: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    anomalies: list[str] = Field(default_factory=list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("xnch-executor starting (workspace_root=%s)", WORKSPACE_ROOT)
    yield
    logger.info("xnch-executor shutting down")


app = FastAPI(title="xnch-executor", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "xnch-executor"}


@app.post("/execute", response_model=ExecuteResponse)
async def execute(req: ExecuteRequest):
    """Execute a single action spec in sandboxed workspace."""
    import time
    start = time.perf_counter()

    # Determine workspace
    ws_id = req.workspace_id or req.decision_id[:8]
    workspace = WORKSPACE_ROOT / ws_id
    workspace.mkdir(parents=True, exist_ok=True)

    action = req.action_spec
    action_type = action.type.upper()
    target = action.target
    params = action.params

    logger.info("execute: type=%s target=%s ws=%s", action_type, target, ws_id)

    # Simulation override
    if req.simulation.outcome:
        return ExecuteResponse(
            execution_ref=req.execution_ref,
            decision_id=req.decision_id,
            outcome_status=req.simulation.outcome.upper(),
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

    try:
        result = await _execute_action(action_type, target, params, workspace)
        status = result.get("outcome", "SUCCESS")
        delta = result.get("delta", {})
        effects = result.get("effects", [])
        anomalies = result.get("anomalies", [])
    except Exception as exc:
        logger.exception("execution failed: %s", exc)
        status = "FAILURE"
        delta = {}
        effects = []
        anomalies = [f"{type(exc).__name__}: {exc}"]

    return ExecuteResponse(
        execution_ref=req.execution_ref,
        decision_id=req.decision_id,
        outcome_status=status,
        observed_state_delta=delta,
        side_effects_observed=effects,
        duration_ms=int((time.perf_counter() - start) * 1000),
        anomalies=anomalies,
    )


async def _execute_action(action_type: str, target: str, params: dict, workspace: Path) -> dict:
    """Dispatch to action-specific handler."""
    handlers = {
        "WRITE_FILE": _exec_write_file,
        "READ_FILE": _exec_read_file,
        "LIST": _exec_list,
        "DELETE_FILE": _exec_delete_file,
        "RUN_COMMAND": _exec_run_command,
        "WEB_SEARCH": _exec_web_search,
        "AGENT_DISPATCH": _exec_agent_dispatch,
    }
    handler = handlers.get(action_type)
    if not handler:
        raise ValueError(f"unknown action type: {action_type}")
    return await handler(target, params, workspace)


# --- Action Handlers ---

async def _exec_write_file(target: str, params: dict, workspace: Path) -> dict:
    """Write file to workspace."""
    content = params.get("content", "")
    mode = params.get("mode", "w")
    path = workspace / target
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"outcome": "SUCCESS", "delta": {"path": str(path)}, "effects": [f"wrote {path}"]}


async def _exec_read_file(target: str, params: dict, workspace: Path) -> dict:
    """Read file from workspace."""
    path = workspace / target
    if not path.exists():
        return {"outcome": "FAILURE", "anomalies": [f"not found: {path}"]}
    content = path.read_text(encoding="utf-8")
    return {"outcome": "SUCCESS", "delta": {"content": content}, "effects": [f"read {path}"]}


async def _exec_list(target: str, params: dict, workspace: Path) -> dict:
    """List files in workspace."""
    path = workspace / target if target else workspace
    items = [{"name": p.name, "type": "dir" if p.is_dir() else "file", "size": p.stat().st_size} for p in path.iterdir()]
    return {"outcome": "SUCCESS", "delta": {"items": items}, "effects": [f"listed {path}"]}


async def _exec_delete_file(target: str, params: dict, workspace: Path) -> dict:
    """Delete file from workspace."""
    path = workspace / target
    if path.exists():
        path.unlink()
        return {"outcome": "SUCCESS", "effects": [f"deleted {path}"]}
    return {"outcome": "FAILURE", "anomalies": [f"not found: {path}"]}


async def _exec_run_command(target: str, params: dict, workspace: Path) -> dict:
    """Run shell command in workspace (restricted)."""
    import subprocess
    cmd = params.get("command") or target
    timeout = params.get("timeout", 30)
    # Security: only allow commands in workspace, no shell operators
    if any(op in cmd for op in [";", "&", "|", "`", "$(", ">", "<"]):
        return {"outcome": "FAILURE", "anomalies": ["command contains disallowed operators"]}
    try:
        result = subprocess.run(
            cmd.split(),
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "outcome": "SUCCESS" if result.returncode == 0 else "FAILURE",
            "delta": {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode},
            "effects": [f"ran: {cmd}"],
        }
    except subprocess.TimeoutExpired:
        return {"outcome": "FAILURE", "anomalies": [f"timeout after {timeout}s"]}
    except Exception as exc:
        return {"outcome": "FAILURE", "anomalies": [str(exc)]}


async def _exec_web_search(target: str, params: dict, workspace: Path) -> dict:
    """Proxy to xnch web_search service (via xnch gateway)."""
    import httpx
    query = params.get("query") or target
    # TODO: call actual web_search service via xnch gateway
    # For now, return mock
    return {"outcome": "SUCCESS", "delta": {"query": query, "results": []}, "effects": [f"web_search: {query}"]}


async def _exec_agent_dispatch(target: str, params: dict, workspace: Path) -> dict:
    """Execute agent dispatch using local opencode in workspace."""
    prompt = params.get("prompt") or target
    # TODO: invoke opencode CLI with prompt in workspace
    # For now, write prompt as file and return success
    prompt_file = workspace / "agent_prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    return {"outcome": "SUCCESS", "delta": {"prompt_file": str(prompt_file)}, "effects": [f"agent_dispatch prepared: {prompt_file}"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)