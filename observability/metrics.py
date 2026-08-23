"""Prometheus instrumentation for nexi: pipeline stages, gate decisions, goal loop, callbacks."""
from __future__ import annotations

import ipaddress
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

HTTP_REQUESTS = Counter(
    "nexi_http_requests_total",
    "HTTP requests processed, by method/route-template/status.",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "nexi_http_request_seconds",
    "End-to-end request latency by route template.",
    ["method", "route"],
)

STAGE_SECONDS = Histogram(
    "nexi_pipeline_stage_seconds",
    "Duration of each decision-pipeline stage.",
    ["stage"],
)
PIPELINE_PASS = Counter(
    "nexi_pipeline_pass_total",
    "Completed pipeline passes by terminal status (EXECUTING/ESCALATED/ERROR).",
    ["status"],
)
POLICY_OPTIONS = Counter(
    "nexi_policy_options_total",
    "PolicyFilter option outcomes (pass/blocked/modified).",
    ["verdict"],
)
POLICY_ALL_BLOCKED = Counter(
    "nexi_policy_all_blocked_total",
    "Pipeline passes where every generated option was blocked by policy.",
)
GOAL_CLAIM = Counter(
    "nexi_goal_claim_total",
    "Goal-driver lease claims by result (claimed/none/error).",
    ["result"],
)
GOAL_STEP = Counter(
    "nexi_goal_step_total",
    "Goal-driver step outcomes (executing/blocked/clarification/error).",
    ["result"],
)
OUTCOME_CALLBACK = Counter(
    "nexi_outcome_callback_total",
    "Outcome callbacks processed by result (recorded/write_failed/skipped).",
    ["result"],
)

CAPABILITY_REFRESH = Gauge(  # noqa: F841 — exposed for scrape completeness
    "nexi_capability_refresh_last_success_unixtime",
    "Unix timestamp of the last successful capability refresh; 0 until first success.",
)


@asynccontextmanager
async def stage_timer(stage: str) -> AsyncIterator[None]:
    """Time an awaited pipeline stage into STAGE_SECONDS, even when it raises."""
    start = time.perf_counter()
    try:
        yield
    finally:
        STAGE_SECONDS.labels(stage=stage).observe(time.perf_counter() - start)


def record_pass_outcome(status: str) -> None:
    PIPELINE_PASS.labels(status=status).inc()


def host_allowed(host: str | None, allowlist: Iterable[str]) -> bool:
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host in set(allowlist)
    for entry in allowlist:
        try:
            if addr in ipaddress.ip_network(entry, strict=False):
                return True
        except ValueError:
            continue
    return False


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        method = request.method
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = getattr(request.scope.get("route"), "path_format", None) or "unmatched"
            HTTP_REQUESTS.labels(method=method, route=route, status=str(status)).inc()
            HTTP_LATENCY.labels(method=method, route=route).observe(time.perf_counter() - start)


def install_metrics_middleware(app: Any) -> None:
    app.add_middleware(MetricsMiddleware)


async def metrics_endpoint(request: Request) -> Response:
    from ..config import settings

    client_host = request.client.host if request.client else ""
    if not host_allowed(client_host, list(settings.metrics_allow_cidrs)):
        return Response("forbidden", status_code=403)
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
