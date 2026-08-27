"""HTTP serve mode for the eval runner — no deployment-specific contracts.

The spend cap is a bounded-overshoot budget, not an a-priori hard cap.
Admission uses a realistic per-case estimate: a case is admitted while
``spent + in_flight_estimate + estimate(case) <= cap``. Actual spend is
reconciled after every case, admission stops the moment the cap is reached,
and the receipt prints both the actual total and the bound:

    total spend <= cap + (concurrency - 1) * max_observed_case_cost

A separately-derived per-case ceiling is used only as an anomaly alarm. If a
case's actual cost exceeds a configured multiple of the running median (or the
derived ceiling), the receipt flags it as anomalous and keeps the
``derived-ceiling invariant violated`` wording when the derived ceiling is the
breached threshold. Lower concurrency tightens the bound; concurrency 1 gives
``cap + one case``.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
import tempfile
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from daimon.adapters.cli.eval.budget import Budget
from daimon.adapters.cli.eval.live import ManagedAgentReplayBackend, ReadOnlyPostgresExecutor
from daimon.adapters.cli.eval.models import CaseResult, Golden
from daimon.adapters.cli.eval.runner import run_eval, validate_golden, write_results
from daimon.core.catalog.models import MeasureRegistry
from daimon.core.config import load_settings
from daimon.core.db import build_engine, build_session_factory
from daimon.core.defaults.loader import load_measure_registry
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

__all__ = [
    "EvalServe",
    "build_app",
    "default_run_battery",
    "healthz",
    "post_eval",
    "get_eval",
    "get_events",
]


def _verdict(status: str, case_result: CaseResult) -> str:
    if status == "aborted" or case_result.get("error_type") == "BudgetExhausted":
        return "aborted"
    return "pass" if case_result["passed"] else "fail"


def _error_class_name(error_type: str | None) -> str | None:
    """Return only the exception class name; never a full message string."""
    if error_type is None:
        return None
    if ":" in error_type:
        return error_type.split(":", 1)[0].strip() or None
    return error_type


def _initial_artifact(run_id: str, total_cases: int) -> dict[str, Any]:
    _ = total_cases
    return {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": None,
        "summary": {
            "cases": 0,
            "passed": 0,
            "failed": 0,
            "all_gating_passed": False,
        },
        "cases": [],
    }


@dataclass
class RunState:
    """Mutable in-memory state for one battery."""

    run_id: str
    total_cases: int
    status: str = "pending"
    artifact: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}), init=False)
    receipt: str = ""
    events: list[dict[str, Any]] = field(
        default_factory=lambda: cast(list[dict[str, Any]], []), init=False
    )
    completed: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    temp_dir: Path = field(init=False)
    _event_queues: set[asyncio.Queue[dict[str, Any] | None]] = field(
        default_factory=lambda: cast(set[asyncio.Queue[dict[str, Any] | None]], set()),
        init=False,
    )

    def __post_init__(self) -> None:
        self.artifact = _initial_artifact(self.run_id, self.total_cases)
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"eval-serve-{self.run_id}-"))

    def add_case(self, case_id: str, status: str, case_result: CaseResult | None) -> None:
        if status == "start" or case_result is None:
            return
        cases = cast(list[dict[str, Any]], self.artifact["cases"])
        cases.append(cast(dict[str, Any], case_result))
        summary = cast(dict[str, Any], self.artifact["summary"])
        total = len(cases)
        passed = sum(1 for c in cases if c.get("passed"))
        summary["cases"] = total
        summary["passed"] = passed
        summary["failed"] = total - passed
        summary["all_gating_passed"] = passed == total and total > 0
        event = {
            "id": case_id,
            "verdict": _verdict(status, case_result),
            "duration_s": case_result.get("duration_s", 0.0),
            "error_class": _error_class_name(case_result.get("error_type")),
        }
        self.events.append(event)
        for queue in set(self._event_queues):
            queue.put_nowait(event)

    def complete(self, result: dict[str, Any], receipt: str) -> None:
        self.artifact = result
        self.receipt = receipt
        self.status = "completed"
        self._close_queues()

    def fail(self, message: str) -> None:
        self.receipt = f"failed: {message}"
        self.status = "failed"
        self._close_queues()

    def _close_queues(self) -> None:
        self.completed.set()
        for queue in set(self._event_queues):
            queue.put_nowait(None)

    async def event_stream(self) -> AsyncGenerator[dict[str, Any], None]:
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._event_queues.add(queue)
        try:
            for event in list(self.events):
                yield event
            if self.completed.is_set():
                return
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield event
        finally:
            self._event_queues.discard(queue)


type RunBattery = Callable[[RunState, list[Golden], dict[str, Any]], Awaitable[None]]


class EvalServe:
    """In-memory eval serve orchestrator."""

    def __init__(
        self,
        token: str,
        measure_registry: MeasureRegistry | None,
        run_battery: RunBattery,
        max_concurrent: int = 1,
    ) -> None:
        self._token = token
        self._measure_registry = measure_registry
        self._run_battery = run_battery
        self._max_concurrent = max_concurrent
        self._runs: dict[str, RunState] = {}
        self._active_count = 0
        self._lock = asyncio.Lock()

    def require_auth(self, authorization: str | None) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = authorization[7:]
        if not hmac.compare_digest(token, self._token):
            raise HTTPException(status_code=401, detail="invalid bearer token")

    def _load_goldens(self, raw: Any) -> list[Golden]:
        items: list[Any] = []
        if isinstance(raw, str):
            for line_number, line in enumerate(raw.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"goldens line {line_number}: {error}") from error
        elif isinstance(raw, list):
            items: list[Any] = cast(list[Any], raw)
        else:
            raise ValueError("goldens must be a JSONL string or a list of objects")
        goldens: list[Golden] = []
        for index, payload in enumerate(items):
            if not isinstance(payload, dict):
                raise ValueError(f"goldens[{index}] is not an object")
            goldens.append(
                validate_golden(
                    cast(dict[str, object], payload),
                    location=f"goldens[{index}]",
                    measure_registry=self._measure_registry,
                )
            )
        if not goldens:
            raise ValueError("goldens is empty")
        return goldens

    async def start(self, body: dict[str, Any]) -> dict[str, Any]:
        raw_goldens = body.get("goldens")
        raw_options: Any = body.get("options") or {}
        if not isinstance(raw_options, dict):
            raise HTTPException(status_code=400, detail="options must be an object")
        options = cast(dict[str, Any], raw_options)
        try:
            goldens = self._load_goldens(raw_goldens)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        concurrency = int(options.get("concurrency", 1))
        if concurrency < 1:
            raise HTTPException(status_code=400, detail="concurrency must be >= 1")
        case_timeout = float(options.get("case_timeout", 600.0))
        if case_timeout <= 0:
            raise HTTPException(status_code=400, detail="case_timeout must be > 0")
        spend_cap = options.get("spend_cap_usd")
        if isinstance(spend_cap, (int, float)) and spend_cap <= 0:
            raise HTTPException(status_code=400, detail="spend_cap_usd must be > 0")
        allow_concurrent = bool(options.get("allow_concurrent", False))
        raw_run_id = options.get("run_id")
        run_id = raw_run_id if isinstance(raw_run_id, str) else uuid.uuid4().hex

        async with self._lock:
            if self._active_count >= 1 and not allow_concurrent:
                raise HTTPException(
                    status_code=409,
                    detail="a run is already active; set allow_concurrent=true to bypass",
                )
            state = RunState(run_id=run_id, total_cases=len(goldens))
            self._runs[run_id] = state
            self._active_count += 1

        asyncio.create_task(self._battery_wrapper(state, goldens, options))
        state.status = "pending"
        return {"run_id": run_id, "status": "pending"}

    async def _battery_wrapper(
        self, state: RunState, goldens: list[Golden], options: dict[str, Any]
    ) -> None:
        try:
            await self._run_battery(state, goldens, options)
        except Exception as error:
            state.fail(str(error))
        finally:
            async with self._lock:
                self._active_count -= 1

    def get_status(self, run_id: str) -> dict[str, Any]:
        state = self._runs.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {
            "status": state.status,
            "receipt": state.receipt,
            "artifact": state.artifact,
        }

    def event_stream(self, run_id: str) -> StreamingResponse:
        state = self._runs.get(run_id)
        if state is None:
            raise HTTPException(status_code=404, detail="run not found")

        async def content() -> AsyncGenerator[bytes, None]:
            async for event in state.event_stream():
                yield (json.dumps(event, separators=(",", ":")) + "\n").encode("utf-8")

        return StreamingResponse(
            content(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )


async def healthz() -> dict[str, str]:
    return {"status": "ok"}


async def post_eval(
    request: Request,
    body: dict[str, Any],
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict[str, Any]:
    eval_serve: EvalServe = request.app.state.eval_serve
    eval_serve.require_auth(authorization)
    return await eval_serve.start(body)


async def get_eval(
    request: Request,
    run_id: str,
    authorization: str | None = Header(None, alias="Authorization"),
) -> dict[str, Any]:
    eval_serve: EvalServe = request.app.state.eval_serve
    eval_serve.require_auth(authorization)
    return eval_serve.get_status(run_id)


async def get_events(
    request: Request,
    run_id: str,
    authorization: str | None = Header(None, alias="Authorization"),
) -> StreamingResponse:
    eval_serve: EvalServe = request.app.state.eval_serve
    eval_serve.require_auth(authorization)
    return eval_serve.event_stream(run_id)


def build_app(eval_serve: EvalServe) -> FastAPI:
    app = FastAPI(title="daimon eval serve", version="0.1.0")
    app.state.eval_serve = eval_serve

    app.add_api_route("/healthz", healthz, methods=["GET"])
    app.add_api_route("/eval", post_eval, methods=["POST"], status_code=202)
    app.add_api_route("/eval/{run_id}", get_eval, methods=["GET"])
    app.add_api_route("/eval/{run_id}/events", get_events, methods=["GET"])

    return app


async def default_run_battery(
    state: RunState,
    goldens: list[Golden],
    options: dict[str, Any],
) -> None:
    """The same runner path as ``daimon eval run``; only the transport differs."""
    from anthropic import AsyncAnthropic

    warehouse_dsn = os.environ.get("DAIMON_EVAL_SERVE__WAREHOUSE_URL") or os.environ.get(
        "GRAVITY_WAREHOUSE_URL"
    )
    if not warehouse_dsn:
        raise RuntimeError(
            "DAIMON_EVAL_SERVE__WAREHOUSE_URL is required "
            "(GRAVITY_WAREHOUSE_URL is a fork-side alias)"
        )
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required")

    settings = load_settings()
    jwt_secret = settings.mcp.jwt_secret
    public_url = settings.mcp.public_url
    if jwt_secret is None or public_url is None:
        raise RuntimeError("DAIMON_MCP__JWT_SECRET and DAIMON_MCP__PUBLIC_URL are required")

    agent_name = cast(str, options.get("agent_name") or "daimon")
    environment_name = cast(str, options.get("environment_name") or "default")
    concurrency = int(options.get("concurrency", 1))
    case_timeout = float(options.get("case_timeout", 600.0))
    spend_cap = options.get("spend_cap_usd")

    run_id = state.run_id
    client = AsyncAnthropic()
    sql = await ReadOnlyPostgresExecutor.connect(warehouse_dsn)
    engine = build_engine(str(settings.database.url))
    session_factory = build_session_factory(engine)

    try:
        replay = await ManagedAgentReplayBackend.discover(
            client,
            agent_name=agent_name,
            environment_name=environment_name,
            run_id=run_id,
            jwt_secret=jwt_secret.get_secret_value().encode(),
            public_url=str(public_url),
            session_factory=session_factory,
            vault_token_ttl_hours=settings.mcp.vault_token_ttl_hours,
        )
    except Exception:
        await sql.close()
        await client.close()
        await engine.dispose()
        raise

    budget: Budget | None = None
    if spend_cap is not None and isinstance(spend_cap, (int, float)):
        budget = Budget(
            cap=float(spend_cap),
            max_case_cost=float(options.get("max_case_cost_usd", 0.5)),
            cost_fn=replay.case_cost,
            concurrency=int(options.get("concurrency", 1)),
            alarm_ceiling=replay.max_case_cost,
        )

    def on_case(case_id: str, status: str, case_result: CaseResult | None = None) -> None:
        state.add_case(case_id, status, case_result)

    try:
        result = await run_eval(
            run_id=run_id,
            goldens=goldens,
            replay=replay,
            sql=sql,
            case_timeout_s=case_timeout,
            concurrency=concurrency,
            on_case=on_case,
            budget=budget,
            measure_registry=load_measure_registry(settings.defaults_root),
        )
    finally:
        await sql.close()
        await client.close()
        await engine.dispose()

    out = state.temp_dir / "out.json"
    out_tmp = state.temp_dir / f"out.json.{secrets.token_hex(8)}"
    write_results(out_tmp, result)
    out_tmp.replace(out)

    receipt = budget.receipt() if budget is not None else "no spend cap configured"
    state.complete(result, receipt)
