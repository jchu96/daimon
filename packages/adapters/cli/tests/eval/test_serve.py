"""Contract tests for the eval serve HTTP mode and consumer client."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import httpx
import pytest
from daimon.adapters.cli.eval.budget import Budget
from daimon.adapters.cli.eval.models import CaseResult, Golden
from daimon.adapters.cli.eval.runner import run_eval
from daimon.adapters.cli.eval.serve import EvalServe, build_app

_DEFAULT_FAKE_PRICING: dict[str, dict[str, int | float]] = {
    "claude-sonnet-5": {
        "input": 1.0,
        "output": 0.0,
        "max_input_tokens": 1_000_000,
        "max_output_tokens": 0,
    },
}


class _ReplayStub:
    """Deterministic replay backend with per-case actual costs."""

    def __init__(
        self,
        answers: dict[str, str],
        case_cost: float | None = 0.0,
        case_costs: dict[str, float] | None = None,
        derived_ceiling: float = 1.0,
    ) -> None:
        self.answers = answers
        self._case_cost = case_cost
        self._case_costs = case_costs or {}
        self._derived_ceiling = derived_ceiling

    async def replay(self, *, case_id: str, question: str) -> str:  # noqa: ARG002
        return self.answers[case_id]

    def case_cost(self, case_id: str) -> float | None:  # noqa: ARG002
        return self._case_costs.get(case_id, self._case_cost)

    @property
    def max_case_cost(self) -> float:
        """Derived ceiling from the fake model's token limits and rates."""
        return self._derived_ceiling


class _SqlStub:
    """Deterministic SQL oracle."""

    def __init__(self, scalar_error: str | None = None) -> None:
        self._scalar_error = scalar_error

    async def scalar(self, query: str) -> object:  # noqa: ARG002
        if self._scalar_error is not None:
            raise RuntimeError(self._scalar_error)
        return 10.0

    async def strings(self, query: str) -> set[str]:  # noqa: ARG002
        return {"Internal billing note"}


def _derive_alarm_ceiling(options: dict[str, Any]) -> float:
    """Derive the anomaly alarm ceiling from the fake model's token limits and rates."""
    model_id = cast(str, options.get("model_id", "claude-sonnet-5"))
    pricing = options.get("pricing", _DEFAULT_FAKE_PRICING)
    rates = pricing.get(model_id)
    if rates is None:
        raise RuntimeError(f"no pricing for model {model_id}")
    if "max_input_tokens" not in rates or "max_output_tokens" not in rates:
        raise RuntimeError(f"model {model_id} has no token limits")
    max_input = int(rates["max_input_tokens"])
    max_output = int(rates["max_output_tokens"])
    return (max_input * float(rates["input"]) + max_output * float(rates["output"])) / 1_000_000.0


async def _fake_battery(state, goldens: list[Golden], options: dict[str, Any]) -> None:
    """The same runner path, but with test-only stubs."""
    answers = {golden["id"]: "Answer." for golden in goldens}
    alarm_ceiling = None
    replay = _ReplayStub(
        answers,
        case_cost=options.get("case_cost", 0.0),
        case_costs=options.get("case_costs"),
        derived_ceiling=1.0,
    )
    sql = _SqlStub(scalar_error=options.get("scalar_error"))
    budget: Budget | None = None
    spend_cap = options.get("spend_cap_usd")
    if spend_cap is not None:
        alarm_ceiling = _derive_alarm_ceiling(options)
        replay = _ReplayStub(
            answers,
            case_cost=options.get("case_cost", 0.0),
            case_costs=options.get("case_costs"),
            derived_ceiling=alarm_ceiling,
        )
        budget = Budget(
            cap=float(spend_cap),
            max_case_cost=float(options.get("max_case_cost_usd", 0.5)),
            cost_fn=replay.case_cost,
            concurrency=int(options.get("concurrency", 1)),
            alarm_ceiling=alarm_ceiling,
        )

    def on_case(case_id: str, status: str, case_result: CaseResult | None = None) -> None:
        state.add_case(case_id, status, case_result)

    try:
        result = await run_eval(
            run_id=state.run_id,
            goldens=goldens,
            replay=replay,
            sql=sql,
            concurrency=int(options.get("concurrency", 1)),
            on_case=on_case,
            budget=budget,
        )
    except Exception as error:
        state.fail(str(error))
        return
    receipt = budget.receipt() if budget is not None else "no spend cap"
    state.complete(result, receipt)


_GOLDEN: Golden = {
    "id": "one",
    "question": "What is the answer?",
    "expected_properties": {},
    "gap_shaped": False,
}


def _client(app, token: str | None = "test-token") -> httpx.AsyncClient:
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
        headers=headers,
    )


def _app(token: str = "test-token", run_battery: Any = _fake_battery) -> Any:
    return build_app(EvalServe(token=token, measure_registry=None, run_battery=run_battery))


@pytest.fixture
def app():
    return _app()


@pytest.mark.asyncio
async def test_healthz(app) -> None:
    async with _client(app, token=None) as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_post_eval_requires_auth(app) -> None:
    async with _client(app, token=None) as client:
        response = await client.post("/eval", json={"goldens": [_GOLDEN]})
    assert response.status_code == 401

    async with _client(app, token="wrong") as client:
        response = await client.post("/eval", json={"goldens": [_GOLDEN]})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_post_eval_returns_202_and_can_poll_status(app) -> None:
    async with _client(app) as client:
        response = await client.post("/eval", json={"goldens": [_GOLDEN]})
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "pending"
    assert "run_id" in payload

    run_id = payload["run_id"]
    async with _client(app) as client:
        for _ in range(50):
            status = (await client.get(f"/eval/{run_id}")).json()
            if status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.05)

    assert status["status"] == "completed"
    assert status["receipt"] == "no spend cap"
    assert status["artifact"]["summary"]["cases"] == 1
    assert status["artifact"]["cases"][0]["id"] == "one"


@pytest.mark.asyncio
async def test_concurrent_run_returns_409(app) -> None:
    async def slow_battery(state, goldens, options):
        await asyncio.sleep(0.3)
        await _fake_battery(state, goldens, options)

    slow_app = _app(run_battery=slow_battery)
    async with _client(slow_app) as client:
        first = await client.post("/eval", json={"goldens": [_GOLDEN]})
        assert first.status_code == 202
        second = await client.post("/eval", json={"goldens": [_GOLDEN]})
        assert second.status_code == 409


@pytest.mark.asyncio
async def test_allow_concurrent_bypasses_409(app) -> None:
    async def slow_battery(state, goldens, options):
        await asyncio.sleep(0.3)
        await _fake_battery(state, goldens, options)

    slow_app = _app(run_battery=slow_battery)
    async with _client(slow_app) as client:
        first = await client.post(
            "/eval",
            json={"goldens": [_GOLDEN], "options": {"allow_concurrent": True}},
        )
        assert first.status_code == 202
        second = await client.post(
            "/eval",
            json={"goldens": [_GOLDEN], "options": {"allow_concurrent": True}},
        )
        assert second.status_code == 202


@pytest.mark.asyncio
async def test_spend_cap_aborts_remaining_cases(app) -> None:
    goldens = [
        {**_GOLDEN, "id": "first"},
        {**_GOLDEN, "id": "second"},
        {**_GOLDEN, "id": "third"},
    ]
    options = {
        "concurrency": 1,
        "case_cost": 1.0,
        "max_case_cost_usd": 1.0,
        "spend_cap_usd": 2.0,
    }

    status = await _await_completion(app, goldens, options)

    assert status["status"] == "completed"
    receipt = status["artifact"]["summary"]["budget_receipt"]
    assert "spend cap 2.0000 USD" in receipt
    assert "1 case(s) aborted" in receipt
    cases = {c["id"]: c for c in status["artifact"]["cases"]}
    assert cases["third"].get("error_type") == "BudgetExhausted"
    assert cases["third"]["passed"] is False


@pytest.mark.asyncio
async def test_spend_cap_holds_under_concurrency(app) -> None:
    """3 cases, concurrency 3, cap 1.0, estimate 1.0: cap reached after first."""
    goldens = [
        {**_GOLDEN, "id": "c0"},
        {**_GOLDEN, "id": "c1"},
        {**_GOLDEN, "id": "c2"},
    ]
    options = {
        "concurrency": 3,
        "case_cost": 1.0,
        "max_case_cost_usd": 1.0,
        "spend_cap_usd": 1.0,
    }

    status = await _await_completion(app, goldens, options)

    cases = {c["id"]: c for c in status["artifact"]["cases"]}
    assert cases["c0"]["passed"] is True
    assert cases["c1"].get("error_type") == "BudgetExhausted"
    assert cases["c2"].get("error_type") == "BudgetExhausted"
    receipt = status["artifact"]["summary"]["budget_receipt"]
    assert "2 case(s) aborted" in receipt
    assert "spent 1.0000" in receipt
    assert "bounded overshoot bound 3.0000" in receipt


@pytest.mark.asyncio
async def test_spend_cap_fail_closed_on_unknown_cost(app) -> None:
    goldens = [
        {**_GOLDEN, "id": "first"},
        {**_GOLDEN, "id": "second"},
    ]
    options = {
        "concurrency": 1,
        "case_cost": None,
        "max_case_cost_usd": 1.0,
        "spend_cap_usd": 10.0,
    }

    status = await _await_completion(app, goldens, options)

    receipt = status["artifact"]["summary"]["budget_receipt"]
    assert "Budget accounting error" in receipt
    cases = {c["id"]: c for c in status["artifact"]["cases"]}
    assert all(c.get("error_type") == "BudgetExhausted" for c in cases.values())


@pytest.mark.asyncio
async def test_spend_cap_reservation_is_load_bearing(app, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without reservation, in-flight cases still record actual spend and mark over-budget."""
    goldens = [
        {**_GOLDEN, "id": "c0"},
        {**_GOLDEN, "id": "c1"},
        {**_GOLDEN, "id": "c2"},
    ]
    options = {
        "concurrency": 3,
        "case_cost": 1.0,
        "max_case_cost_usd": 1.0,
        "spend_cap_usd": 1.0,
    }

    async def _reserve_always_permits(self, case_id: str) -> bool:  # noqa: ARG002
        return True

    monkeypatch.setattr(
        "daimon.adapters.cli.eval.budget.Budget.reserve",
        _reserve_always_permits,
    )

    status = await _await_completion(app, goldens, options)

    cases = {c["id"]: c for c in status["artifact"]["cases"]}
    assert status["artifact"]["summary"]["over_budget"] is True
    assert not all(c["passed"] for c in cases.values())
    assert any(c.get("error_type") == "BudgetExhausted" for c in cases.values())
    receipt = status["artifact"]["summary"]["budget_receipt"]
    assert "spent 3.0000" in receipt
    assert "over budget" in receipt


@pytest.mark.asyncio
async def test_spend_cap_is_load_bearing(app, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypassing the reserve gate still lets cases run, but accounting records overshoot."""
    goldens = [
        {**_GOLDEN, "id": "first"},
        {**_GOLDEN, "id": "second"},
        {**_GOLDEN, "id": "third"},
    ]
    options = {
        "concurrency": 1,
        "case_cost": 1.0,
        "max_case_cost_usd": 1.0,
        "spend_cap_usd": 2.0,
    }

    async def _permit(self, case_id: str) -> bool:  # noqa: ARG002
        return True

    monkeypatch.setattr(
        "daimon.adapters.cli.eval.budget.Budget.reserve",
        _permit,
    )

    status = await _await_completion(app, goldens, options)

    cases = {c["id"]: c for c in status["artifact"]["cases"]}
    assert not all(c["passed"] for c in cases.values())
    assert any(c.get("error_type") == "BudgetExhausted" for c in cases.values())
    assert status["artifact"]["summary"]["over_budget"] is True
    receipt = status["artifact"]["summary"]["budget_receipt"]
    assert "over budget" in receipt


@pytest.mark.asyncio
async def test_spend_cap_bounded_overshoot_barrier(app) -> None:
    """Reviewer's barrier probe: 3 cases, concurrency 3, actual overshoots the estimate."""
    goldens = [
        {**_GOLDEN, "id": "c0"},
        {**_GOLDEN, "id": "c1"},
        {**_GOLDEN, "id": "c2"},
    ]
    options = {
        "concurrency": 3,
        "case_costs": {"c0": 1.44576, "c1": 1.44576, "c2": 1.44576},
        "max_case_cost_usd": 1.0,
        "spend_cap_usd": 3.0,
    }

    status = await _await_completion(app, goldens, options)

    assert status["status"] == "completed"
    total = float(
        status["artifact"]["summary"]["budget_receipt"].split("spent")[1].split(",")[0].strip()
    )
    bound = float(
        status["artifact"]["summary"]["budget_receipt"]
        .split("bounded overshoot bound")[1]
        .split("USD")[0]
        .strip()
    )
    assert total <= bound, f"{total} > {bound}"
    assert "bounded overshoot bound" in status["artifact"]["summary"]["budget_receipt"]
    assert "derived-ceiling invariant violated" in status["artifact"]["summary"]["budget_receipt"]
    cases = {c["id"]: c for c in status["artifact"]["cases"]}
    assert not all(c["passed"] for c in cases.values())


@pytest.mark.asyncio
async def test_spend_cap_40_case_battery_runs_to_completion(app) -> None:
    """A realistic 40-case battery with a cap equal to its actual cost runs to completion."""
    goldens = [{**_GOLDEN, "id": f"c{i}"} for i in range(40)]
    options = {
        "concurrency": 1,
        "case_cost": 0.12,
        "max_case_cost_usd": 0.12,
        "spend_cap_usd": 4.8,
    }

    status = await _await_completion(app, goldens, options)

    assert status["status"] == "completed"
    cases = {c["id"]: c for c in status["artifact"]["cases"]}
    failing = [c["id"] for c in cases.values() if not c["passed"]]
    assert not failing, f"failing cases: {failing}"
    assert not any(c.get("error_type") == "BudgetExhausted" for c in cases.values())
    receipt = status["artifact"]["summary"]["budget_receipt"]
    assert "spent 4.8000" in receipt


@pytest.mark.asyncio
async def test_spend_cap_anomaly_by_median(app) -> None:
    """A case far above the running median is flagged as anomalous."""
    goldens = [
        {**_GOLDEN, "id": "c0"},
        {**_GOLDEN, "id": "c1"},
        {**_GOLDEN, "id": "c2"},
    ]
    options = {
        "concurrency": 1,
        "case_costs": {"c0": 0.12, "c1": 0.12, "c2": 0.5},
        "max_case_cost_usd": 0.12,
        "spend_cap_usd": 2.0,
    }

    status = await _await_completion(app, goldens, options)

    receipt = status["artifact"]["summary"]["budget_receipt"]
    assert "anomalous case c2" in receipt
    assert "c2" in receipt
    assert "0.5000" in receipt


@pytest.mark.asyncio
async def test_spend_cap_missing_pricing_fails_closed(app) -> None:
    """Unknown model id fails closed before any model call."""
    goldens = [
        {**_GOLDEN, "id": "first"},
    ]
    options = {
        "spend_cap_usd": 3.0,
        "model_id": "unknown-model",
    }

    status = await _await_completion(app, goldens, options)

    assert status["status"] == "failed"
    assert "no pricing" in status["receipt"].lower()


@pytest.mark.asyncio
async def test_spend_cap_missing_token_limit_fails_closed(app) -> None:
    """A priced model with missing token limits fails closed before any model call."""
    goldens = [
        {**_GOLDEN, "id": "first"},
    ]
    options = {
        "spend_cap_usd": 3.0,
        "pricing": {
            "claude-sonnet-5": {
                "input": 1.0,
                "output": 0.0,
            },
        },
    }

    status = await _await_completion(app, goldens, options)

    assert status["status"] == "failed"
    assert "token limits" in status["receipt"].lower()


@pytest.mark.asyncio
async def test_spend_cap_freed_headroom(app) -> None:
    """Reconciling down to the actual cost frees headroom for later cases."""
    goldens = [
        {**_GOLDEN, "id": "first"},
        {**_GOLDEN, "id": "second"},
        {**_GOLDEN, "id": "third"},
    ]
    options = {
        "concurrency": 1,
        "case_cost": 0.5,
        "max_case_cost_usd": 1.0,
        "spend_cap_usd": 2.0,
    }

    status = await _await_completion(app, goldens, options)

    assert status["status"] == "completed"
    cases = {c["id"]: c for c in status["artifact"]["cases"]}
    assert all(c["passed"] for c in cases.values())
    assert not status["artifact"]["summary"].get("over_budget", False)
    receipt = status["artifact"]["summary"]["budget_receipt"]
    assert "spent 1.5000" in receipt


@pytest.mark.asyncio
async def test_served_artifact_matches_run_eval_shape(app) -> None:
    goldens = [_GOLDEN]
    answers = {golden["id"]: "Answer." for golden in goldens}
    replay = _ReplayStub(answers)
    sql = _SqlStub()

    direct = await run_eval(
        run_id="parity-run",
        goldens=goldens,
        replay=replay,
        sql=sql,
        concurrency=1,
    )

    status = await _await_completion(
        app,
        goldens,
        {"run_id": "parity-run"},
    )

    served = status["artifact"]
    assert served["schema_version"] == direct["schema_version"]
    assert served["run_id"] == "parity-run"
    assert served["summary"]["mode"] == "run"
    assert served["summary"] == direct["summary"]
    assert len(served["cases"]) == len(direct["cases"])
    for s_case, d_case in zip(served["cases"], direct["cases"], strict=False):
        assert s_case["id"] == d_case["id"]
        assert s_case["passed"] == d_case["passed"]
        assert s_case["grades"] == d_case["grades"]
        assert s_case.get("error_type") == d_case.get("error_type")
        assert s_case.get("claims") == d_case.get("claims")


@pytest.mark.asyncio
async def test_events_stream_does_not_expose_prompts_or_sql(app) -> None:
    golden: Golden = {
        "id": "leaky",
        "question": "what is the secret?",
        "expected_sql": {
            "query": "SELECT 1",
            "answer_regex": r"(?P<value>10)",
            "tolerance": 0.01,
        },
        "expected_properties": {},
        "gap_shaped": False,
    }

    status = await _await_completion(app, [golden])
    events = await _collect_events(app, status["artifact"]["run_id"])
    assert len(events) == 1
    event = events[0]
    assert event["id"] == "leaky"
    assert "question" not in event
    assert "answer" not in event
    assert "sql" not in event
    assert "verdict" in event
    assert "duration_s" in event
    assert "error_class" in event


@pytest.mark.asyncio
async def test_events_stream_hides_sql_in_error_class(app) -> None:
    """Only the exception class name is emitted; the SQL message stays in the artifact."""
    golden: Golden = {
        "id": "sql-leak",
        "question": "what is the secret?",
        "expected_sql": {
            "query": "SELECT secret_column FROM private_table",
            "answer_regex": r"(?P<value>10)",
            "tolerance": 0.01,
        },
        "expected_properties": {},
        "gap_shaped": False,
    }
    sql_text = "database failed for SELECT secret_column FROM private_table"
    options = {"scalar_error": sql_text}

    status = await _await_completion(app, [golden], options)
    cases = {c["id"]: c for c in status["artifact"]["cases"]}
    assert sql_text in cases["sql-leak"].get("error_type", "")

    events = await _collect_events(app, status["artifact"]["run_id"])
    assert len(events) == 1
    event = events[0]
    assert event["id"] == "sql-leak"
    assert event["error_class"] == "RuntimeError"
    assert "SELECT" not in json.dumps(event)
    assert "private_table" not in json.dumps(event)


async def _await_completion(
    app, goldens: list[Golden], options: dict[str, Any] | None = None
) -> dict[str, Any]:
    options = options or {}
    status: dict[str, Any] = {"status": "pending"}
    run_id = ""

    async with _client(app) as client:
        response = await client.post(
            "/eval",
            json={"goldens": goldens, "options": options},
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        for _ in range(50):
            status = (await client.get(f"/eval/{run_id}")).json()
            if status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.05)

    if status["status"] == "completed":
        status["artifact"]["run_id"] = run_id
    return status


async def _collect_events(app, run_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async with _client(app) as client, client.stream("GET", f"/eval/{run_id}/events") as stream:
        async for line in stream.aiter_lines():
            if line:
                events.append(json.loads(line))
    return events
