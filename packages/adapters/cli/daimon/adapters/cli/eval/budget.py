"""Fail-closed, bounded-overshoot per-battery spend budget for the eval runner.

The contract is NOT an a-priori hard cap. Per-case cost is not dominated a
priori because turn count is open-ended and cache pricing is not part of the
admission estimate. Instead, the budget guarantees a bounded overshoot:

    total spend <= cap + (concurrency - 1) * max_observed_case_cost

Admission uses a realistic per-case estimate: a case is admitted only while
``spent + in_flight_estimate + estimate(case) <= cap``. The budget reconciles
every case against its actual cost, stops admitting the moment cumulative
actual spend reaches the cap, and prints the bound and the observed total in
the receipt.

A separately-provided derived ceiling is used only as an anomaly alarm. If a
case's actual cost exceeds a configured multiple of the running median (or the
derived ceiling), the receipt flags it as anomalous and keeps the previous
"derived-ceiling invariant violated" wording when the derived ceiling is the
breached threshold.
"""

from __future__ import annotations

import asyncio
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field

from daimon.adapters.cli.eval.models import CaseResult


@dataclass
class Budget:
    """Bounded-overshoot budget for one battery.

    ``max_case_cost`` is a realistic per-case admission estimate.
    ``cost_fn`` returns the actual cost after a case completes. ``concurrency``
    is used to compute the honest overshoot bound: the worst case is
    ``(concurrency - 1)`` in-flight cases, each up to the largest observed
    actual cost, that complete after the cap is reached.

    ``alarm_ceiling`` is a derived per-case ceiling used only for anomaly
    detection. A case whose actual cost exceeds ``anomaly_multiplier * median
    cost`` or ``alarm_ceiling`` is reported in the receipt as anomalous.
    """

    cap: float
    max_case_cost: float
    cost_fn: Callable[[str], float | None] = field(default=lambda _id: None)
    concurrency: int = field(default=1)
    alarm_ceiling: float | None = field(default=None)
    anomaly_multiplier: float = field(default=3.0)

    _spent: float = field(default=0.0, init=False)
    _reserved: float = field(default=0.0, init=False)
    _exhausted: bool = field(default=False, init=False)
    _over_budget: bool = field(default=False, init=False)
    _aborted_cases: list[str] = field(default_factory=list[str], init=False)
    _reservations: dict[str, float] = field(default_factory=dict[str, float], init=False)
    _error: str | None = field(default=None, init=False)
    _offending_case: tuple[str, float, float] | None = field(default=None, init=False)
    _observed_costs: list[float] = field(default_factory=list[float], init=False)
    _max_observed_case_cost: float = field(default=0.0, init=False)
    _anomaly_case: tuple[str, float, float] | None = field(default=None, init=False)
    _invariant_violated: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.cap = round(self.cap, 6)
        self.max_case_cost = round(self.max_case_cost, 6)
        if self.alarm_ceiling is not None:
            self.alarm_ceiling = round(self.alarm_ceiling, 6)
        self._max_observed_case_cost = self.max_case_cost

    async def reserve(self, case_id: str) -> bool:
        """Admit ``case_id`` while the realistic estimate keeps spend under cap."""
        async with self._lock:
            if self._error is not None or self._exhausted or self._over_budget:
                self._aborted_cases.append(case_id)
                return False
            projected = round(self._spent + self._reserved + self.max_case_cost, 6)
            if projected > self.cap:
                self._exhausted = True
                self._aborted_cases.append(case_id)
                return False
            self._reservations[case_id] = self.max_case_cost
            self._reserved += self.max_case_cost
            return True

    async def record(self, case_id: str, case_result: CaseResult) -> None:
        """Reconcile actual spend, stop admitting at the cap, and alarm anomalies."""
        async with self._lock:
            reserved = self._reservations.pop(case_id, 0.0)
            self._reserved -= reserved
            if case_result.get("error_type") == "BudgetExhausted":
                return
            raw_actual = self.cost_fn(case_id)
            if raw_actual is None:
                self._error = f"Budget accounting error: no actual cost for case {case_id}"
                self._exhausted = True
                return
            actual = round(raw_actual, 6)
            self._spent = round(self._spent + actual, 6)
            self._observed_costs.append(actual)
            if actual > self._max_observed_case_cost:
                self._max_observed_case_cost = actual

            if (
                self.alarm_ceiling is not None
                and actual > self.alarm_ceiling
                and self._anomaly_case is None
            ):
                self._anomaly_case = (case_id, actual, self.alarm_ceiling)
                self._invariant_violated = True
            if self._observed_costs and self.anomaly_multiplier > 0:
                median_threshold = statistics.median(self._observed_costs) * self.anomaly_multiplier
                if actual > median_threshold and self._anomaly_case is None:
                    self._anomaly_case = (case_id, actual, median_threshold)

            if self._spent >= self.cap and self._offending_case is None:
                self._offending_case = (case_id, actual, self.max_case_cost)
                self._exhausted = True
            if self._spent > self.cap:
                self._over_budget = True

    def exhausted(self) -> bool:
        return self._exhausted or self._error is not None or self._over_budget

    def over_budget(self) -> bool:
        return self._over_budget

    def has_error(self) -> bool:
        return self._error is not None

    def offending_case_id(self) -> str | None:
        return self._offending_case[0] if self._offending_case is not None else None

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def aborted_cases(self) -> list[str]:
        return list(self._aborted_cases)

    def _overshoot_bound(self) -> float:
        return self.cap + (self.concurrency - 1) * self._max_observed_case_cost

    def receipt(self) -> str:
        bound = self._overshoot_bound()
        parts: list[str] = [
            f"spend cap {self.cap:.4f} USD: spent {self._spent:.4f}, "
            f"bounded overshoot bound {bound:.4f} USD "
            f"(max observed case cost {self._max_observed_case_cost:.4f}, "
            f"concurrency {self.concurrency})"
        ]
        if self._error is not None:
            parts.append(f"{len(self._aborted_cases)} case(s) aborted, error: {self._error}")
        if self._offending_case is not None:
            case_id, actual, _estimate = self._offending_case
            if self._over_budget:
                parts.append(f"over budget: cap overshoot at case {case_id} (actual {actual:.4f})")
            else:
                parts.append(f"cap reached at case {case_id} (actual {actual:.4f})")
        if self._anomaly_case is not None:
            case_id, actual, threshold = self._anomaly_case
            if self._invariant_violated:
                parts.append(
                    f"derived-ceiling invariant violated: case {case_id} "
                    f"exceeded ceiling (actual {actual:.4f} > ceiling {self.alarm_ceiling:.4f})"
                )
            else:
                parts.append(
                    f"anomalous case {case_id} (actual {actual:.4f} > threshold {threshold:.4f})"
                )
        if self._aborted_cases:
            parts.append(f"{len(self._aborted_cases)} case(s) aborted")
        return "; ".join(parts)
