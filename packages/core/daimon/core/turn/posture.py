"""Billing posture for a turn — the tagged union `run_turn` requires.

Per CONTEXT.md D-05/D-06/D-07: every call to `run_turn` must declare how the
turn is billed. `Billed` carries an event-only recorder (no `session_id`
passthrough — the recorder binds session/tenant context itself via
`functools.partial`, per RESEARCH "Pattern 2"). `BillingExempt` carries a
closed `ExemptReason` literal: widening the set of exempt reasons is a
deliberate, reviewable act (edit the `Literal`, not a free-form string),
not something a caller can invent inline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from anthropic.types.beta.sessions.beta_managed_agents_span_model_request_end_event import (
    BetaManagedAgentsSpanModelRequestEndEvent,
)

ExemptReason = Literal["cli-operator-run"]


class UsageRecorder(Protocol):
    async def __call__(self, *, event: BetaManagedAgentsSpanModelRequestEndEvent) -> None: ...


@dataclass(frozen=True)
class Billed:
    record: UsageRecorder


@dataclass(frozen=True)
class BillingExempt:
    reason: ExemptReason


BillingPosture = Billed | BillingExempt
