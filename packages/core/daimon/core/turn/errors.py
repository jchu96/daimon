"""Typed errors raised by the turn admission chokepoint (D-01 stage one).

Split out of `admission.py` to keep that module under the 200-line target.
Both subclass `DaimonError` so the adapters' existing
`except (DaimonError, anthropic.APIError, <platform error>)` edges catch them
unchanged; adapters add specific handlers on top in the cutover plans.
"""

from __future__ import annotations

from typing import Literal

from daimon.core.errors import DaimonError
from daimon.core.scope import ConfigTier

AdmissionDenialReason = Literal["balance_depleted", "cap_exceeded"]
MissingConfigPart = Literal["agent", "environment"]


class AdmissionDenied(DaimonError):
    """Raised by `admit()` when a balance or cap gate rejects the turn.

    Carries only a closed reason literal — no rendered user-facing text.
    Denial copy stays adapter-side.
    """

    def __init__(self, *, reason: AdmissionDenialReason) -> None:
        super().__init__(reason)
        self.reason: AdmissionDenialReason = reason


class MissingTurnConfigError(DaimonError):
    """Raised by `admit()` when the config cascade resolves no agent and/or
    no environment for the channel/tenant/deployment scope.

    Carries structured data only (which parts are missing, and which tier —
    if any — each part's config resolved from) so an adapter can render its
    own hint text; never rendered text itself.
    """

    def __init__(
        self,
        *,
        missing: tuple[MissingConfigPart, ...],
        agent_name_tier: ConfigTier | None,
        environment_name_tier: ConfigTier | None,
    ) -> None:
        super().__init__(f"missing turn config: {missing!r}")
        self.missing: tuple[MissingConfigPart, ...] = missing
        self.agent_name_tier: ConfigTier | None = agent_name_tier
        self.environment_name_tier: ConfigTier | None = environment_name_tier
