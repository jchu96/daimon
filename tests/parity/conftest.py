"""Shared fixtures for the adapter parity suite.

Re-exports the standard per-worker-schema DB fixtures from `daimon.testing.db`
(the established pattern — no local Base import) and the shared turn router
(`daimon.testing.turn_router`) both `DiscordDriver` and `SlackDriver` use to
fake the MA transport for a turn (agent/environment resolution + SSE event
stream). The platform entry points themselves stay in their own driver
modules.
"""

from __future__ import annotations

import pytest
from daimon.testing.db import db_clean as db_clean  # noqa: F401
from daimon.testing.db import db_engine as db_engine  # noqa: F401
from daimon.testing.db import db_schema as db_schema  # noqa: F401
from daimon.testing.db import db_session as db_session  # noqa: F401
from daimon.testing.db import db_session_factory as db_session_factory  # noqa: F401
from daimon.testing.turn_router import (
    AGENT_ID,
    AGENT_TEXT,
    ENV_ID,
    MODEL_ID,
    build_turn_router,
    turn_events,
)

from .drivers.discord_driver import DiscordDriver
from .drivers.protocol import PlatformDriver
from .drivers.slack_driver import SlackDriver

__all__ = ["build_turn_router", "turn_events", "AGENT_TEXT", "AGENT_ID", "ENV_ID", "MODEL_ID"]


@pytest.fixture(params=["discord", "slack"], ids=["discord", "slack"])
def driver(request: pytest.FixtureRequest) -> PlatformDriver:
    """Yield the PlatformDriver under test, parametrized over both platforms."""
    if request.param == "discord":
        return DiscordDriver()
    return SlackDriver()
