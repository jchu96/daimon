"""Constants tests. SYSTEM_ACCOUNT_ID was removed in the tenant migration."""

from __future__ import annotations

import inspect

from daimon.adapters.discord.agent_setup import edit_view
from daimon.adapters.mcp.tools import agents as mcp_agents_tool
from daimon.core.constants import AGENT_MCP_CAP, AGENT_SKILL_CAP


def test_agent_caps_are_positive_ints() -> None:
    assert isinstance(AGENT_SKILL_CAP, int) and AGENT_SKILL_CAP > 0, (
        "AGENT_SKILL_CAP must be a positive int"
    )
    assert isinstance(AGENT_MCP_CAP, int) and AGENT_MCP_CAP > 0, (
        "AGENT_MCP_CAP must be a positive int"
    )


def test_adapter_modules_declare_no_private_cap_literal_of_their_own() -> None:
    """Both surfaces must read the one shared constant.

    A value-equality test (e.g. asserting some adapter-local number == 20)
    would keep passing if a future edit reintroduced a second private literal
    that happened to match today's cap — drift would go undetected until the
    two numbers diverged. Asserting the source itself carries no
    `_SKILL_CAP =` / `_MCP_CAP =` assignment catches the reintroduction at the
    moment it happens, not only when it later disagrees.
    """
    for module in (edit_view, mcp_agents_tool):
        source = inspect.getsource(module)
        assert "_SKILL_CAP =" not in source, (
            f"{module.__name__} must not declare a private _SKILL_CAP literal"
        )
        assert "_MCP_CAP =" not in source, (
            f"{module.__name__} must not declare a private _MCP_CAP literal"
        )
