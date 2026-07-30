"""Platform-neutral core for agent-authored, multi-step interactive forms.

This package lives in `daimon.core`, not in an adapter, because the two
processes that need it cannot import each other: the MCP server posts a
form on behalf of an agent, while the Discord bot later dispatches the
taps on that form's buttons and selects. Import-linter's independence
contract forbids `daimon.adapters.mcp` from importing
`daimon.adapters.discord` (or vice versa), so the shared schema, state,
and wire grammar have to sit one level up, in core. This layer emits a
platform-neutral screen description; each of the two processes owns a
thin renderer that turns that description into real platform components.

The wizard is deliberately Discord-only for now. The platform-neutral
screen description this package produces is the seam a future Slack
renderer would attach to — the pure core underneath does not change,
only a new renderer would be added. `tests/parity/test_wizard_discord_only.py`
is the executable record of that exemption: it fails the day a Slack
wizard renderer appears without being updated alongside it, rather than
silently going stale as a comment would.
"""

from __future__ import annotations
