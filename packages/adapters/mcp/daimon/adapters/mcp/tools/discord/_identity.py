"""Discord display identity: daimon's own nickname and avatar in a server.

Provides: _set_display_identity_impl.

Both fields live on the bot's guild member (``PATCH /guilds/{id}/members/@me``),
so they apply to the whole server; Discord has no per-channel identity. A
server-wide, everyone-visible change is a tenant-wide mutation, so it is
gated on a server admin rather than a channel permission. Follows the locked
sequence from ``_read.py``: validate inputs -> rest_client -> _resolve_member
FIRST -> typed discord.py call -> map row.
"""

from __future__ import annotations

import aiohttp
import discord
from daimon.adapters.mcp.auth.resolver import AuthIdentity
from daimon.adapters.mcp.runtime import McpRuntime
from daimon.adapters.mcp.tools._ctx import _require_admin  # pyright: ignore[reportPrivateUsage]
from daimon.adapters.mcp.tools.discord._client import (
    _require_bot_token,  # pyright: ignore[reportPrivateUsage]
    _require_discord_identity,  # pyright: ignore[reportPrivateUsage]
    _require_guild_id,  # pyright: ignore[reportPrivateUsage]
    _resolve_member,  # pyright: ignore[reportPrivateUsage]
    rest_client,  # pyright: ignore[reportPrivateUsage]
)
from daimon.adapters.mcp.tools.discord._models import DisplayIdentityRow
from daimon.adapters.mcp.tools.discord._send import (
    _fetch_attachment,  # pyright: ignore[reportPrivateUsage]
)
from fastmcp.exceptions import ToolError

# Discord's nickname limit.
_MAX_DISPLAY_NAME_CHARS = 32

_SERVER_WIDE_HINT = (
    "Display name and avatar apply to the whole server; Discord has no per-channel identity."
)


def _validate_display_name(name: str) -> str:
    stripped = name.strip()
    if not stripped:
        raise ToolError("display_name must not be empty")
    if len(stripped) > _MAX_DISPLAY_NAME_CHARS:
        raise ToolError(f"display_name must be at most {_MAX_DISPLAY_NAME_CHARS} characters")
    return stripped


async def _set_display_identity_impl(  # pyright: ignore[reportUnusedFunction]
    runtime: McpRuntime,
    auth: AuthIdentity,
    *,
    display_name: str | None = None,
    avatar_url: str | None = None,
    session: aiohttp.ClientSession | None = None,
) -> DisplayIdentityRow:
    """Set daimon's nickname and/or avatar in the caller's guild.

    ``avatar_url`` must be a Discord CDN link (the signed URL of a user's
    attachment); ``_fetch_attachment`` enforces the host allowlist and size
    cap. Discord itself rejects anything but png/jpeg/gif/webp, surfaced by
    discord.py as ``ValueError`` before any request is made.
    """
    if display_name is None and avatar_url is None:
        raise ToolError("pass display_name, avatar_url or both")
    validated_name = _validate_display_name(display_name) if display_name is not None else None
    _require_admin(auth)
    user_id = _require_discord_identity(auth)
    guild_id = _require_guild_id(auth)
    token = _require_bot_token(runtime)

    avatar_bytes: bytes | None = None
    if avatar_url is not None:
        if session is not None:
            avatar_bytes = await _fetch_attachment(session, avatar_url)
        else:
            async with aiohttp.ClientSession() as http_session:
                avatar_bytes = await _fetch_attachment(http_session, avatar_url)

    async with rest_client(token) as c:
        guild, _ = await _resolve_member(c, guild_id, user_id)
        if c.user is None:
            raise ToolError("internal: discord client has no user")
        me = await guild.fetch_member(c.user.id)
        try:
            edited = await me.edit(
                nick=validated_name if validated_name is not None else discord.utils.MISSING,
                avatar=avatar_bytes if avatar_bytes is not None else discord.utils.MISSING,
            )
        except ValueError as e:
            raise ToolError("avatar_url must point to a png, jpeg, gif or webp image") from e
        except discord.Forbidden as e:
            raise ToolError(
                "daimon is missing the change_nickname permission needed to change how it appears"
            ) from e
        except discord.HTTPException as e:
            raise ToolError(f"discord refused the change: {e.text}") from e
        # Member.edit returns the updated member only when Discord echoes one back.
        shown = edited if edited is not None else me
        return DisplayIdentityRow(
            guild_id=guild_id,
            display_name=shown.display_name,
            avatar_url=shown.display_avatar.url,
            hint=_SERVER_WIDE_HINT,
        )
