"""Transport-level tests for _post_wizard_impl (posting a wizard's first screen).

Each test calls `_post_wizard_impl` directly with a hand-built `AuthIdentity`
and a transport-level patched `discord.http.HTTPClient` (`patch_discord_http`),
exactly like `test_discord.py`'s credential-button tests. Inline route
handlers and payload builders per call site (no DRY across tests) so SDK-
payload drift breaks the relevant test, per guideline:testing.

discord.py sends the create-message request as `json=payload` when the
message carries no files, or as multipart (`files=..., form=[...]`) when it
does -- the wizard post always attaches at least the view, and attaches real
files whenever any step declares an `image_handle`. `_payload_from_kwargs`
reads whichever transport the captured kwargs carry.
"""

from __future__ import annotations

import importlib.util
import json
import uuid
from pathlib import Path
from typing import Any

import discord
import discord.http
import pytest
from anthropic import AsyncAnthropic
from daimon.adapters.mcp.auth.resolver import AuthIdentity
from daimon.adapters.mcp.file_store import FileStore
from daimon.adapters.mcp.runtime import McpRuntime
from daimon.adapters.mcp.tools.discord import (
    _post_wizard_impl,  # pyright: ignore[reportPrivateUsage]
)
from daimon.core.config import AnthropicSettings, DatabaseSettings, DiscordSettings, Settings
from daimon.core.scope import DeploymentDefault
from daimon.core.stores.domain import Role
from daimon.core.wizard.spec import Option, Step, StepKind, WizardSpec
from fastmcp.exceptions import ToolError
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Load patch_discord_http directly from the sibling conftest.py by file path,
# matching test_discord.py's workaround for the "from conftest import ..."
# collision with the parent tests/conftest.py.
_conftest_path = Path(__file__).parent / "conftest.py"
_spec = importlib.util.spec_from_file_location("_tools_conftest", _conftest_path)
assert _spec is not None and _spec.loader is not None
_tools_conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tools_conftest)
patch_discord_http = _tools_conftest.patch_discord_http

pytestmark = pytest.mark.asyncio

# Permission flag constants (Discord docs).
_VIEW_CHANNEL = 1 << 10  # 1024
_SEND_MESSAGES = 1 << 11  # 2048


def _payload_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the parsed JSON payload regardless of json vs multipart transport."""
    if "json" in kwargs:
        return kwargs["json"]  # type: ignore[no-any-return]
    for part in kwargs["form"]:
        if part["name"] == "payload_json":
            return json.loads(part["value"])  # type: ignore[no-any-return]
    raise AssertionError(f"no payload_json part in multipart form: {kwargs!r}")


def _runtime_with_discord_token(*, file_store: FileStore | None) -> McpRuntime:
    """Build an McpRuntime with a Settings carrying a discord.bot_token.

    `session_factory` and `client` are real (unbound/unrouted) SDK objects
    the impl never touches in these tests -- `rest_client` opens its own
    per-call Discord HTTP client instead of reusing either.
    """
    settings = Settings(
        database=DatabaseSettings(url="postgresql+asyncpg://x/y"),  # pyright: ignore[reportArgumentType]
        anthropic=AnthropicSettings(api_key=SecretStr("k")),
        discord=DiscordSettings(bot_token=SecretStr("test-bot-token")),
    )
    session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker()  # type: ignore[call-arg]
    return McpRuntime(
        session_factory=session_factory,
        client=AsyncAnthropic(api_key="k"),
        settings=settings,
        deployment_default=DeploymentDefault(),
        file_store=file_store,
    )


def _auth(*, external_id: str = "111", platform_user_id: str = "42") -> AuthIdentity:
    return AuthIdentity(
        account_id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        role=Role.USER,
        platform="discord",
        external_id=external_id,
        platform_user_id=platform_user_id,
    )


def _guild_payload(*, guild_id: str = "111", owner_id: str = "1") -> dict[str, Any]:
    return {
        "id": guild_id,
        "name": "test-guild",
        "owner_id": owner_id,
        "afk_timeout": 0,
        "verification_level": 0,
        "default_message_notifications": 0,
        "explicit_content_filter": 0,
        "roles": [],
        "emojis": [],
        "features": [],
        "mfa_level": 0,
        "system_channel_flags": 0,
        "premium_tier": 0,
        "preferred_locale": "en-US",
        "nsfw_level": 0,
        "premium_progress_bar_enabled": False,
        "stickers": [],
        "region": "us-east",
    }


def _everyone_role(guild_id: str, perms: int) -> dict[str, Any]:
    return {
        "id": guild_id,
        "name": "@everyone",
        "permissions": str(perms),
        "position": 0,
        "color": 0,
        "hoist": False,
        "managed": False,
        "mentionable": False,
        "flags": 0,
    }


def _member_payload(user_id: str = "42") -> dict[str, Any]:
    return {
        "user": {
            "id": user_id,
            "username": "caller",
            "discriminator": "0001",
            "global_name": "caller",
            "avatar": None,
            "bot": False,
            "flags": 0,
        },
        "roles": [],
        "joined_at": "2024-01-01T00:00:00+00:00",
        "deaf": False,
        "mute": False,
        "flags": 0,
    }


def _text_channel_payload(*, channel_id: str = "222", guild_id: str = "111") -> dict[str, Any]:
    return {
        "id": channel_id,
        "type": 0,
        "guild_id": guild_id,
        "name": "general",
        "position": 0,
        "permission_overwrites": [],
        "nsfw": False,
        "rate_limit_per_user": 0,
        "parent_id": None,
    }


def _author_payload(author_id: str = "42") -> dict[str, Any]:
    return {
        "id": author_id,
        "username": "caller",
        "discriminator": "0001",
        "global_name": "caller",
        "avatar": None,
        "bot": False,
        "flags": 0,
    }


def _message_payload(
    *,
    message_id: str,
    channel_id: str = "222",
    attachments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": message_id,
        "channel_id": channel_id,
        "author": _author_payload(),
        "content": None,
        "timestamp": "2026-05-09T00:00:00+00:00",
        "edited_timestamp": None,
        "tts": False,
        "mention_everyone": False,
        "mentions": [],
        "mention_roles": [],
        "attachments": attachments or [],
        "embeds": [],
        "type": 0,
        "pinned": False,
        "flags": 32768,
    }


def _two_image_spec_for(handle1: str, handle2: str) -> WizardSpec:
    return WizardSpec(
        prompt="Configure the pipeline",
        steps=[
            Step(
                key="stage",
                question="Pick a stage",
                kind=StepKind.CHOICE,
                options=[Option(label="dev", value="dev")],
                image_handle=handle1,
            ),
            Step(
                key="region",
                question="Pick a region",
                kind=StepKind.CHOICE,
                options=[Option(label="us", value="us")],
                image_handle=handle2,
            ),
        ],
    )


def _happy_path_handler(
    posted: dict[str, Any], *, attachments: list[dict[str, Any]] | None = None
) -> Any:
    async def handler(route: discord.http.Route, kwargs: dict[str, Any]) -> Any:
        if route.path == "/guilds/{guild_id}":
            return _guild_payload()
        if route.path == "/guilds/{guild_id}/roles":
            return [_everyone_role("111", _VIEW_CHANNEL | _SEND_MESSAGES)]
        if route.path == "/guilds/{guild_id}/members/{member_id}":
            return _member_payload()
        if route.path == "/channels/{channel_id}":
            return _text_channel_payload()
        if route.method == "POST" and route.path == "/channels/{channel_id}/messages":
            posted.update(kwargs)
            return _message_payload(message_id="9201", attachments=attachments)
        raise AssertionError(f"unexpected route {route.method} {route.path}")

    return handler


async def test_post_wizard_sets_the_components_v2_flag_from_the_view(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    posted: dict[str, Any] = {}
    patch_discord_http(monkeypatch, _happy_path_handler(posted))
    spec = WizardSpec(
        prompt="p",
        steps=[
            Step(
                key="k", question="q", kind=StepKind.CHOICE, options=[Option(label="a", value="a")]
            )
        ],
    )
    await _post_wizard_impl(
        _runtime_with_discord_token(file_store=None),
        _auth(),
        channel_id="222",
        spec=spec,
        short_id="abcd1234",
    )
    payload = _payload_from_kwargs(posted)
    assert payload["flags"] == discord.MessageFlags(components_v2=True).value, (
        "the sent flags value must equal the library's own components_v2 flag value"
    )


async def test_post_wizard_sends_no_message_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    posted: dict[str, Any] = {}
    patch_discord_http(monkeypatch, _happy_path_handler(posted))
    spec = WizardSpec(
        prompt="p",
        steps=[
            Step(
                key="k", question="q", kind=StepKind.CHOICE, options=[Option(label="a", value="a")]
            )
        ],
    )
    await _post_wizard_impl(
        _runtime_with_discord_token(file_store=None),
        _auth(),
        channel_id="222",
        spec=spec,
        short_id="abcd1234",
    )
    payload = _payload_from_kwargs(posted)
    assert payload.get("content") is None, (
        "a components-v2 message cannot carry content; the sent payload must "
        "have no content or a null one"
    )


async def test_post_wizard_suppresses_every_mention(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The head text carries the agent-authored prompt and question verbatim,
    so an injected mass mention would ping the whole channel."""
    posted: dict[str, Any] = {}
    patch_discord_http(monkeypatch, _happy_path_handler(posted))
    spec = WizardSpec(
        prompt="@everyone pick one",
        steps=[
            Step(
                key="k",
                question="<@&999> which?",
                kind=StepKind.CHOICE,
                options=[Option(label="a", value="a")],
            )
        ],
    )
    await _post_wizard_impl(
        _runtime_with_discord_token(file_store=None),
        _auth(),
        channel_id="222",
        spec=spec,
        short_id="abcd1234",
    )
    payload = _payload_from_kwargs(posted)
    assert payload["allowed_mentions"] == discord.AllowedMentions.none().to_dict(), (
        "a wizard post must suppress every mention -- nothing in a form needs to ping"
    )


async def test_post_wizard_uploads_every_step_image_but_references_only_the_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = FileStore(base_dir=tmp_path)
    handle1 = store.put(data=b"PNGDATA1", mime_type="image/png", title="step1")
    handle2 = store.put(data=b"PNGDATA2", mime_type="image/png", title="step2")
    spec = _two_image_spec_for(handle1.id, handle2.id)
    posted: dict[str, Any] = {}
    patch_discord_http(monkeypatch, _happy_path_handler(posted))
    await _post_wizard_impl(
        _runtime_with_discord_token(file_store=store),
        _auth(),
        channel_id="222",
        spec=spec,
        short_id="abcd1234",
    )
    assert posted["files"] is not None and len(posted["files"]) == 2, (
        "both step images must upload even though only step one's shows"
    )
    payload = _payload_from_kwargs(posted)
    serialized = json.dumps(payload["components"])
    assert serialized.count("attachment://") == 1, (
        "later steps' images must mint content-delivery addresses without "
        "being displayed -- only step one's image may be referenced by a "
        "rendered component"
    )


async def test_post_wizard_returns_a_url_for_every_uploaded_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = FileStore(base_dir=tmp_path)
    handle1 = store.put(data=b"PNGDATA1", mime_type="image/png", title="step1")
    handle2 = store.put(data=b"PNGDATA2", mime_type="image/png", title="step2")
    spec = _two_image_spec_for(handle1.id, handle2.id)
    posted: dict[str, Any] = {}

    def handler_factory() -> Any:
        async def handler(route: discord.http.Route, kwargs: dict[str, Any]) -> Any:
            if route.path == "/guilds/{guild_id}":
                return _guild_payload()
            if route.path == "/guilds/{guild_id}/roles":
                return [_everyone_role("111", _VIEW_CHANNEL | _SEND_MESSAGES)]
            if route.path == "/guilds/{guild_id}/members/{member_id}":
                return _member_payload()
            if route.path == "/channels/{channel_id}":
                return _text_channel_payload()
            if route.method == "POST" and route.path == "/channels/{channel_id}/messages":
                posted.update(kwargs)
                return _message_payload(
                    message_id="9202",
                    attachments=[
                        {
                            "id": "8001",
                            "filename": "step1.png",
                            "url": "https://cdn.discordapp.com/attachments/1/2/step1.png?ex=abc123",
                            "proxy_url": "https://media.discordapp.net/attachments/1/2/step1.png?ex=abc123",
                            "size": 8,
                        },
                        {
                            "id": "8002",
                            "filename": "step2.png",
                            "url": "https://cdn.discordapp.com/attachments/1/2/step2.png?ex=def456",
                            "proxy_url": "https://media.discordapp.net/attachments/1/2/step2.png?ex=def456",
                            "size": 8,
                        },
                    ],
                )
            raise AssertionError(f"unexpected route {route.method} {route.path}")

        return handler

    patch_discord_http(monkeypatch, handler_factory())
    posted_wizard = await _post_wizard_impl(
        _runtime_with_discord_token(file_store=store),
        _auth(),
        channel_id="222",
        spec=spec,
        short_id="abcd1234",
    )
    assert posted_wizard.image_urls[handle1.id] == (
        "https://cdn.discordapp.com/attachments/1/2/step1.png?ex=abc123"
    ), "each handle must map to the durable url the platform minted for it"
    assert posted_wizard.image_urls[handle2.id] == (
        "https://cdn.discordapp.com/attachments/1/2/step2.png?ex=def456"
    ), "the later step's handle must also resolve, even though it isn't displayed"


async def test_post_wizard_maps_same_named_attachments_to_distinct_handles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A file's display name is slugged from the agent-supplied title, so two
    steps whose images share a title upload under one filename -- each handle
    must still resolve to its own attachment."""
    store = FileStore(base_dir=tmp_path)
    handle1 = store.put(data=b"PNGDATA1", mime_type="image/png", title="chart")
    handle2 = store.put(data=b"PNGDATA2", mime_type="image/png", title="chart")
    assert store.get(handle1.id).display_filename == store.get(handle2.id).display_filename, (
        "the two handles must genuinely collide on filename for this test to mean anything"
    )
    spec = _two_image_spec_for(handle1.id, handle2.id)

    async def handler(route: discord.http.Route, kwargs: dict[str, Any]) -> Any:
        if route.path == "/guilds/{guild_id}":
            return _guild_payload()
        if route.path == "/guilds/{guild_id}/roles":
            return [_everyone_role("111", _VIEW_CHANNEL | _SEND_MESSAGES)]
        if route.path == "/guilds/{guild_id}/members/{member_id}":
            return _member_payload()
        if route.path == "/channels/{channel_id}":
            return _text_channel_payload()
        if route.method == "POST" and route.path == "/channels/{channel_id}/messages":
            return _message_payload(
                message_id="9203",
                attachments=[
                    {
                        "id": "8001",
                        "filename": "chart.png",
                        "url": "https://cdn.discordapp.com/attachments/1/2/chart.png?ex=first",
                        "proxy_url": "https://media.discordapp.net/attachments/1/2/chart.png?ex=first",
                        "size": 8,
                    },
                    {
                        "id": "8002",
                        "filename": "chart.png",
                        "url": "https://cdn.discordapp.com/attachments/1/3/chart.png?ex=second",
                        "proxy_url": "https://media.discordapp.net/attachments/1/3/chart.png?ex=second",
                        "size": 8,
                    },
                ],
            )
        raise AssertionError(f"unexpected route {route.method} {route.path}")

    patch_discord_http(monkeypatch, handler)
    posted_wizard = await _post_wizard_impl(
        _runtime_with_discord_token(file_store=store),
        _auth(),
        channel_id="222",
        spec=spec,
        short_id="abcd1234",
    )

    assert posted_wizard.image_urls[handle1.id] == (
        "https://cdn.discordapp.com/attachments/1/2/chart.png?ex=first"
    ), "the first step's handle must resolve to the first uploaded attachment"
    assert posted_wizard.image_urls[handle2.id] == (
        "https://cdn.discordapp.com/attachments/1/3/chart.png?ex=second"
    ), "the second step must not inherit the first step's image"


async def test_post_wizard_rejects_a_missing_file_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = FileStore(base_dir=tmp_path)
    spec = WizardSpec(
        prompt="p",
        steps=[
            Step(
                key="k",
                question="q",
                kind=StepKind.CHOICE,
                options=[Option(label="a", value="a")],
                image_handle="nope.png",
            )
        ],
    )

    async def handler(_route: discord.http.Route, _kwargs: dict[str, Any]) -> Any:
        raise AssertionError("Discord HTTP must not be hit before the handle resolves")

    patch_discord_http(monkeypatch, handler)
    with pytest.raises(ToolError, match="nope.png"):
        await _post_wizard_impl(
            _runtime_with_discord_token(file_store=store),
            _auth(),
            channel_id="222",
            spec=spec,
            short_id="abcd1234",
        )


async def test_post_wizard_refuses_a_channel_the_requester_cannot_see(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = WizardSpec(
        prompt="p",
        steps=[
            Step(
                key="k", question="q", kind=StepKind.CHOICE, options=[Option(label="a", value="a")]
            )
        ],
    )

    async def handler(route: discord.http.Route, _kwargs: dict[str, Any]) -> Any:
        if route.path == "/guilds/{guild_id}":
            return _guild_payload()
        if route.path == "/guilds/{guild_id}/roles":
            return [_everyone_role("111", _VIEW_CHANNEL)]  # no send_messages
        if route.path == "/guilds/{guild_id}/members/{member_id}":
            return _member_payload()
        if route.path == "/channels/{channel_id}":
            return _text_channel_payload()
        if route.method == "POST" and route.path == "/channels/{channel_id}/messages":
            raise AssertionError("must not POST when send permission denied")
        raise AssertionError(f"unexpected route {route.method} {route.path}")

    patch_discord_http(monkeypatch, handler)
    with pytest.raises(ToolError, match="send_messages"):
        await _post_wizard_impl(
            _runtime_with_discord_token(file_store=None),
            _auth(),
            channel_id="222",
            spec=spec,
            short_id="abcd1234",
        )


async def test_post_wizard_posts_a_form_with_no_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    posted: dict[str, Any] = {}
    patch_discord_http(monkeypatch, _happy_path_handler(posted))
    spec = WizardSpec(
        prompt="p",
        steps=[
            Step(
                key="k", question="q", kind=StepKind.CHOICE, options=[Option(label="a", value="a")]
            )
        ],
    )
    posted_wizard = await _post_wizard_impl(
        _runtime_with_discord_token(file_store=None),
        _auth(),
        channel_id="222",
        spec=spec,
        short_id="abcd1234",
    )
    assert posted_wizard.image_urls == {}, "a form with no images uploads nothing"
    payload = _payload_from_kwargs(posted)
    assert len(payload["components"]) == 1, "still exactly one top-level container"
    assert payload["flags"] == discord.MessageFlags(components_v2=True).value, (
        "the v2 flag must still be set even with zero attachments"
    )
