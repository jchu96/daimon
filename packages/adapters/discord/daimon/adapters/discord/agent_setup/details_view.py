"""Details — one agent's whole readable state on the panel's one message.

The card answers "what can I use, and how do I change it?" without teaching
anybody the configuration structure: what the agent is for, where a mention
actually reaches it, what it is wired to, and the two ways to act on it — talk
to Daimon about it, or drive it from a coding tool.

``build_details_container`` is pure: it folds an `AgentDetails` into a
container and never reads a clock, a session or a credential. ``DetailsView``
is the shell that attaches the callbacks. The key list is names only — the
model it renders has no field for a value, and must never grow one.
"""

from __future__ import annotations

import structlog
from daimon.adapters.discord.agent_setup.budget import KEYS_COLLAPSED_COUNT, KEYS_EXPANDED_CAP
from daimon.adapters.discord.agent_setup.conversations import open_setup_conversation
from daimon.adapters.discord.agent_setup.mcp_access import send_coding_tools_access
from daimon.adapters.discord.agent_setup.navigation import PanelViewBase
from daimon.adapters.discord.agent_setup.state import PanelState
from daimon.adapters.discord.checks import is_guild_admin
from daimon.adapters.discord.layout import hairline, header
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.agent_details import AgentDetails, RepoBinding
from daimon.core.github_repo_auth import RepoAccess, normalize_owner_repo
from daimon.core.scope import AnsweringPlace
from daimon.core.setup_conversations import SETUP_ACTION_LABEL, shared_keys_sentence

import discord

log = structlog.get_logger()

CODING_TOOLS_LABEL = "🧰 Use from your coding tools"
BACK_LABEL = "◀ Back"
SHOW_ALL_LABEL = "Show all"
SHOW_FEWER_LABEL = "Show fewer"

CODING_TOOLS_HINT = "-# Use from your coding tools: get a token below, then `claude mcp add …`"


def coding_tools_refusal(agent_name: str) -> str:
    """What a member sees instead of a token, naming the permission and the way round it."""
    return (
        f"Minting an access token for {agent_name} needs Manage Server. "
        f"Ask an admin to open Details and use this button."
    )


def _answers_line(places: tuple[AnsweringPlace, ...]) -> str:
    """Where mentions reach this agent, as channel mentions plus named tiers."""
    parts: list[str] = []
    for place in places:
        if place.tier == "channel" and place.channel_id is not None:
            parts.append(f"<#{place.channel_id}>")
        elif place.tier == "tenant":
            parts.append("the server default")
        else:
            parts.append("the deployment default")
    return f"-# answers in {' · '.join(parts)}"


def _last_checked(access: RepoAccess) -> str:
    return (
        f" · last checked <t:{int(access.checked_at.timestamp())}:R>"
        if access.checked_at is not None
        else ""
    )


def _repo_access_line(access: RepoAccess) -> str:
    """Say exactly what was recorded — never read a URL or an App as proof."""
    if access.kind == "needs_attention":
        return f"⚠️ needs attention — {access.corrective or 'nothing would authorize a clone.'}"
    if access.kind == "not_checked":
        return "not checked yet"
    if access.credential == "per_agent_token":
        return f"via token{_last_checked(access)}"
    if access.credential == "deployment_public":
        return f"public repo{_last_checked(access)}"
    if access.credential == "github_app":
        return f"via GitHub App{_last_checked(access)}"
    return f"access recorded{_last_checked(access)}"


def _repo_text(repo: RepoBinding | None) -> str:
    if repo is None:
        return "📦 **Working repo**\n-# no working repo yet"
    slug = normalize_owner_repo(repo.repo_url)
    return (
        f"📦 **Working repo** [{slug}](https://github.com/{slug}) `{repo.default_branch}`\n"
        f"-# {_repo_access_line(repo.access)}"
    )


def _keys_text(details: AgentDetails, *, keys_expanded: bool) -> str:
    """Key names, never values, plus the sentence saying who they reach."""
    if not details.keys:
        return "🔑 **Keys**\n-# no keys yet"
    shown = (
        details.keys[:KEYS_EXPANDED_CAP] if keys_expanded else details.keys[:KEYS_COLLAPSED_COUNT]
    )
    body = "\n".join(entry.name for entry in shown)
    remainder = len(details.keys) - len(shown)
    if keys_expanded and remainder > 0:
        body = f"{body}\n-# +{remainder} more"
    return f"🔑 **Keys**\n{body}\n-# {shared_keys_sentence(details.name)}"


def _keys_item(
    details: AgentDetails, *, keys_expanded: bool
) -> discord.ui.Item[discord.ui.LayoutView]:
    """The key list, wearing a toggle only when there is something hidden behind it."""
    text: discord.ui.TextDisplay[discord.ui.LayoutView] = discord.ui.TextDisplay(
        _keys_text(details, keys_expanded=keys_expanded)
    )
    if len(details.keys) <= KEYS_COLLAPSED_COUNT:
        return text
    toggle: discord.ui.Button[discord.ui.LayoutView] = discord.ui.Button(
        label=SHOW_FEWER_LABEL if keys_expanded else SHOW_ALL_LABEL,
        style=discord.ButtonStyle.secondary,
    )
    section: discord.ui.Section[discord.ui.LayoutView] = discord.ui.Section(text, accessory=toggle)
    return section


def _skills_text(details: AgentDetails) -> str:
    if not details.skills:
        return "🧩 **Skills**\n-# no skills yet"
    body = "\n".join(skill.title or skill.skill_id for skill in details.skills)
    text = f"🧩 **Skills**\n{body}"
    if details.skills_listing_truncated:
        text = f"{text}\n-# some skill names may be missing"
    return text


def _mcp_text(details: AgentDetails) -> str:
    if not details.mcp_servers:
        return "🔌 **MCP servers**\n-# no servers yet"
    body = "\n".join(f"{server.name} — {server.url}" for server in details.mcp_servers)
    return f"🔌 **MCP servers**\n{body}"


def build_details_container(
    state: PanelState,
    details: AgentDetails,
    *,
    keys_expanded: bool,
    is_admin: bool,
    attribution: str | None,
) -> discord.ui.Container[discord.ui.LayoutView]:
    """Fold one agent's details into the panel card. Pure — no I/O, no clock.

    ``state`` and ``is_admin`` are carried for symmetry with the other two
    screens' builders; the role already reached this card through
    ``details.unrouted_note``, which core wrote in the reader's own voice.
    """
    del state, is_admin
    container: discord.ui.Container[discord.ui.LayoutView] = discord.ui.Container()
    container.add_item(header(f"🤖 {details.name}", subtext=details.purpose))
    if details.answers_in:
        container.add_item(discord.ui.TextDisplay(_answers_line(details.answers_in)))
    elif details.unrouted_note is not None:
        container.add_item(discord.ui.TextDisplay(details.unrouted_note))
    container.add_item(hairline())
    container.add_item(discord.ui.TextDisplay(f"**Model** {details.model_display_name}"))
    container.add_item(discord.ui.TextDisplay(_repo_text(details.repo)))
    container.add_item(_keys_item(details, keys_expanded=keys_expanded))
    container.add_item(discord.ui.TextDisplay(_skills_text(details)))
    container.add_item(discord.ui.TextDisplay(_mcp_text(details)))
    container.add_item(hairline())
    container.add_item(discord.ui.TextDisplay(CODING_TOOLS_HINT))
    if attribution is not None:
        container.add_item(discord.ui.TextDisplay(f"-# made by {attribution}"))
    return container


def _keys_toggle_button(
    container: discord.ui.Container[discord.ui.LayoutView],
) -> discord.ui.Button[discord.ui.LayoutView] | None:
    """Find the Show all / Show fewer accessory the builder left unwired."""
    for child in container.children:
        if isinstance(child, discord.ui.Section):
            accessory = child.accessory
            if isinstance(accessory, discord.ui.Button) and accessory.label in (
                SHOW_ALL_LABEL,
                SHOW_FEWER_LABEL,
            ):
                return accessory
    return None


class DetailsView(PanelViewBase):
    """The Details screen: one agent, two actions, and the way back.

    Rebuilt from ``state`` on every render, so expanding the key list or coming
    back from a setup conversation costs no refetch.
    """

    def __init__(
        self,
        state: PanelState,
        *,
        runtime: DiscordRuntime,
        allowed_user_id: int,
    ) -> None:
        super().__init__(state, runtime=runtime, allowed_user_id=allowed_user_id)
        details = state.details
        assert details is not None, "DetailsView needs a loaded AgentDetails on the panel state"
        self.details = details
        # Only a creator who resolves to a live Discord principal is named; the
        # panel resolved that once, at open, for every agent on the roster.
        self.attribution = state.attributions.get(details.ma_agent_id)

        container = build_details_container(
            state,
            details,
            keys_expanded=state.keys_expanded,
            is_admin=state.is_admin,
            attribution=self.attribution,
        )
        toggle = _keys_toggle_button(container)
        if toggle is not None:
            toggle.callback = self._on_toggle_keys  # type: ignore[method-assign]  # per-instance callback

        action_row: discord.ui.ActionRow[discord.ui.LayoutView] = discord.ui.ActionRow()
        setup_button: discord.ui.Button[discord.ui.LayoutView] = discord.ui.Button(
            label=SETUP_ACTION_LABEL, style=discord.ButtonStyle.primary
        )
        setup_button.callback = self._on_setup  # type: ignore[method-assign]  # per-instance callback
        coding_button: discord.ui.Button[discord.ui.LayoutView] = discord.ui.Button(
            label=CODING_TOOLS_LABEL, style=discord.ButtonStyle.secondary
        )
        coding_button.callback = self._on_coding_tools  # type: ignore[method-assign]  # per-instance callback
        action_row.add_item(setup_button)
        action_row.add_item(coding_button)
        container.add_item(action_row)

        nav_row: discord.ui.ActionRow[discord.ui.LayoutView] = discord.ui.ActionRow()
        back_button: discord.ui.Button[discord.ui.LayoutView] = discord.ui.Button(
            label=BACK_LABEL, style=discord.ButtonStyle.secondary
        )
        back_button.callback = self._on_back  # type: ignore[method-assign]  # per-instance callback
        nav_row.add_item(back_button)
        nav_row.add_item(self.done_button())  # pyright: ignore[reportArgumentType]  # Button[Self] is the same runtime item
        container.add_item(nav_row)

        self.add_item(container)

    async def _on_setup(self, interaction: discord.Interaction) -> None:
        """Open a setup conversation about the agent this card is describing."""
        log.info("agent_setup.details.setup.click", agent_name=self.details.name)
        await interaction.response.defer(ephemeral=True, thinking=True)
        await open_setup_conversation(
            interaction,
            runtime=self.runtime,
            state=self.state,
            target=self.state.selected_agent,
        )

    async def _on_coding_tools(self, interaction: discord.Interaction) -> None:
        """Mint a coding-tool token, but only for a caller who is an admin right now.

        The live re-check comes before anything else: the view's own
        ``is_admin`` is a snapshot from panel-open, and a caller demoted since
        then must reach no token material.
        """
        log.info("agent_setup.details.coding_tools.click", agent_name=self.details.name)
        if not is_guild_admin(interaction):  # pyright: ignore[reportArgumentType]  # reads only user and guild
            await interaction.response.send_message(
                coding_tools_refusal(self.details.name), ephemeral=True
            )
            return
        await send_coding_tools_access(
            interaction,
            runtime=self.runtime,
            state=self.state,
            allowed_user_id=self.allowed_user_id,
        )

    async def _on_toggle_keys(self, interaction: discord.Interaction) -> None:
        """Flip the key list between the collapsed count and the full list."""
        self.state.keys_expanded = not self.state.keys_expanded
        await self.swap_to(
            interaction,
            DetailsView(self.state, runtime=self.runtime, allowed_user_id=self.allowed_user_id),
        )

    async def _on_back(self, interaction: discord.Interaction) -> None:
        """Return to the roster with its page and selection exactly as they were."""
        # Lazy import: the roster screen opens this one, so a top-level import
        # here would close the cycle.
        from daimon.adapters.discord.agent_setup.roster_view import RosterView

        await self.swap_to(
            interaction,
            RosterView(self.state, runtime=self.runtime, allowed_user_id=self.allowed_user_id),
        )
