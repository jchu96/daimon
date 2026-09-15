"""Pure Block Kit builders for the read-only setup panel.

Three screens — Agents, one agent's Details, and Who answers where — plus the
New agent form and the placeholder shown while a creation is in flight. Every
one of them ends in `finish_modal`, so a view that would break one of Slack's
limits raises here rather than being rejected by the API in front of a person.

The core models carry the facts and the sentences; this module decides only
how a Slack modal says them. It reads no settings, resolves no credential and
re-derives no precedence: an unrouted agent is unrouted because
`AgentDetails.answers_in` is empty, and the sentence about it is the one
`daimon.core.routing_facts` wrote. Admin and member see identical blocks; the
role changes only whose voice the routing request is in.

Pure — no I/O, no clock, no slack_sdk.
"""

from __future__ import annotations

import uuid
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from typing import Any, Final

from daimon.adapters.slack.agent_setup.state import (
    PanelMetadata,
    encode_panel_metadata,
)
from daimon.adapters.slack.modal_limits import (
    MAX_SECTION_TEXT_CHARS,
    finish_modal,
    fit_title,
)
from daimon.adapters.slack.mrkdwn import escape_mrkdwn, escape_mrkdwn_preserving_mentions
from daimon.adapters.slack.setup_conversations import setup_button
from daimon.core.agent_details import AgentDetails
from daimon.core.answering_map import AnsweringMap, ChannelAnswer
from daimon.core.github_repo_auth import RepoAccess, normalize_owner_repo
from daimon.core.models_catalog import ModelChoice
from daimon.core.roster import Page, Roster, RosterAgent
from daimon.core.routing_facts import (
    PRECEDENCE_LINE,
    UNROUTED_LINE,
    build_routing_request,
)
from daimon.core.scope import AnsweringPlace, ConfigTier
from daimon.core.setup_conversations import EMPTY_ROSTER_COPY, shared_keys_sentence

__all__ = [
    "ACTION_CODING_TOOLS",
    "ACTION_DETAILS",
    "ACTION_EXPAND_KEYS",
    "ACTION_EXPAND_SKILLS",
    "ACTION_NEW",
    "ACTION_PAGE_NEXT",
    "ACTION_PAGE_PREV",
    "ACTION_REVOKE_TOKEN",
    "ACTION_ROUTING",
    "CALLBACK_AGENTS",
    "CALLBACK_CREATING",
    "CALLBACK_DETAILS",
    "CALLBACK_NEW_AGENT",
    "CALLBACK_ROUTING",
    "LEGACY_ACTION_IDS",
    "build_agents_view",
    "build_creating_view",
    "build_details_view",
    "build_new_agent_form",
    "build_routing_view",
]

# ---------------------------------------------------------------------------
# Action and callback identifiers
# ---------------------------------------------------------------------------

ACTION_DETAILS: Final = "agent_setup__details"
"""Open one agent's Details. The button's `value` is the agent name."""

ACTION_ROUTING: Final = "agent_setup__routing"
ACTION_PAGE_NEXT: Final = "agent_setup__page:next"
ACTION_PAGE_PREV: Final = "agent_setup__page:prev"
ACTION_EXPAND_KEYS: Final = "agent_setup__expand:keys"
ACTION_EXPAND_SKILLS: Final = "agent_setup__expand:skills"
ACTION_NEW: Final = "agent_setup__new"
ACTION_CODING_TOOLS: Final = "agent_setup__coding_tools"
"""Mint a coding-tool token for the agent named in the button's `value`."""

ACTION_REVOKE_TOKEN: Final = "agent_setup__revoke_token"
"""Revoke one minted token; the button's `value` is its jti.

Rendered by the coding-tools ephemeral rather than by any view here, but the
identifier belongs with the panel's other action ids so the dispatcher reads
one list.
"""

CALLBACK_AGENTS: Final = "agent_setup"
CALLBACK_DETAILS: Final = "agent_setup__details_view"
CALLBACK_ROUTING: Final = "agent_setup__routing_view"
CALLBACK_NEW_AGENT: Final = "agent_setup__new_agent"
CALLBACK_CREATING: Final = "agent_setup__creating"

LEGACY_ACTION_IDS: Final[frozenset[str]] = frozenset(
    {
        "agent_setup__roster_select",
        "agent_setup__fork",
        "agent_setup__edit",
        "agent_setup__delete",
        "agent_setup__scope:workspace",
        "agent_setup__scope:channel",
        "agent_setup__scope:clear",
        "agent_setup__tab:agent",
        "agent_setup__tab:repo_auth",
        "agent_setup__tab:skills",
        "agent_setup__tab:mcps",
        "agent_setup__tab:secrets",
        "agent_setup__edit_agent_form",
        "agent_setup__edit_repo_form",
        "agent_setup__add_skill",
        "agent_setup__remove_skill",
        "agent_setup__add_mcp",
        "agent_setup__remove_mcp",
        "agent_setup__connect_mcp",
        "agent_setup__paste_secrets",
        "agent_setup__remove_secret",
    }
)
"""Every action the editor panel emitted that the read-only panel must not.

`agent_setup__new` and `agent_setup__conversation` are deliberately absent:
both survive into the new panel with the same meaning, so a test that walks a
new view's action ids against this set would fail on a reused id that is
correct.
"""

# ---------------------------------------------------------------------------
# Copy and sizing
# ---------------------------------------------------------------------------

KEYS_COLLAPSED_COUNT: Final = 8
"""Key names shown before the list collapses behind Show all."""

KEYS_EXPANDED_CAP: Final = 60
"""Key names shown once expanded, before a `+N more` tail."""

MAX_SETUP_LINKS: Final = 10

DETAILS_BUTTON_LABEL: Final = "🔍 Details"
NEW_AGENT_LABEL: Final = "➕ New agent"
ROUTING_LABEL: Final = "📍 Who answers where"
CODING_TOOLS_LABEL: Final = "🧰 Use from your coding tools"

CODING_TOOLS_NOTE: Final = (
    "Use from your coding tools: get a token with the button below, then run the "
    "claude mcp add command it gives you."
)
CODING_TOOLS_UNAVAILABLE_NOTE: Final = "Coding-tool access is not configured for this deployment."

_TIER_PHRASES: Final[dict[ConfigTier, str]] = {
    "thread": "this thread",
    "channel": "channel setting",
    "tenant": "workspace default",
    "deployment": "deployment default",
}


# ---------------------------------------------------------------------------
# Small block helpers
# ---------------------------------------------------------------------------


def _clip(text: str) -> str:
    """Keep one section's text inside Slack's per-section cap."""
    if len(text) <= MAX_SECTION_TEXT_CHARS:
        return text
    return f"{text[: MAX_SECTION_TEXT_CHARS - 1]}…"


def _section(text: str, *, accessory: dict[str, Any] | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "section", "text": {"type": "mrkdwn", "text": _clip(text)}}
    if accessory is not None:
        block["accessory"] = accessory
    return block


def _context(*lines: str) -> dict[str, Any]:
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": _clip(line)} for line in lines],
    }


def _button(
    *, action_id: str, label: str, value: str | None = None, style: str | None = None
) -> dict[str, Any]:
    element: dict[str, Any] = {
        "type": "button",
        "action_id": action_id,
        "text": {"type": "plain_text", "text": label, "emoji": True},
    }
    if value is not None:
        element["value"] = value
    if style is not None:
        element["style"] = style
    return element


def _setup_elements(target_ma_agent_id: str | None) -> list[dict[str, Any]]:
    """The Set-up-with-Daimon button, styled as the view's primary action.

    Taken from `setup_conversations.setup_button` so the label, action id and
    `value` convention have one home — the panel only restyles it and puts it
    in a row with its neighbours.
    """
    elements: list[dict[str, Any]] = list(setup_button(target_ma_agent_id)["elements"])
    for element in elements:
        element["style"] = "primary"
    return elements


def _slack_date(moment: datetime) -> str:
    """A timestamp Slack renders in the reader's own timezone and locale."""
    return f"<!date^{int(moment.timestamp())}^{{date_short}}|{moment.date().isoformat()}>"


def _pager_blocks(page: Page[Any], *, meta: PanelMetadata) -> list[dict[str, Any]]:
    """Page counter and arrows, or nothing at all when there is one page."""
    if page.page_count <= 1:
        return []
    arrows: list[dict[str, Any]] = []
    if page.has_previous:
        arrows.append(
            _button(action_id=ACTION_PAGE_PREV, label="◀ Previous", value=str(page.page - 1))
        )
    if page.has_next:
        arrows.append(_button(action_id=ACTION_PAGE_NEXT, label="Next ▶", value=str(page.page + 1)))
    blocks = [_context(f"Page {page.page + 1} of {page.page_count}")]
    if arrows:
        blocks.append({"type": "actions", "elements": arrows})
    return blocks


def _place_labels(places: Sequence[AnsweringPlace]) -> list[str]:
    labels: list[str] = []
    for place in places:
        if place.tier == "channel" and place.channel_id is not None:
            labels.append(f"<#{place.channel_id}>")
        elif place.tier == "tenant":
            labels.append("the workspace default")
        else:
            labels.append("the deployment default")
    return labels


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def build_agents_view(
    roster: Roster,
    *,
    page: Page[RosterAgent],
    meta: PanelMetadata,
    is_admin: bool,
    attributions: Mapping[uuid.UUID, str],
    channel_id: str,
    routed_agent_names: Collection[str] | None = None,
) -> dict[str, Any]:
    """The root view: who answers here, then everyone else, then the actions.

    `is_admin` is accepted and deliberately unused in the block layout —
    members and admins see the same roster and the same entries, because
    entering a conversation about a change is not the change.

    `routed_agent_names` is every agent this install routes to somewhere. It
    is optional because the roster alone cannot tell "answers in another
    channel" from "answers nowhere", and the panel must not guess: without it
    a row carries no routing claim at all.
    """
    _ = is_admin
    blocks: list[dict[str, Any]] = []
    answering = roster.answering
    if not roster.rows:
        blocks.append(_section(EMPTY_ROSTER_COPY))
    for agent in page.items:
        is_answering = answering is not None and agent.name == answering.name
        heading = (
            f"*{escape_mrkdwn(agent.name)}* answers here"
            if is_answering
            else f"*{escape_mrkdwn(agent.name)}*"
        )
        blocks.append(
            _section(
                heading,
                accessory=_button(
                    action_id=ACTION_DETAILS, label=DETAILS_BUTTON_LABEL, value=agent.name
                ),
            )
        )
        facts = _roster_row_facts(
            agent,
            is_answering=is_answering,
            attributions=attributions,
            routed_agent_names=routed_agent_names,
        )
        if facts:
            blocks.append(_context(" · ".join(facts)))
    blocks.append({"type": "divider"})
    elements = _setup_elements(answering.ma_agent_id if answering is not None else None)
    elements.append(_button(action_id=ACTION_NEW, label=NEW_AGENT_LABEL))
    elements.append(_button(action_id=ACTION_ROUTING, label=ROUTING_LABEL))
    blocks.append({"type": "actions", "elements": elements})
    blocks.extend(_pager_blocks(page, meta=meta))
    return finish_modal(
        title="Agents",
        blocks=blocks,
        private_metadata=encode_panel_metadata(
            PanelMetadata(
                team_id=meta.team_id,
                channel_id=channel_id,
                view="agents",
                page=page.page,
                root_view_id=meta.root_view_id,
            )
        ),
        callback_id=CALLBACK_AGENTS,
    )


def _roster_row_facts(
    agent: RosterAgent,
    *,
    is_answering: bool,
    attributions: Mapping[uuid.UUID, str],
    routed_agent_names: Collection[str] | None,
) -> list[str]:
    """The context line under one roster row — only facts we actually hold."""
    facts: list[str] = []
    if agent.is_built_in:
        facts.append("built in")
    mention = (
        attributions.get(agent.created_by_account_id)
        if agent.created_by_account_id is not None
        else None
    )
    if mention is not None:
        facts.append(f"made by {mention}")
    if is_answering:
        tier = agent.answering_tier
        if tier is not None:
            facts.append(_TIER_PHRASES[tier])
    elif routed_agent_names is not None:
        facts.append(
            "answers in other channels" if agent.name in routed_agent_names else UNROUTED_LINE
        )
    return facts


# ---------------------------------------------------------------------------
# Details
# ---------------------------------------------------------------------------


def build_details_view(
    details: AgentDetails,
    *,
    meta: PanelMetadata,
    is_admin: bool,
    coding_tools_available: bool,
    channel_id: str,
    attribution: str | None,
) -> dict[str, Any]:
    """One agent's whole readable state, in the order the roster promised.

    `is_admin` reaches this view only through `details.unrouted_note`, which
    the core already phrased in the reader's voice; nothing here is shown or
    hidden by role. Key values are not a parameter and cannot be: the model
    carries names and attribution only.
    """
    _ = is_admin
    title = fit_title(details.name)
    blocks: list[dict[str, Any]] = []
    if title != details.name:
        blocks.append(_section(f"*{escape_mrkdwn(details.name)}*"))
    header = _details_header(details, attribution=attribution)
    if header:
        blocks.append(_context(*header))
    if details.purpose:
        blocks.append(_section(escape_mrkdwn(details.purpose)))
    blocks.append(
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": _clip(f"*Model*\n{escape_mrkdwn(details.model_display_name)}"),
                },
                {"type": "mrkdwn", "text": _clip(_skills_text(details))},
            ],
        }
    )
    blocks.extend(_keys_blocks(details, meta=meta))
    blocks.extend(_repo_blocks(details))
    blocks.append(_section(_mcp_servers_text(details)))
    if details.unrouted_note is not None:
        blocks.append(_section(escape_mrkdwn_preserving_mentions(details.unrouted_note)))
    blocks.append(
        _context(CODING_TOOLS_NOTE if coding_tools_available else CODING_TOOLS_UNAVAILABLE_NOTE)
    )
    blocks.append({"type": "divider"})
    elements = _setup_elements(details.ma_agent_id)
    if coding_tools_available:
        elements.append(
            _button(action_id=ACTION_CODING_TOOLS, label=CODING_TOOLS_LABEL, value=details.name)
        )
    blocks.append({"type": "actions", "elements": elements})
    return finish_modal(
        title=title,
        blocks=blocks,
        private_metadata=encode_panel_metadata(
            PanelMetadata(
                team_id=meta.team_id,
                channel_id=channel_id,
                view="details",
                agent_name=details.name,
                root_view_id=meta.root_view_id,
                expanded=meta.expanded,
            )
        ),
        callback_id=CALLBACK_DETAILS,
    )


def _details_header(details: AgentDetails, *, attribution: str | None) -> list[str]:
    parts: list[str] = []
    if attribution is not None:
        parts.append(f"made by {attribution}")
    labels = _place_labels(details.answers_in)
    parts.append(f"answers in {', '.join(labels)}" if labels else UNROUTED_LINE)
    return parts


def _skills_text(details: AgentDetails) -> str:
    if not details.skills:
        return "🧩 *Skills*\n_no skills yet_"
    names = "\n".join(escape_mrkdwn(skill.title or skill.skill_id) for skill in details.skills)
    if details.skills_listing_truncated:
        names = f"{names}\n_some skill names may be missing_"
    return f"🧩 *Skills*\n{names}"


def _keys_blocks(details: AgentDetails, *, meta: PanelMetadata) -> list[dict[str, Any]]:
    """Key names, collapsed past `KEYS_COLLAPSED_COUNT`, never key values."""
    if not details.keys:
        return [_section("🔑 *Keys*\n_no keys yet_")]
    names = [escape_mrkdwn(key.name) for key in details.keys]
    expanded = "keys" in meta.expanded
    accessory: dict[str, Any] | None = None
    if expanded:
        shown = names[:KEYS_EXPANDED_CAP]
        remainder = len(names) - len(shown)
        if remainder > 0:
            shown = [*shown, f"+{remainder} more"]
        if len(names) > KEYS_COLLAPSED_COUNT:
            accessory = _button(action_id=ACTION_EXPAND_KEYS, label="Show fewer")
    else:
        shown = names[:KEYS_COLLAPSED_COUNT]
        if len(names) > KEYS_COLLAPSED_COUNT:
            accessory = _button(action_id=ACTION_EXPAND_KEYS, label="Show all")
    listing = "\n".join(shown)
    return [
        _section(f"🔑 *Keys*\n{listing}", accessory=accessory),
        _context(shared_keys_sentence(escape_mrkdwn(details.name))),
    ]


def _repo_blocks(details: AgentDetails) -> list[dict[str, Any]]:
    repo = details.repo
    if repo is None:
        return [_section("📦 *Working repo*\n_no working repo yet_")]
    owner_repo = normalize_owner_repo(repo.repo_url)
    link = f"<https://github.com/{owner_repo}|{escape_mrkdwn(owner_repo)}>"
    branch = f" `{escape_mrkdwn(repo.default_branch)}`" if repo.default_branch else ""
    return [
        _section(f"📦 *Working repo*\n{link}{branch}"),
        _context(_repo_access_line(repo.access)),
    ]


def _repo_access_line(access: RepoAccess) -> str:
    """Say exactly what was recorded about this repo, and nothing more."""
    if access.kind == "needs_attention":
        corrective = access.corrective or "nothing would authorize a clone right now."
        return f"⚠️ needs attention: {escape_mrkdwn(corrective)}"
    if access.kind == "not_checked":
        return "not checked yet"
    if access.credential == "per_agent_token":
        lead = "via token"
    elif access.credential == "deployment_public":
        lead = "public repo"
    else:
        lead = "via the GitHub App"
    if access.checked_at is None:
        return lead
    return f"{lead} · last checked {_slack_date(access.checked_at)}"


def _mcp_servers_text(details: AgentDetails) -> str:
    if not details.mcp_servers:
        return "🔌 *MCP servers*\n_no servers yet_"
    listing = "\n".join(
        f"{escape_mrkdwn(server.name)} · {escape_mrkdwn(server.url)}"
        for server in details.mcp_servers
    )
    return f"🔌 *MCP servers*\n{listing}"


# ---------------------------------------------------------------------------
# Who answers where
# ---------------------------------------------------------------------------


def build_routing_view(
    answering_map: AnsweringMap,
    *,
    page: Page[ChannelAnswer],
    meta: PanelMetadata,
    is_admin: bool,
    attributions: Mapping[uuid.UUID, str],
    setup_links: Sequence[str],
    channel_id: str,
    unrouted_agent_name: str | None,
) -> dict[str, Any]:
    """The whole cascade, laid out so the precedence is visible, not inferred.

    No setup button: this view answers where mentions go, and the change it
    describes is a sentence to say to Daimon rather than a control here.
    """
    blocks: list[dict[str, Any]] = []
    if page.items:
        for answer in page.items:
            blocks.append(
                _section(f"<#{answer.channel_id}> → *{escape_mrkdwn(answer.agent_name)}*")
            )
            audit = _audit_parts(
                account_id=answer.set_by_account_id,
                moment=answer.set_at,
                attributions=attributions,
            )
            if audit:
                blocks.append(_context(" · ".join(audit)))
    else:
        blocks.append(_section("_no channel has its own setting_"))
    tenant_default = answering_map.tenant_default
    if tenant_default is not None:
        blocks.append(
            _section(f"🌐 Workspace default: *{escape_mrkdwn(tenant_default.agent_name)}*")
        )
        audit = _audit_parts(
            account_id=tenant_default.set_by_account_id,
            moment=tenant_default.set_at,
            attributions=attributions,
        )
        if audit:
            blocks.append(_context(" · ".join(audit)))
    else:
        blocks.append(_section("🌐 _no workspace default_"))
    if answering_map.deployment_default is not None:
        line = f"Deployment default: *{escape_mrkdwn(answering_map.deployment_default)}*"
        if answering_map.tenant_consumes_fallthrough:
            line = f"{line}\n_not in effect while a workspace default is set_"
        blocks.append(_section(line))
    else:
        blocks.append(_section("_no deployment default_"))
    blocks.append({"type": "divider"})
    links = list(setup_links[:MAX_SETUP_LINKS])
    listing = "\n".join(links) if links else "_none open_"
    blocks.append(_section(f"*Setup conversations*\n{listing}"))
    blocks.append({"type": "divider"})
    blocks.append(
        _context(
            _routing_request_line(
                answering_map,
                channel_id=channel_id,
                is_admin=is_admin,
                unrouted_agent_name=unrouted_agent_name,
            )
        )
    )
    blocks.extend(_pager_blocks(page, meta=meta))
    return finish_modal(
        title="Who answers where",
        blocks=blocks,
        private_metadata=encode_panel_metadata(
            PanelMetadata(
                team_id=meta.team_id,
                channel_id=channel_id,
                view="routing",
                page=page.page,
                agent_name=unrouted_agent_name,
                root_view_id=meta.root_view_id,
            )
        ),
        callback_id=CALLBACK_ROUTING,
    )


def _audit_parts(
    *,
    account_id: uuid.UUID | None,
    moment: datetime | None,
    attributions: Mapping[uuid.UUID, str],
) -> list[str]:
    parts: list[str] = []
    mention = attributions.get(account_id) if account_id is not None else None
    if mention is not None:
        parts.append(f"set by {mention}")
    if moment is not None:
        parts.append(_slack_date(moment))
    return parts


def _answering_name(answering_map: AnsweringMap, *, channel_id: str) -> str | None:
    """Which agent this channel's mentions reach today, by the same precedence."""
    for answer in answering_map.channel_overrides:
        if answer.channel_id == channel_id:
            return answer.agent_name
    if answering_map.tenant_default is not None:
        return answering_map.tenant_default.agent_name
    return answering_map.deployment_default


def _routing_request_line(
    answering_map: AnsweringMap,
    *,
    channel_id: str,
    is_admin: bool,
    unrouted_agent_name: str | None,
) -> str:
    agent_name = unrouted_agent_name or _answering_name(answering_map, channel_id=channel_id)
    lead = "Tell Daimon: " if is_admin else "An admin can tell Daimon: "
    request = build_routing_request(
        agent_name=escape_mrkdwn(agent_name or "an agent"), channel_label=f"<#{channel_id}>"
    )
    return f"{PRECEDENCE_LINE} {lead}{request}"


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


def build_creating_view(*, agent_name: str, meta: PanelMetadata) -> dict[str, Any]:
    """The placeholder the New agent form acks into while the create runs."""
    return finish_modal(
        title="New agent",
        blocks=[
            _section(f"Creating *{escape_mrkdwn(agent_name)}*…"),
            _context("This takes a few seconds."),
        ],
        private_metadata=encode_panel_metadata(
            meta.with_view("creating", agent_name=agent_name, root_view_id=meta.root_view_id)
        ),
        callback_id=CALLBACK_CREATING,
    )


def build_new_agent_form(
    *,
    meta: PanelMetadata,
    model_choices: Sequence[ModelChoice],
    initial_name: str | None = None,
    initial_purpose: str | None = None,
    initial_model: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """The one creation shortcut: name, purpose, model, and nothing else.

    The block and action ids are the ones the submission evaluator already
    reads, so a failed create can be restored into this same form with the
    person's inputs intact rather than handing them a blank one.
    """
    options = [
        {
            "text": {"type": "plain_text", "text": choice.label},
            "value": choice.id,
        }
        for choice in model_choices
    ]
    selected_id = initial_model or next(
        (choice.id for choice in model_choices if choice.is_default),
        model_choices[0].id if model_choices else None,
    )
    initial_option = next((option for option in options if option["value"] == selected_id), None)
    model_element: dict[str, Any] = {
        "type": "static_select",
        "action_id": "new_agent__model",
        "options": options,
    }
    if initial_option is not None:
        model_element["initial_option"] = initial_option
    name_element: dict[str, Any] = {
        "type": "plain_text_input",
        "action_id": "new_agent__name",
        "placeholder": {"type": "plain_text", "text": "e.g. my-data-analyst"},
    }
    if initial_name:
        name_element["initial_value"] = initial_name
    purpose_element: dict[str, Any] = {
        "type": "plain_text_input",
        "action_id": "new_agent__prompt",
        "multiline": True,
        "placeholder": {"type": "plain_text", "text": "One or two sentences…"},
    }
    if initial_purpose:
        purpose_element["initial_value"] = initial_purpose
    blocks: list[dict[str, Any]] = []
    if error is not None:
        blocks.append(_section(f"⚠️ {escape_mrkdwn(error)}"))
    blocks.extend(
        [
            {
                "type": "input",
                "block_id": "new_agent__name",
                "label": {"type": "plain_text", "text": "Agent name"},
                "element": name_element,
            },
            {
                "type": "input",
                "block_id": "new_agent__prompt",
                "label": {"type": "plain_text", "text": "What should it help with?"},
                "optional": True,
                "element": purpose_element,
            },
            {
                "type": "input",
                "block_id": "new_agent__model",
                "label": {"type": "plain_text", "text": "Model"},
                "element": model_element,
            },
        ]
    )
    return finish_modal(
        title="New agent",
        blocks=blocks,
        private_metadata=encode_panel_metadata(
            meta.with_view("new_agent", root_view_id=meta.root_view_id)
        ),
        callback_id=CALLBACK_NEW_AGENT,
        close="Cancel",
        submit="Create",
    )
