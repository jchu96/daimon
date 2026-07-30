"""Answer serialization for the resumed turn and for rendered screens.

`format_answer_block` and `summarize_answers` deliberately disagree on
escaping and on values-vs-labels, because they serve different readers.

`format_answer_block` is what the resumed turn's user message carries. The
agent that authored the `WizardSpec` already knows the question text and the
option labels — restating them as prose would make the agent re-derive
information it already has. So this emits raw, unescaped option *values*
(the machine identifiers the agent itself assigned), one JSON array per step
key inside a fenced code block. JSON-encoding each step's value list is what
makes a value containing a colon, a newline or a bracket unambiguous to
parse back out — the agent does not have to guess where one answer ends and
the next begins.

`summarize_answers` is what a human reads on the review and submitted
screens. It substitutes each recorded value back to its option's label
(falling back to the raw value for user-typed custom text, which has no
label), and every substituted string passes through
`escape_discord_markup` — display text a user only partially controls
(their own typed answers) must never be interpreted as markdown or as a
mass-mention by Discord's renderer.
"""

from __future__ import annotations

import json

from daimon.core.wizard.spec import WizardSpec
from daimon.core.wizard.state import WizardState

_MARKDOWN_CONTROL_CHARS = "*_~`|>#-[]()"
_UNANSWERED_DISPLAY = "—"  # em dash


_ZERO_WIDTH_SPACE = "​"


def escape_discord_markup(text: str) -> str:
    """Backslash-escape Discord markdown control characters and neutralize
    mass mentions (`@everyone`, `@here`, `<@id>`, `<@&role_id>`) with a
    zero-width space inserted right after the `@`.

    Mass-mention neutralization runs before markdown escaping so the `<@`
    prefix `<@&role_id>` also starts with is matched exactly once, regardless
    of how many of `*_~`|>#-[]()` the surrounding text also contains.
    """
    neutralized = text.replace("@everyone", f"@{_ZERO_WIDTH_SPACE}everyone")
    neutralized = neutralized.replace("@here", f"@{_ZERO_WIDTH_SPACE}here")
    neutralized = neutralized.replace("<@", f"<@{_ZERO_WIDTH_SPACE}")

    escaped_chars: list[str] = []
    for char in neutralized:
        if char in _MARKDOWN_CONTROL_CHARS:
            escaped_chars.append(f"\\{char}")
        else:
            escaped_chars.append(char)
    return "".join(escaped_chars)


def _label_for_value(spec_step_options: list[tuple[str, str]], value: str) -> str:
    for option_value, option_label in spec_step_options:
        if option_value == value:
            return option_label
    return value


def summarize_answers(spec: WizardSpec, state: WizardState) -> str:
    """Return a human-readable, escaped, one-line-per-step summary."""
    lines: list[str] = []
    for step in spec.steps:
        recorded = state.answers.get(step.key)
        if not recorded:
            lines.append(f"**{step.question}**: {_UNANSWERED_DISPLAY}")
            continue
        option_pairs = [(option.value, option.label) for option in step.options]
        display_values = [
            escape_discord_markup(_label_for_value(option_pairs, value)) for value in recorded
        ]
        lines.append(f"**{step.question}**: {', '.join(display_values)}")
    return "\n".join(lines)


def format_answer_block(spec: WizardSpec, state: WizardState) -> str:
    """Return the fenced answer block the resumed turn's user message carries."""
    header = f"Wizard answers ({spec.prompt}):"
    body_lines = [
        f"{step.key}: {json.dumps(state.answers.get(step.key, []))}" for step in spec.steps
    ]
    body = "\n".join(body_lines)
    return f"{header}\n```\n{body}\n```"
