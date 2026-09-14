"""Literal `.env` parsing and serialization. Pure: no I/O, no clock, no state.

This module is the one place that decides what an uploaded `.env` file means.
The grammar is a deliberately small subset of the informal dotenv format, and
the docstring below *is* the specification — there is no other authority.

Grammar
-------
- The file is split on ``"\\n"``; one trailing ``"\\r"`` per line is stripped, so
  CRLF files parse identically to LF files.
- Blank and whitespace-only lines are ignored. A line whose first non-space
  character is ``#`` is a comment and is ignored.
- An entry line may start with the literal word ``export`` followed by one or
  more spaces or tabs; the prefix is dropped.
- The name runs up to the first ``=`` and must fullmatch ``ENV_NAME_PATTERN``.
  Whitespace on either side of the ``=`` is ignored.
- Values take one of three forms:

  - ``'…'`` — single-quoted. Everything up to the next ``'`` is literal; there
    are no escapes, so a single quote cannot appear inside.
  - ``"…"`` — double-quoted. ``\\\\``, ``\\"``, ``\\n``, ``\\r`` and ``\\t`` are
    recognised escapes; any other ``\\x`` is kept as a literal backslash
    followed by ``x``.
  - unquoted — everything to the end of the line, with trailing whitespace
    stripped.

- A quoted value must close on the same line; there are no multi-line values.
- Nothing but whitespace may follow a closing quote.

Two deliberate departures from other dotenv readers
---------------------------------------------------
1. **An unescaped ``#`` inside an unquoted value is literal, not a comment.**
   Most readers strip from ``#`` to end of line. Secrets contain ``#``, and a
   reader that strips mid-line silently truncates a working token into a
   broken one that fails much later, somewhere else. Only a whole line that
   starts with ``#`` is a comment here.
2. **No interpolation and no command substitution.** ``$VAR``, ``${VAR}``,
   ``$(…)`` and backticks are stored as the literal characters typed.

Rejection is whole-file
-----------------------
`parse_env_file` reads every line, collects the problems it finds, and then
raises a single `EnvFileRejected`: a partially-applied secrets file is worse
than none. `EnvProblem` carries a line number and, when the name is known to
be valid, that name — **there is no field for a value**, by type, so no
rejection path can leak one into a log or a chat message.

Collisions with keys already stored are not this module's business; the caller
compares parsed names against what it holds.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final, Literal

from daimon.core.errors import DaimonError
from pydantic import BaseModel, ConfigDict

__all__ = [
    "ENV_NAME_PATTERN",
    "MAX_ENV_FILE_BYTES",
    "MAX_ENV_FILE_ENTRIES",
    "MAX_ENV_VALUE_BYTES",
    "EnvEntry",
    "EnvFileRejected",
    "EnvProblem",
    "EnvRejection",
    "decode_env_bytes",
    "parse_env_file",
    "serialize_env_file",
    "serialize_env_line",
]

MAX_ENV_FILE_BYTES: Final[int] = 64 * 1024
MAX_ENV_FILE_ENTRIES: Final[int] = 200
MAX_ENV_VALUE_BYTES: Final[int] = 4096
ENV_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

EnvRejection = Literal[
    "file_too_large",
    "not_utf8",
    "syntax",
    "bad_name",
    "duplicate_name",
    "value_too_large",
    "too_many_entries",
    "empty",
]

#: Rejections are reported one kind at a time, most structural first: a file
#: whose lines do not parse has no meaningful duplicate or size report to give.
_REJECTION_PRIORITY: Final[tuple[EnvRejection, ...]] = (
    "syntax",
    "bad_name",
    "duplicate_name",
    "value_too_large",
    "too_many_entries",
)

_EXPORT_PREFIX: Final[re.Pattern[str]] = re.compile(r"export[ \t]+")
_DOUBLE_QUOTE_ESCAPES: Final[dict[str, str]] = {
    "\\": "\\",
    '"': '"',
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
#: Characters that force a serialized value into double quotes wherever they
#: appear. Leading/trailing whitespace and an empty value force quoting too.
_MUST_QUOTE_CHARS: Final[frozenset[str]] = frozenset("\n\r\"'\\")


class EnvProblem(BaseModel):
    """One rejected line. Carries no value — that is the point of the type."""

    model_config = ConfigDict(frozen=True)

    name: str | None
    line: int


class EnvEntry(BaseModel):
    """One accepted `NAME=value` entry and the line it came from."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: str
    line: int


class EnvFileRejected(DaimonError):
    """The whole uploaded file was rejected; nothing in it was applied.

    `rejection` says why, `problems` names the offending lines. Neither this
    exception's `str()` nor any problem it carries contains a value.
    """

    def __init__(self, rejection: EnvRejection, problems: Sequence[EnvProblem] = ()) -> None:
        super().__init__(rejection)
        self.rejection: EnvRejection = rejection
        self.problems: tuple[EnvProblem, ...] = tuple(problems)

    def __str__(self) -> str:
        if not self.problems:
            return self.rejection
        lines = ", ".join(str(problem.line) for problem in self.problems)
        return f"{self.rejection} (lines {lines})"


def decode_env_bytes(raw: bytes) -> str:
    """Decode uploaded bytes to text, enforcing the size cap first.

    The size and UTF-8 boundaries live here alone so no caller can parse text
    that was never checked. Raises `EnvFileRejected` with `"file_too_large"`
    or `"not_utf8"`.
    """
    if len(raw) > MAX_ENV_FILE_BYTES:
        raise EnvFileRejected("file_too_large")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as err:
        raise EnvFileRejected("not_utf8") from err


def _parse_double_quoted(body: str) -> str | None:
    """Parse a `"…"` value, returning None when the line is malformed."""
    out: list[str] = []
    index = 1
    while index < len(body):
        char = body[index]
        if char == '"':
            return "".join(out) if not body[index + 1 :].strip() else None
        if char == "\\" and index + 1 < len(body):
            following = body[index + 1]
            out.append(_DOUBLE_QUOTE_ESCAPES.get(following, "\\" + following))
            index += 2
            continue
        out.append(char)
        index += 1
    return None


def _parse_single_quoted(body: str) -> str | None:
    """Parse a `'…'` value, returning None when the line is malformed."""
    close = body.find("'", 1)
    if close == -1 or body[close + 1 :].strip():
        return None
    return body[1:close]


def _parse_value(body: str) -> str | None:
    """Parse the right-hand side of an entry line; None means a syntax error."""
    if body.startswith('"'):
        return _parse_double_quoted(body)
    if body.startswith("'"):
        return _parse_single_quoted(body)
    return body


def parse_env_file(text: str) -> tuple[EnvEntry, ...]:
    """Parse the documented subset, or reject the whole file.

    Every line is parsed before anything is raised, so the person gets the
    full picture in one pass rather than one error per upload. Raises
    `EnvFileRejected`; returns at least one entry when it returns.
    """
    entries: list[EnvEntry] = []
    problems: dict[EnvRejection, list[EnvProblem]] = {kind: [] for kind in _REJECTION_PRIORITY}
    lines_by_name: dict[str, list[int]] = {}

    for number, raw_line in enumerate(text.split("\n"), start=1):
        line = raw_line.removesuffix("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        content = line.strip()
        export_match = _EXPORT_PREFIX.match(content)
        if export_match is not None:
            content = content[export_match.end() :]
        name_part, separator, value_part = content.partition("=")
        if not separator:
            # No name is reported: the token left of a missing "=" may itself be
            # a pasted secret (base64 padding makes this less theoretical).
            problems["syntax"].append(EnvProblem(name=None, line=number))
            continue
        name = name_part.strip()
        if ENV_NAME_PATTERN.fullmatch(name) is None:
            problems["bad_name"].append(EnvProblem(name=None, line=number))
            continue
        value = _parse_value(value_part.strip())
        if value is None:
            problems["syntax"].append(EnvProblem(name=name, line=number))
            continue
        if len(value.encode()) > MAX_ENV_VALUE_BYTES:
            problems["value_too_large"].append(EnvProblem(name=name, line=number))
            continue
        lines_by_name.setdefault(name, []).append(number)
        entries.append(EnvEntry(name=name, value=value, line=number))

    for name, numbers in lines_by_name.items():
        if len(numbers) > 1:
            problems["duplicate_name"].extend(EnvProblem(name=name, line=n) for n in numbers)

    if len(entries) > MAX_ENV_FILE_ENTRIES:
        problems["too_many_entries"].extend(
            EnvProblem(name=entry.name, line=entry.line) for entry in entries[MAX_ENV_FILE_ENTRIES:]
        )

    for kind in _REJECTION_PRIORITY:
        found = problems[kind]
        if found:
            raise EnvFileRejected(kind, sorted(found, key=lambda problem: problem.line))
    if not entries:
        raise EnvFileRejected("empty")
    return tuple(entries)


def serialize_env_line(name: str, value: str) -> str:
    """Render one `NAME=value` line, quoting only when the value needs it.

    Minimal quoting is load-bearing: the assembled bytes are hashed into the
    fingerprint that decides whether a running agent's mounted `.env` is still
    current, so quoting a value that did not need it would change every
    fingerprint at once.
    """
    if value and value == value.strip() and not _MUST_QUOTE_CHARS & set(value):
        return f"{name}={value}"
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'{name}="{escaped}"'


def serialize_env_file(entries: Sequence[tuple[str, str]]) -> bytes:
    """Render `(name, value)` pairs as `.env` bytes with a trailing newline.

    Returns `b""` for no entries; the caller decides what an empty file means.
    """
    if not entries:
        return b""
    body = "\n".join(serialize_env_line(name, value) for name, value in entries)
    return (body + "\n").encode()
