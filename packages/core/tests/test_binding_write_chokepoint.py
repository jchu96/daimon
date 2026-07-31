"""Drift gate: no production caller writes a repo binding with an explicitly
absent proof, and exactly one binding-write chokepoint exists.

`set_binding`'s `proof` keyword has no default, so a caller that omits it
entirely is a pyright error. But a type checker cannot catch a caller that
supplies an explicitly absent proof on purpose — that keystroke is valid
Python and passes strict type-checking. The only thing that can catch it is a
scan: a write with no established proof is a write that should not happen in
production code, so this module fails CI the moment one appears anywhere
under the production source trees. Test call sites are exempt — constructing
an unproven binding is a legitimate fixture for exercising the fail-closed
read-side gates elsewhere in the suite.

This module also pins that `set_binding` has exactly one definition under the
production trees, so a parallel write helper cannot quietly reopen a
per-adapter write path and make the chokepoint argument false without ever
touching the guarded function.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PRODUCTION_ROOTS = [
    *sorted(p for p in (_REPO_ROOT / "packages").rglob("daimon") if p.is_dir()),
    _REPO_ROOT / "apps" / "notebook-host" / "src",
]

_ELIDED_PROOF_RE = re.compile(r"proof\s*=\s*None")
_SET_BINDING_DEF_RE = re.compile(r"^async def set_binding\(")


def _production_python_files() -> list[Path]:
    files: list[Path] = []
    for root in _PRODUCTION_ROOTS:
        assert root.is_dir(), f"expected production root to exist: {root}"
        for path in root.rglob("*.py"):
            if "tests" in path.relative_to(root).parts:
                continue
            files.append(path)
    return files


def test_no_production_file_writes_an_explicitly_absent_proof() -> None:
    offenders: list[str] = []
    for path in _production_python_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _ELIDED_PROOF_RE.search(line):
                offenders.append(f"{path}:{lineno}: {line.strip()}")

    assert offenders == [], (
        "a production binding write must state the proof it established at bind "
        "time; an explicitly absent proof means the write should not happen. "
        "Offending line(s):\n" + "\n".join(offenders)
    )


def test_exactly_one_set_binding_definition_exists_in_production() -> None:
    definitions: list[str] = []
    for path in _production_python_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _SET_BINDING_DEF_RE.match(line.strip()):
                definitions.append(f"{path}:{lineno}")

    assert len(definitions) == 1, (
        "exactly one set_binding definition must exist in production code — a "
        f"second write helper would bypass the chokepoint. Found: {definitions}"
    )
