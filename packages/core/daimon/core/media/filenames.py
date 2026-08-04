"""Pure filename derivation for files delivered as chat attachments.

No I/O. The extension matters because chat clients pick their preview from it,
so it is derived from the declared mime type rather than trusted from the
agent-supplied title.
"""

from __future__ import annotations

import mimetypes
import re

_TITLE_SLUG = re.compile(r"[^A-Za-z0-9._-]+")

# Used when mimetypes cannot decide — better an opaque extension than a wrong one.
_FALLBACK_EXT = ".bin"


def extension_for_mime(mime_type: str) -> str:
    """Return the file extension a chat client should see for ``mime_type``."""
    ext = mimetypes.guess_extension(mime_type, strict=False)
    if ext in (None, ""):
        # mimetypes has no mapping for image/jpeg on some platforms, and ".jpe"
        # on others; neither is what a user expects to download.
        if mime_type == "image/jpeg":
            return ".jpg"
        return _FALLBACK_EXT
    if ext == ".jpe":
        return ".jpg"
    return ext


def sanitize_title(title: str) -> str:
    """Reduce an agent-supplied title to a safe filename stem.

    Everything outside ``[A-Za-z0-9._-]`` collapses to an underscore, which is
    also what stops a title from carrying a path separator.
    """
    cleaned = _TITLE_SLUG.sub("_", title).strip("._-")
    return cleaned or "file"


def display_filename_for(title: str, mime_type: str) -> str:
    """Build the user-visible filename for an agent-supplied title.

    The extension is appended only when the title does not already end in it —
    an agent that helpfully passes "chart.jpg" should not get "chart.jpg.jpg".
    """
    cleaned = sanitize_title(title)
    ext = extension_for_mime(mime_type)
    if cleaned.lower().endswith(ext.lower()):
        return cleaned
    return f"{cleaned}{ext}"
