"""Dig real, hand-written Pebble out of the documentation.

There's no big pile of Pebble code on the internet, but there are hundreds of
little examples tucked inside Pebble's own docs. This scoops out every fenced
```pebble code block so we can mix genuine, idiomatic snippets into our
home-grown corpus. (Some doc snippets are deliberately broken to show errors, so
callers filter these through the validity scorer before keeping them.)
"""

import re
from os import PathLike
from pathlib import Path

_FENCE = re.compile(r"```pebble\n(.*?)```", re.DOTALL)


def harvest_text(markdown: str) -> list[str]:
    """Return every fenced ```pebble block found in a markdown string."""
    return [match.group(1).rstrip("\n") for match in _FENCE.finditer(markdown)]


def harvest_dir(path: str | PathLike[str]) -> list[str]:
    """Return every ```pebble block across all *.md files under `path` (sorted).

    A folder that isn't there raises `FileNotFoundError` rather than quietly
    finding nothing -- otherwise a typo'd path would silently shrink the corpus.
    """
    folder = Path(path)
    if not folder.is_dir():
        msg = f"no docs folder to harvest at {folder}"
        raise FileNotFoundError(msg)
    blocks: list[str] = []
    for md in sorted(folder.rglob("*.md")):
        blocks.extend(harvest_text(md.read_text(encoding="utf-8")))
    return blocks
