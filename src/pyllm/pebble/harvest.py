"""Dig real, hand-written Pebble out of the documentation.

There's no big pile of Pebble code on the internet, but there are hundreds of
little examples tucked inside Pebble's own docs. This scoops out every fenced
```pebble code block so we can mix genuine, idiomatic snippets into our
home-grown corpus. (Some doc snippets are deliberately broken to show errors, so
callers filter these through the validity scorer before keeping them.)
"""

import re
from pathlib import Path

_FENCE = re.compile(r"```pebble\n(.*?)```", re.DOTALL)


def harvest_text(markdown):
    """Return every fenced ```pebble block found in a markdown string."""
    return [match.group(1).rstrip("\n") for match in _FENCE.finditer(markdown)]


def harvest_dir(path):
    """Return every ```pebble block across all *.md files under `path` (sorted)."""
    blocks = []
    for md in sorted(Path(path).rglob("*.md")):
        blocks.extend(harvest_text(md.read_text(encoding="utf-8")))
    return blocks
