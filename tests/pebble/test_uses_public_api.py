"""PyLLM talks to Pebble only through Pebble's public front door."""

import re
from pathlib import Path

SRC = Path(__file__).parent.parent.parent / "src" / "pyllm"
DEEP_IMPORT = re.compile(r"^\s*(from pebble\.|import pebble\.)", re.MULTILINE)


def test_pyllm_imports_only_pebbles_public_api():
    offenders = [
        str(path.relative_to(SRC))
        for path in sorted(SRC.rglob("*.py"))
        if DEEP_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
