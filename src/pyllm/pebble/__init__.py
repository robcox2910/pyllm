"""The Pebble flagship: generate Pebble code, harvest real code, grade validity."""

import importlib.util

# True when the pebble-lang oracle is importable (optional build-time dep).
PEBBLE_AVAILABLE = importlib.util.find_spec("pebble") is not None

from pyllm.pebble.corpus import build_corpus  # noqa: E402
from pyllm.pebble.generator import random_program  # noqa: E402
from pyllm.pebble.harvest import harvest_dir  # noqa: E402
from pyllm.pebble.render import render  # noqa: E402
from pyllm.pebble.score import is_valid, parse_rate  # noqa: E402

__all__ = [
    "PEBBLE_AVAILABLE",
    "random_program",
    "render",
    "build_corpus",
    "is_valid",
    "parse_rate",
    "harvest_dir",
]
