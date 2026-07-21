"""The Pebble flagship: generate Pebble code, harvest real code, grade validity."""

import importlib.util

# True when the pebble-lang oracle is importable (optional build-time dep).
PEBBLE_AVAILABLE = importlib.util.find_spec("pebble") is not None
