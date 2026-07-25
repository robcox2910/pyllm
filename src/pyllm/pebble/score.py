"""Grade Pebble text by asking the real Pebble parser: "does this parse?"

Because we own the Pebble toolchain, we have a perfect marker: feed a program to
Pebble's own lexer and parser and see if it complains. Almost no AI project can
objectively score its own output; we can. `is_valid` marks one program;
`parse_rate` marks a pile of them and reports the percentage that pass.
"""


def _require_pebble():
    """Import the Pebble oracle lazily, with a friendly error if it's missing."""
    try:
        from pebble.errors import PebbleError
        from pebble.lexer import Lexer
        from pebble.parser import Parser
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Live Pebble scoring needs pebble-lang. Install the optional extra: "
            "`uv sync --all-extras` (uses ../pebble-lang)."
        ) from exc
    return Lexer, Parser, PebbleError


def is_valid(source):
    """True if `source` is syntactically valid Pebble (it lexes and parses)."""
    Lexer, Parser, PebbleError = _require_pebble()
    try:
        Parser(Lexer(source).tokenize()).parse()
    except PebbleError:
        return False
    return True


def parse_rate(sources):
    """The fraction of the given programs that parse (0.0 if there are none)."""
    sources = list(sources)
    if not sources:
        return 0.0
    return sum(is_valid(s) for s in sources) / len(sources)
