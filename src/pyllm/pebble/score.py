"""Grade Pebble text by asking the real Pebble parser whether it parses.

Because we own the Pebble toolchain, we have a perfect marker: hand a program
to Pebble's own `check` (its lexer and parser) and see if it complains. Almost
no AI project can objectively score its own output; we can. `is_valid` marks
one program; `parse_rate` marks a pile of them and reports the percentage that
pass.
"""

from collections.abc import Callable, Iterable


def _require_pebble() -> tuple[Callable[[str], None], type[Exception]]:
    """Import the Pebble oracle lazily, with a friendly error if it's missing."""
    try:
        from pebble import PebbleError, check  # noqa: PLC0415 - optional dependency
    except ImportError as exc:  # pragma: no cover - environment-dependent
        msg = (
            "Live Pebble scoring needs pebble-lang. Install the optional extra: "
            "`uv sync --all-extras`."
        )
        raise RuntimeError(msg) from exc
    return check, PebbleError


def is_valid(source: str) -> bool:
    """Say whether `source` is syntactically valid Pebble (it lexes and parses)."""
    check, pebble_error = _require_pebble()
    try:
        check(source)
    except pebble_error:
        return False
    return True


def parse_rate(sources: Iterable[str]) -> float:
    """Return the fraction of the given programs that parse (0.0 if there are none)."""
    programs = list(sources)
    if not programs:
        return 0.0
    return sum(is_valid(s) for s in programs) / len(programs)
