"""Build-time: generate + harvest the bundled Pebble corpus.

Run once with `uv run python scripts/build_pebble_corpus.py`; the resulting
`src/pyllm/data/pebble_corpus.txt` is committed.

By default it mines the ```pebble snippets in a sibling `../pebble-lang/docs`
checkout. Point it somewhere else with `--pebble-docs PATH`. If that folder is
missing the build stops with an error rather than quietly leaving the real
snippets out; pass `--no-harvest` to build from generated programs only.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pyllm.data import CORPUS_DIR
from pyllm.pebble import build_corpus, parse_rate

_PEBBLE_DOCS = Path(__file__).resolve().parents[2] / "pebble-lang" / "docs"


def _say(message: str) -> None:
    """Show a message on the screen."""
    sys.stdout.write(message + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="Build the bundled Pebble corpus.")
    parser.add_argument(
        "--pebble-docs",
        type=Path,
        default=_PEBBLE_DOCS,
        help=f"pebble-lang docs folder to harvest snippets from (default: {_PEBBLE_DOCS})",
    )
    parser.add_argument(
        "--no-harvest",
        dest="harvest",
        action="store_false",
        help="skip harvesting doc snippets and use generated programs only",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CORPUS_DIR / "pebble_corpus.txt",
        help="where to write the corpus (default: the bundled pebble_corpus.txt)",
    )
    return parser.parse_args(argv)


def harvest_paths(args: argparse.Namespace) -> list[Path]:
    """Return the docs folders to mine: none with --no-harvest, else the checked one."""
    if not args.harvest:
        return []
    return [args.pebble_docs]


def main(argv: Sequence[str] | None = None) -> None:
    """Run the build step and report what happened."""
    args = parse_args(argv)
    if args.harvest and not args.pebble_docs.is_dir():
        sys.stderr.write(
            f"error: no pebble-lang docs folder at {args.pebble_docs}\n"
            "Check out pebble-lang next to pyllm, pass --pebble-docs PATH, "
            "or pass --no-harvest to build without the doc snippets.\n"
        )
        sys.exit(2)
    rng = np.random.default_rng(2024)
    text = build_corpus(rng, num_generated=600, harvest_paths=harvest_paths(args), canonical=True)
    programs = [p for p in text.split("\n\n") if p.strip()]
    rate = parse_rate(programs)
    _say(f"{len(programs)} programs, parse rate {rate:.3f}")
    args.out.write_text(text, encoding="utf-8")
    _say(f"wrote {args.out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
