"""Assemble a training corpus: mostly home-grown programs, seasoned with real ones.

This is the "grow your own data" recipe in code: generate a big batch of
guaranteed-valid programs, optionally fold in genuine snippets harvested from the
docs (keeping only the ones that actually parse), and glue them into one text
file the model can learn from.
"""

from pyllm.pebble import PEBBLE_AVAILABLE
from pyllm.pebble.generator import random_program
from pyllm.pebble.harvest import harvest_dir
from pyllm.pebble.render import render


def canonicalize(source):
    """Tidy Pebble source into its one true formatting, via Pebble's formatter."""
    try:
        from pebble.formatter import Formatter
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Canonicalizing needs pebble-lang. Install it: `uv sync --all-extras`."
        ) from exc
    return Formatter(source).format()


def build_corpus(rng, num_generated=400, harvest_paths=(), canonical=False):
    """Build a Pebble training corpus as one big string (blank-line-separated).

    - `num_generated`: how many random valid programs to grow.
    - `harvest_paths`: directories of markdown to mine for real Pebble snippets
      (kept only if they parse; skipped entirely when pebble-lang is absent).
    - `canonical`: if True, run every program through Pebble's formatter.
    """
    programs = [render(random_program(rng)) for _ in range(num_generated)]
    if harvest_paths and PEBBLE_AVAILABLE:
        from pyllm.pebble.score import is_valid

        for path in harvest_paths:
            programs.extend(block for block in harvest_dir(path) if is_valid(block))
    if canonical:
        programs = [canonicalize(p) for p in programs]
    return "\n\n".join(p.strip() for p in programs) + "\n"
