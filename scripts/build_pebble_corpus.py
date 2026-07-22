"""Build-time: generate + harvest the bundled Pebble corpus.

Run once with `uv run python scripts/build_pebble_corpus.py`; the resulting
`src/pyllm/data/pebble_corpus.txt` is committed.
"""

from pathlib import Path

import numpy as np

from pyllm.data import CORPUS_DIR
from pyllm.pebble import build_corpus
from pyllm.pebble import parse_rate

_PEBBLE_DOCS = Path(__file__).resolve().parents[2] / "pebble-lang" / "docs"


def main():
    rng = np.random.default_rng(2024)
    harvest = [_PEBBLE_DOCS] if _PEBBLE_DOCS.exists() else []
    text = build_corpus(rng, num_generated=600, harvest_paths=harvest,
                        canonical=True)
    programs = [p for p in text.split("\n\n") if p.strip()]
    rate = parse_rate(programs)
    print(f"{len(programs)} programs, parse rate {rate:.3f}")
    out = CORPUS_DIR / "pebble_corpus.txt"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
