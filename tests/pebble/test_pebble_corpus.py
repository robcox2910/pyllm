import numpy as np

from pyllm.pebble.corpus import build_corpus


def test_build_corpus_generates_programs_without_pebble():
    # Generation must NOT require pebble; harvest_paths empty => no pebble needed.
    text = build_corpus(np.random.default_rng(0), num_generated=5)
    assert isinstance(text, str)
    assert text.count("let ") >= 5  # every program declares variables
    programs = [p for p in text.split("\n\n") if p.strip()]
    assert len(programs) == 5


def test_build_corpus_is_deterministic():
    a = build_corpus(np.random.default_rng(1), num_generated=4)
    b = build_corpus(np.random.default_rng(1), num_generated=4)
    assert a == b


def test_public_api_exports():
    import pyllm.pebble as pebble

    for name in [
        "random_program",
        "render",
        "build_corpus",
        "is_valid",
        "parse_rate",
        "harvest_dir",
        "viable",
        "complete",
        "PEBBLE_AVAILABLE",
    ]:
        assert hasattr(pebble, name), f"pyllm.pebble is missing {name}"


def test_canonical_corpus_drops_snippets_the_formatter_breaks(tmp_path):
    import numpy as np
    import pytest

    pytest.importorskip("pebble")
    from pyllm.pebble import build_corpus, is_valid

    (tmp_path / "doc.md").write_text("```pebble\nstruct Point { x, y }\n```\n")
    text = build_corpus(
        np.random.default_rng(0),
        num_generated=3,
        harvest_paths=[tmp_path],
        canonical=True,
    )
    programs = [p for p in text.split("\n\n") if p.strip()]
    assert "struct" not in text
    assert len(programs) == 3
    assert all(is_valid(p) for p in programs)


def test_every_bundled_corpus_program_parses():
    import pytest

    pytest.importorskip("pebble")
    from pyllm.data import load_corpus
    from pyllm.pebble import is_valid

    programs = [p for p in load_corpus("pebble").split("\n\n") if p.strip()]
    broken = [p for p in programs if not is_valid(p)]
    assert not broken, broken[0]
