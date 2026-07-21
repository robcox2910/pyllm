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

    for name in ["random_program", "render", "build_corpus", "is_valid",
                 "parse_rate", "harvest_dir", "PEBBLE_AVAILABLE"]:
        assert hasattr(pebble, name), f"pyllm.pebble is missing {name}"
