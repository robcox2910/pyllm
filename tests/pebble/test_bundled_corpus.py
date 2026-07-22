from pyllm.data import load_corpus


def test_bundled_pebble_corpus_loads():
    text = load_corpus("pebble")
    assert isinstance(text, str)
    assert len(text) > 2000
    assert "let " in text and "fn " in text


def test_bundled_pebble_corpus_parses_when_pebble_available():
    import pytest

    pytest.importorskip("pebble")
    from pyllm.pebble import parse_rate

    programs = [p for p in load_corpus("pebble").split("\n\n") if p.strip()]
    assert parse_rate(programs) > 0.98  # harvested docs may contain a rare edge case
