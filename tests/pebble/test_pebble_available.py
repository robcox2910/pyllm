"""Tests for pebble available."""


def test_pebble_is_importable_in_dev_env():
    """Pebble is importable in the dev env, where the optional `pebble` extra is installed.

    The pebble-lang oracle must be importable there; the test is skipped elsewhere.
    """
    import importlib.util

    if importlib.util.find_spec("pebble") is None:
        import pytest

        pytest.skip("pebble-lang not installed (optional build-time dependency)")
    for module, name in [
        ("pebble.formatter", "Formatter"),
        ("pebble.lexer", "Lexer"),
        ("pebble.parser", "Parser"),
    ]:
        assert hasattr(importlib.import_module(module), name)
