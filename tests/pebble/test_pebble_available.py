def test_pebble_is_importable_in_dev_env():
    # In the dev/build environment (with the optional `pebble` extra installed),
    # the pebble-lang oracle must be importable. Skipped elsewhere.
    import importlib.util

    if importlib.util.find_spec("pebble") is None:
        import pytest

        pytest.skip("pebble-lang not installed (optional build-time dependency)")
    from pebble.formatter import Formatter  # noqa: F401
    from pebble.lexer import Lexer  # noqa: F401
    from pebble.parser import Parser  # noqa: F401
