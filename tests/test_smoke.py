"""Tests for smoke."""

import pyllm


def test_package_imports():
    """Package imports."""
    assert pyllm.__name__ == "pyllm"
