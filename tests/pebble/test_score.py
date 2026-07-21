import numpy as np
import pytest

pytest.importorskip("pebble")  # oracle-backed tests need pebble-lang installed

from pyllm.pebble.generator import random_program  # noqa: E402
from pyllm.pebble.render import render  # noqa: E402
from pyllm.pebble.score import is_valid  # noqa: E402
from pyllm.pebble.score import parse_rate  # noqa: E402


def test_is_valid_accepts_known_good():
    assert is_valid("let x = 1 + 2\nprint(x)\n")


def test_is_valid_rejects_known_bad():
    assert not is_valid("let = = = 1 +\n")


def test_parse_rate_of_empty_is_zero():
    assert parse_rate([]) == 0.0


def test_parse_rate_counts_fraction_valid():
    rate = parse_rate(["let x = 1\n", "@@@ not pebble @@@"])
    assert rate == 0.5


def test_every_generated_program_parses():
    rng = np.random.default_rng(0)
    sources = [render(random_program(rng)) for _ in range(200)]
    assert parse_rate(sources) == 1.0, "generator must emit 100% valid Pebble"
