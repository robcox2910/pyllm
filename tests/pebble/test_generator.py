import numpy as np

from pyllm.pebble.ast import Program
from pyllm.pebble.generator import random_program
from pyllm.pebble.render import render


def test_generator_returns_a_program():
    prog = random_program(np.random.default_rng(0))
    assert isinstance(prog, Program)
    assert len(prog.statements) > 0


def test_generator_is_deterministic_with_seed():
    a = render(random_program(np.random.default_rng(7)))
    b = render(random_program(np.random.default_rng(7)))
    assert a == b


def test_generator_varies_with_seed():
    a = render(random_program(np.random.default_rng(1)))
    b = render(random_program(np.random.default_rng(2)))
    assert a != b


def test_generated_source_is_nonempty_text():
    src = render(random_program(np.random.default_rng(3), num_statements=6))
    assert isinstance(src, str)
    assert "let " in src  # always declares at least one variable
