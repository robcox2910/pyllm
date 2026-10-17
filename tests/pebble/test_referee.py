"""Tests for referee."""

import numpy as np
import pytest

from pyllm.pebble import random_program, render
from pyllm.pebble.referee import complete, viable

FINISHED = [
    "let a = 1",
    "let a = 1\n",
    "let a = n",
    "let letter = no",
    "# just a comment",
    "fn add(a, b) {\n    return a + b\n}",
    "if not a == 1 and b {\n    print(-a)\n} else if a > 2 {\n    a = 0\n"
    '} else {\n    x = f(1, "hi")\n}',
    "for i in range(3) {\n    while i < 2 {\n        i = i + 1\n    }\n}",
    "let a = 1\n\nlet b = 2",
]

UNFINISHED = [
    "",
    "le",
    "whi",
    "let a = ",
    "let a = 1 an",
    "let a = not",
    'print("hi',
    "fn f(a, ",
    "if a {\n    ",
    "if a {\n    b = 1\n  ",
    "let a = 1\n\n",
]

FOULS = {
    "return outside a function": "return 1",
    "chained comparison": "let a = 1 < 2 < 3",
    "empty block": "if x {\n}",
    "keyword as a name": "let let = 1",
    "wrong indent": "if x {\n  let a = 1\n}",
    "indented top level": "let a = 1\n    let b = 2",
    "string across a newline": 'let a = "x\n"',
    "string interpolation": 'let a = "{x}"',
    "closing brace at top level": "}",
    "function inside a block": "if a {\n    fn f() {\n        let a = 1\n    }\n}",
    "two blank lines": "let a = 1\n\n\nlet b = 2",
    "blank line mid-program": "let a = (1\n\nlet b = 1",
    "trailing spaces": "print(x)  ",
    "spaced minus": "let a = - 1",
    "bare expression": "x == 1",
}


@pytest.mark.parametrize("text", FINISHED)
def test_finished_programs_are_complete_and_viable(text: str) -> None:
    """Finished programs are complete and viable."""
    assert complete(text)
    assert viable(text)


@pytest.mark.parametrize("text", UNFINISHED)
def test_unfinished_text_is_viable_but_not_complete(text: str) -> None:
    """Unfinished text is viable but not complete."""
    assert viable(text)
    assert not complete(text)


@pytest.mark.parametrize("text", list(FOULS.values()), ids=list(FOULS))
def test_rule_breaking_text_gets_the_whistle(text: str) -> None:
    """Rule breaking text gets the whistle."""
    assert not viable(text)
    assert not complete(text)


def test_every_generated_program_is_complete():
    """Every generated program is complete."""
    for seed in range(300):
        source = render(random_program(np.random.default_rng(seed)))
        assert complete(source), source


def test_every_prefix_of_a_generated_program_is_viable():
    """Every prefix of a generated program is viable."""
    for seed in range(20):
        source = render(random_program(np.random.default_rng(seed)))
        for end in range(len(source) + 1):
            assert viable(source[:end]), repr(source[:end])


def test_formatter_style_programs_are_complete():
    """Formatter style programs are complete."""
    pytest.importorskip("pebble")
    from pyllm.pebble.corpus import canonicalize

    for seed in range(100):
        source = canonicalize(render(random_program(np.random.default_rng(seed))))
        assert complete(source), source


def test_everything_the_referee_accepts_really_parses():
    """Everything the referee accepts really parses."""
    pytest.importorskip("pebble")
    from pyllm.pebble import is_valid

    for text in FINISHED:
        for program in text.split("\n\n"):
            assert is_valid(program), program


@pytest.mark.parametrize("word", ["match", "class", "from", "case", "await"])
def test_pebble_reserved_words_cannot_be_names(word: str) -> None:
    """Pebble reserved words cannot be names."""
    assert not complete(f"let {word} = 1")
    assert not complete(f"let x = {word}")


def test_null_is_a_value():
    """Null is a value."""
    assert complete("let x = null")


def test_referee_knows_every_pebble_reserved_word():
    """Referee knows every pebble reserved word."""
    pytest.importorskip("pebble")
    from pebble.tokens import KEYWORDS

    from pyllm.pebble.referee import _KEYWORDS

    assert set(KEYWORDS) <= _KEYWORDS
