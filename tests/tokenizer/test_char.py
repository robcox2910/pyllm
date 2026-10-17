"""Tests for char."""

from pyllm.tokenizer.char import CharTokenizer


def test_vocab_size_counts_unique_chars():
    """Vocab size counts unique chars."""
    tok = CharTokenizer("hello")  # h, e, l, o
    assert tok.vocab_size == 4


def test_encode_decode_roundtrip():
    """Encode decode roundtrip."""
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"


def test_encode_returns_ints_in_range():
    """Encode returns ints in range."""
    tok = CharTokenizer("abc")
    ids = tok.encode("cab")
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_vocab_is_sorted_and_stable():
    # sorted unique chars -> 'a','b','c' => a=0, b=1, c=2
    """Vocab is sorted and stable."""
    tok = CharTokenizer("cba")
    assert tok.encode("abc") == [0, 1, 2]
