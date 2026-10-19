"""Tests for bpe."""

from collections import Counter

import pytest

from pyllm.tokenizer.bpe import BPETokenizer, _best_pair


def test_bpe_roundtrip():
    """Bpe roundtrip."""
    tok = BPETokenizer()
    tok.train("ababab abc", num_merges=3)
    assert tok.decode(tok.encode("ababab")) == "ababab"


def test_bpe_merges_reduce_token_count():
    """Bpe merges reduce token count."""
    text = "abababababab"
    tok = BPETokenizer()
    tok.train(text, num_merges=2)
    # "ab" should merge, so encoding is much shorter than 12 chars
    assert len(tok.encode(text)) < len(text)


def test_bpe_vocab_grows_by_merges():
    """Bpe vocab grows by merges."""
    text = "ababab"  # base chars: a, b  => 2
    tok = BPETokenizer()
    tok.train(text, num_merges=1)
    assert tok.vocab_size == 3  # a, b, and merged "ab"


def test_bpe_learns_most_frequent_pair_first():
    """Bpe learns most frequent pair first."""
    tok = BPETokenizer()
    tok.train("ababab", num_merges=1)
    # the first (and only) merge should be the pair ('a', 'b')
    assert tok.merges[0] == ("a", "b")


def test_from_merges_refuses_two_merges_that_make_the_same_token():
    """(ab, c) and (a, bc) both make "abc"; a tokenizer must never hold one token twice."""
    merges = [("a", "b"), ("b", "c"), ("ab", "c"), ("a", "bc")]
    with pytest.raises(ValueError, match="'abc'"):
        BPETokenizer.from_merges(["a", "b", "c"], merges)


def test_training_never_picks_a_pair_that_makes_a_known_token():
    """Training skips a pair whose glued token already exists and takes the next best."""
    pairs = Counter({("a", "bc"): 5, ("c", "a"): 3})
    assert _best_pair(pairs, known={"a", "b", "c", "bc", "abc"}) == ("c", "a")


def test_training_stops_when_only_known_tokens_are_left():
    """If every common pair would remake a known token, there is nothing left to learn."""
    pairs = Counter({("a", "bc"): 5})
    assert _best_pair(pairs, known={"a", "b", "c", "bc", "abc"}) is None
