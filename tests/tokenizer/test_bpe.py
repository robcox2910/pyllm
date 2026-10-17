"""Tests for bpe."""

from pyllm.tokenizer.bpe import BPETokenizer


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
