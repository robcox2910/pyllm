from pyllm.nn import Embedding
from pyllm.nn import Module


class Bigram(Module):
    """The simplest language model: "given this letter, what usually comes next?"

    It's a giant lookup table. For every possible current character it stores a
    row of scores -- one score per possible *next* character. There is no
    thinking and no memory of anything before the current letter; it just reads
    off the row. It learns which next-letters are common (after "q" comes "u"),
    which is already enough to babble vaguely word-shaped text.
    """

    def __init__(self, vocab_size, block_size=8, rng=None):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.mode = "sequence"
        # Each row IS the next-token scores for that token id.
        self.token_embedding = Embedding(vocab_size, vocab_size, rng=rng)

    def forward(self, idx):
        return self.token_embedding(idx)  # (B, T, vocab_size)

    def config(self):
        """The recipe card needed to rebuild an identical (untrained) model."""
        return {
            "kind": "bigram",
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
        }
