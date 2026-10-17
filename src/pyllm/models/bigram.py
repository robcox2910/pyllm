"""Bigram: the simplest language model, a giant next-letter lookup table."""

import numpy as np
from numpy.typing import ArrayLike

from pyllm.autograd import Tensor
from pyllm.models.base import LanguageModel, ModelConfig
from pyllm.nn import Embedding


class Bigram(LanguageModel):
    """The simplest language model, it only asks what usually follows this letter.

    It's a giant lookup table. For every possible current character it stores a
    row of scores -- one score per possible *next* character. There is no
    thinking and no memory of anything before the current letter; it just reads
    off the row. It learns which next-letters are common (after "q" comes "u"),
    which is already enough to babble vaguely word-shaped text.
    """

    def __init__(
        self, vocab_size: int, block_size: int = 8, rng: np.random.Generator | None = None
    ) -> None:
        """Create the lookup table: one row of next-token scores per token."""
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.mode = "sequence"
        # Each row IS the next-token scores for that token id.
        self.token_embedding = Embedding(vocab_size, vocab_size, rng=rng)

    def forward(self, idx: ArrayLike) -> Tensor:
        """Look up the next-token scores for every token id."""
        return self.token_embedding(idx)  # (B, T, vocab_size)

    def config(self) -> ModelConfig:
        """Return the recipe card needed to rebuild an identical (untrained) model."""
        return {
            "kind": "bigram",
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
        }
