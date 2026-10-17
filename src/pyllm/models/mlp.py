"""MLP: a small neural net that reads the last few characters at once."""

import numpy as np
from numpy.typing import NDArray

from pyllm.autograd import Tensor
from pyllm.models.base import LanguageModel, ModelConfig
from pyllm.nn import Embedding, Linear, gelu


class MLP(LanguageModel):
    """A small neural net that reads the last few characters, then guesses the next.

    The bigram only ever looks at *one* letter. This model looks at a little
    window of the last `block_size` letters at once. It turns each letter into a
    short list of numbers (an embedding), lays those lists out in one long row,
    and passes them through a tiny "thinking" layer (a hidden Linear + a smooth
    `gelu` switch) before guessing the next letter. Wider context = smarter
    guesses than the bigram.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 3,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Create the embedding table and the two Linear layers."""
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.mode = "single"
        self.token_embedding = Embedding(vocab_size, embed_dim, rng=rng)
        self.fc1 = Linear(block_size * embed_dim, hidden_dim, rng=rng)
        self.fc2 = Linear(hidden_dim, vocab_size, rng=rng)

    def forward(self, idx: NDArray[np.int64]) -> Tensor:
        """Read a window of token ids and score the one token that follows."""
        batch = idx.shape[0]
        emb = self.token_embedding(idx)  # (B, block_size, embed_dim)
        flat = emb.reshape((batch, self.block_size * self.embed_dim))
        hidden = gelu(self.fc1(flat))
        return self.fc2(hidden)  # (B, vocab_size)

    def config(self) -> ModelConfig:
        """Return the recipe card needed to rebuild an identical (untrained) model."""
        return {
            "kind": "mlp",
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
        }
