import numpy as np

from pyllm.nn import Embedding, LayerNorm, Linear, Module, TransformerBlock, embedding


class GPT(Module):
    """The real thing: a stack of Transformer blocks that reads a whole window.

    Each character becomes a vector (token embedding), and because "the cat sat"
    is not "sat the cat", we also add a vector for *where* it sits (position
    embedding). Then a tower of Transformer blocks lets every position gather
    clues from the positions before it (attention) and think them over
    (feed-forward). A final tidy-up (LayerNorm) and a last Linear turn each
    position into scores for the next character. Stack more blocks -> smarter.
    Pass `dropout` above 0 to randomly ignore some signals while training (which
    helps the model generalise instead of memorising); it turns itself off during
    generation, so the default of 0 keeps behaviour exactly as before.
    """

    def __init__(
        self,
        vocab_size,
        block_size=8,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        rng=None,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode = "sequence"
        self.token_embedding = Embedding(vocab_size, embed_dim, rng=rng)
        self.position_embedding = Embedding(block_size, embed_dim, rng=rng)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, block_size, dropout=dropout, rng=rng)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size, rng=rng)

    def forward(self, idx):
        seq_len = idx.shape[1]
        tok = self.token_embedding(idx)  # (B, T, embed_dim)
        pos = embedding(self.position_embedding.weight, np.arange(seq_len))
        x = tok + pos  # (T, embed_dim) broadcasts over the batch
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)

    def config(self):
        """The recipe card needed to rebuild an identical (untrained) model."""
        return {
            "kind": "gpt",
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }
