from pyllm.nn import Embedding, Linear, Module, gelu


class MLP(Module):
    """A small neural net that reads the last few characters, then guesses the next.

    The bigram only ever looks at *one* letter. This model looks at a little
    window of the last `block_size` letters at once. It turns each letter into a
    short list of numbers (an embedding), lays those lists out in one long row,
    and passes them through a tiny "thinking" layer (a hidden Linear + a smooth
    `gelu` switch) before guessing the next letter. Wider context = smarter
    guesses than the bigram.
    """

    def __init__(self, vocab_size, block_size=3, embed_dim=16, hidden_dim=64, rng=None):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.mode = "single"
        self.token_embedding = Embedding(vocab_size, embed_dim, rng=rng)
        self.fc1 = Linear(block_size * embed_dim, hidden_dim, rng=rng)
        self.fc2 = Linear(hidden_dim, vocab_size, rng=rng)

    def forward(self, idx):
        batch = idx.shape[0]
        emb = self.token_embedding(idx)  # (B, block_size, embed_dim)
        flat = emb.reshape((batch, self.block_size * self.embed_dim))
        hidden = gelu(self.fc1(flat))
        return self.fc2(hidden)  # (B, vocab_size)

    def config(self):
        """The recipe card needed to rebuild an identical (untrained) model."""
        return {
            "kind": "mlp",
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
        }
