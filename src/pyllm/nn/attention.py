import numpy as np

from pyllm.nn.functional import concat, softmax
from pyllm.nn.linear import Linear
from pyllm.nn.module import Module


class Head(Module):
    """One "attention" head: each token decides which earlier tokens to listen to.

    Reading a sentence, the word "it" looks back to find what "it" refers to.
    A head does that: every token makes a *query* ("what am I looking for?"),
    every token offers a *key* ("here's what I am"), and tokens whose keys match
    the query get listened to most. The *value* is the information actually
    passed along. A causal mask hides the future, so a token can only look
    backward -- you can't peek at words you haven't read yet.
    """

    def __init__(self, embed_dim, head_size, block_size, rng=None):
        self.key = Linear(embed_dim, head_size, bias=False, rng=rng)
        self.query = Linear(embed_dim, head_size, bias=False, rng=rng)
        self.value = Linear(embed_dim, head_size, bias=False, rng=rng)
        self.head_size = head_size
        # Buffer (NOT a parameter): -1e9 above the diagonal blocks the future.
        allowed = np.tril(np.ones((block_size, block_size)))
        self.mask = np.where(allowed == 0, -1e9, 0.0)

    def forward(self, x):
        seq_len = x.shape[1]
        q = self.query(x)            # (B, T, head_size)
        k = self.key(x)
        v = self.value(x)
        scores = (q @ k.transpose()) / np.sqrt(self.head_size)  # (B, T, T)
        scores = scores + self.mask[:seq_len, :seq_len]
        weights = softmax(scores, axis=-1)
        return weights @ v           # (B, T, head_size)


class MultiHeadAttention(Module):
    """Several attention heads working in parallel, then combined.

    One head can only track one kind of relationship at a time. Like a panel of
    readers where one watches for "who did what", another for "when", another for
    "where" -- each head looks for a different pattern. We run them all at once,
    lay their reports side by side (`concat`), and blend them with a final
    Linear so the network can mix what every head found.
    """

    def __init__(self, embed_dim, num_heads, block_size, rng=None):
        assert embed_dim % num_heads == 0, "embed_dim must divide evenly by num_heads"
        head_size = embed_dim // num_heads
        self.heads = [
            Head(embed_dim, head_size, block_size, rng=rng) for _ in range(num_heads)
        ]
        self.proj = Linear(embed_dim, embed_dim, rng=rng)

    def forward(self, x):
        combined = concat([head(x) for head in self.heads], axis=-1)
        return self.proj(combined)
