from pyllm.nn.attention import MultiHeadAttention
from pyllm.nn.functional import gelu
from pyllm.nn.linear import Linear
from pyllm.nn.module import Module
from pyllm.nn.normalization import LayerNorm


class FeedForward(Module):
    """A little "think it over" step applied to each position on its own.

    After attention gathers information from other tokens, each token does some
    private thinking: expand to a bigger scratch space (4x), apply a smooth
    activation (`gelu`), then shrink back. It's where a lot of the model's
    per-token reasoning happens.
    """

    def __init__(self, embed_dim, rng=None):
        self.fc1 = Linear(embed_dim, 4 * embed_dim, rng=rng)
        self.fc2 = Linear(4 * embed_dim, embed_dim, rng=rng)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class TransformerBlock(Module):
    """The repeatable Lego brick of a GPT: communicate, then think.

    Two steps, each wrapped in a *residual* connection (we add the step's result
    back onto the input, like keeping your original notes and only adding edits):
    1. tokens talk to each other (multi-head attention),
    2. each token thinks privately (feed-forward).
    LayerNorm tidies the numbers before each step. Stack many of these bricks and
    you get a real language model.
    """

    def __init__(self, embed_dim, num_heads, block_size, rng=None):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size, rng=rng)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, rng=rng)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
