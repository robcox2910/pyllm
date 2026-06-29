"""Neural-net building blocks — think of this as a box of Lego bricks (layers) you snap
together to build a brain, all standing on the autograd Tensor from Plan 1.

Each class here (Linear, Attention, etc.) is a reusable block that learns patterns.
The `functional` module gives you the mathematical glue: softmax, cross_entropy, gelu.
Combine them to build everything from a single neuron to a full transformer.
"""

from pyllm.nn.attention import Head, MultiHeadAttention
from pyllm.nn.dropout import Dropout
from pyllm.nn.embedding import Embedding
from pyllm.nn.functional import concat, cross_entropy, embedding, gelu, softmax
from pyllm.nn.linear import Linear
from pyllm.nn.module import Module
from pyllm.nn.normalization import LayerNorm
from pyllm.nn.transformer import FeedForward, TransformerBlock

__all__ = [
    "Module",
    "Linear",
    "Embedding",
    "LayerNorm",
    "Dropout",
    "Head",
    "MultiHeadAttention",
    "FeedForward",
    "TransformerBlock",
    "softmax",
    "cross_entropy",
    "gelu",
    "embedding",
    "concat",
]
