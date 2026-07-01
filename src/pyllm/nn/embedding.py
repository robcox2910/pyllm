import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.functional import embedding
from pyllm.nn.module import Module


class Embedding(Module):
    """A lookup table that gives every token its own little vector of meaning.

    Think of a picture dictionary: each word points to a small list of numbers
    that captures something about it. At the start the numbers are random; as the
    model learns, words that behave alike drift to similar numbers. `forward`
    just looks up the row for each incoming token id.
    """

    def __init__(self, num_embeddings, dim, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.weight = Tensor(rng.normal(0.0, 1.0, size=(num_embeddings, dim)) * 0.02)

    def forward(self, ids):
        return embedding(self.weight, ids)
