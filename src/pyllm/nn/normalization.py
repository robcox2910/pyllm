import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.module import Module


class LayerNorm(Module):
    """Put every row on the same fair scale before comparing them.

    Imagine grading tests from different teachers who mark very differently. To
    compare students fairly you re-scale each class so it has the same average
    and spread. LayerNorm does that to each row of numbers: subtract the row's
    average, divide by its spread. Then two learnable dials (`gamma`, `beta`) let
    the network stretch and shift the result if it wants.
    """

    def __init__(self, dim, eps=1e-5):
        self.gamma = Tensor(np.ones(dim))
        self.beta = Tensor(np.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        centered = x - mean
        variance = (centered ** 2).mean(axis=-1, keepdims=True)
        normed = centered / ((variance + self.eps) ** 0.5)
        return normed * self.gamma + self.beta
