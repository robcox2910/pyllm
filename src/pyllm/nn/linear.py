import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.module import Module


class Linear(Module):
    """A fully-connected layer: every input talks to every output.

    Picture a panel of mixing dials. Each output is a weighted blend of all the
    inputs (the `weight` numbers say how loud each input is) plus a constant
    nudge (`bias`). Learning just means turning the dials. Maths: `out = x @ W + b`.
    """

    def __init__(self, in_features, out_features, bias=True, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        scale = 1.0 / np.sqrt(in_features)
        self.weight = Tensor(
            rng.uniform(-scale, scale, size=(in_features, out_features))
        )
        self.bias = Tensor(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out
