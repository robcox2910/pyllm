import numpy as np

from pyllm.nn.module import Module


class Dropout(Module):
    """Randomly ignore some signals during training so the net doesn't over-rely.

    Like a sports team practising with random players sitting out each drill, so
    nobody becomes a single point of failure. During training we randomly zero
    out a fraction `p` of the numbers and make the survivors a bit louder (divide
    by `1-p`) so the total stays about the same. At test time everyone plays, so
    dropout does nothing.
    """

    def __init__(self, p=0.1, rng=None):
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()
        self.training = True

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = (self.rng.uniform(size=x.shape) > self.p) / (1.0 - self.p)
        return x * keep
