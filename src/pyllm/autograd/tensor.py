import numpy as np


class Tensor:
    """A numpy array that remembers how to compute its own gradient."""

    def __init__(self, data, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
