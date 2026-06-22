import numpy as np


def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape`, reversing numpy broadcasting.

    Broadcasting often expands a tensor (e.g., shape (2,) to (2, 2)).
    Gradients flow backward against this expansion, so we sum away the
    extra dimensions to match the original shape.
    """
    # Sum away leading dimensions we didn't have originally.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum along any dimension that was size 1 (broadcast to larger).
    for axis, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


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

    def __add__(self, other):
        """Add two tensors (or a tensor and a scalar), building the compute graph.

        The result tensor remembers its parents so gradients can flow back.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # Gradients push back to both parents, unbroadcasting as needed.
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        """Support scalar + Tensor (reverse operand order)."""
        return self + other

    def backward(self):
        """Run reverse-mode autodiff from this tensor to all ancestors.

        Builds a topological order of the compute graph (dependencies-last),
        seeds this tensor's gradient with ones, then marches through nodes
        in reverse topological order, calling each node's _backward closure
        to accumulate gradients in its parents.
        """
        topo = []
        visited = set()

        def build(node):
            """Depth-first traversal: visit ancestors, then append this node."""
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data)
        # Walk backward: reverse topological order runs _backward closures.
        for node in reversed(topo):
            node._backward()
