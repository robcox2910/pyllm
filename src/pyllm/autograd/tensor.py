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
    """A number (or grid of numbers) that drops breadcrumbs as it travels through math.

    Think of it like a hiker who leaves a trail of pebbles on the way to the top
    of a hill.  When we reach the summit (the final answer, like a loss), we walk
    back along those pebbles to figure out how much each step contributed to how
    high we climbed.  That walk-back is called the *backward pass*, and the
    breadcrumbs are stored in `_prev` (the parents) and `_backward` (the recipe
    for sharing blame with each parent).

    `data`  — the actual number(s) carried right now.
    `grad`  — how much this tensor should change to make the final answer smaller.
    """

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
        """Pour two cups of water together to get one bigger cup.

        When we add `a + b`, the result is just the combined total — simple!
        The clever bit is in *blame*: if the total is off by a little, both
        cups share that blame equally.  Each parent receives a copy of the
        output's gradient, unbroadcasted back to its original shape (breadcrumb:
        "the error flows back unchanged to both sides").
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
        """Support `scalar + Tensor` — called when the left side isn't a Tensor.

        Flips the order and delegates to `__add__`, so `3 + t` works like `t + 3`.
        """
        return self + other

    def __mul__(self, other):
        """Scale one number by another — like doubling a recipe.

        If you double the amount of flour (`a * 2`), the whole cake gets twice as big.
        When we ask "how much did `a` affect the result?", the answer depends on what
        `b` was (and vice versa).  Breadcrumb rule: each side's gradient is the
        output's gradient multiplied by the *other* side's value
        (`d(a*b)/da = b`, `d(a*b)/db = a`).
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # Gradient with respect to self: out.grad * other (d(a*b)/da = b)
            # Gradient with respect to other: out.grad * self (d(a*b)/db = a)
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        """Support `scalar * Tensor` — called when the left side isn't a Tensor.

        Flips the order and delegates to `__mul__`, so `3 * t` works like `t * 3`.
        """
        return self * other

    def __matmul__(self, other):
        """Combine two tables of numbers — like running many recipes at once.

        Imagine `A` is a table of ingredients (one row per dish) and `B` is a
        table of recipes (one column per flavour).  `A @ B` mixes every dish
        through every recipe in one go.  Breadcrumb rule: error flows back to
        `A` via `out.grad @ B^T`, and back to `B` via `A^T @ out.grad`
        (`d(A@B)/dA = out.grad @ B^T`, `d(A@B)/dB = A^T @ out.grad`).
        """
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        """Pile up all the numbers into one stack; each piece shares the blame equally.

        Like adding up everyone's pocket money into one pile: if the total is wrong,
        every coin shares the blame equally. Gradient flows back as 1.0 to each element.
        """
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        """Find the average — each number pitches in equally to the overall picture.

        Like a class average: each student's score contributes a fair (1/N) share,
        so if the class average is off, each student gets blamed by 1/N of that error.
        """
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")
        count = self.data.size if axis is None else self.data.shape[axis]

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad / count

        out._backward = _backward
        return out

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
        # This is the root — the final answer we're asking "how much does it change?"
        # The answer: by exactly 1 (the loss itself). We start the breadcrumb trail
        # with ones here because it's the very beginning. Every other tensor will
        # accumulate gradient with += because they may have been used in several places.
        self.grad = np.ones_like(self.data)
        # Walk backward: reverse topological order runs _backward closures.
        for node in reversed(topo):
            node._backward()
