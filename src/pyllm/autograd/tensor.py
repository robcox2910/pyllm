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

    def __neg__(self):
        """Flip the sign, like owing money instead of having it.

        Negation is just multiplication by -1, so gradients flow backward
        with the negative sign automatically.
        """
        return self * -1.0

    def __sub__(self, other):
        """Take away — like removing coins from a pile.

        Subtraction is built from addition and negation, so gradients
        just reuse those paths: `a - b` becomes `a + (-b)`.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        """Support `scalar - Tensor` — called when the left side isn't a Tensor.

        Flips the operation via negation: `c - t` becomes `(-t) + c`.
        """
        return (-self) + other

    def __pow__(self, power):
        """Repeated multiplication, e.g., squaring (x^2) or cubing (x^3).

        The slope of x^n is n·x^(n-1), so exponentiation is the only new
        primitive here; the gradient flows back via that power rule.
        """
        if not isinstance(power, (int, float)):
            raise TypeError("Tensor ** power supports int/float exponents only")
        out = Tensor(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        """Share into equal parts — like splitting a pizza among friends.

        Division is built as multiply by the reciprocal: `a / b` becomes
        `a * (b ^ -1)`, so gradients flow through multiplication and power.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        """Support `scalar / Tensor` — called when the left side isn't a Tensor.

        Reorders via power: `c / t` becomes `(t ^ -1) * c`.
        """
        return (self ** -1) * other

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

    def relu(self):
        """A one-way gate for numbers — lets positive signals through, blocks negatives.

        Think of a water valve that only opens when water is flowing forward: positive
        numbers pass unchanged, but negatives get shut out (become zero). Gradient is 1
        where it let the signal through, 0 where it blocked it.
        """
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        """Rapid growth, like money doubling each time — the faster the more it grows.

        Exponential turns small numbers into huge ones (e^2 ≈ 7, e^5 ≈ 148), and the
        slope at any point equals its own height, so its gradient is just itself.
        """
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        """The opposite of exp — undoes growth by asking 'e to what power equals this?'

        Log shrinks big numbers down (log(e^2) = 2, log(100) ≈ 4.6), so it's the
        inverse of exponential. Its gradient is 1/x, so it gets flatter as x grows.
        """
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        """A squasher that gently flattens any number into the range -1 to 1.

        No matter how big or negative your number is, tanh squeezes it: tanh(0) = 0,
        tanh(big) ≈ 1, tanh(-big) ≈ -1. Its gradient is 1 - tanh²(x), so the slope is
        steepest at zero and flattens toward the edges.
        """
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

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

    def zero_grad(self):
        """Wipe the slate clean before the next learning step.

        Like erasing last round's notes so this round's blame doesn't pile on top.
        When you run backward() again, the gradients will accumulate from scratch,
        letting you take a fresh gradient descent step without old errors interfering.
        """
        self.grad = np.zeros_like(self.data)
