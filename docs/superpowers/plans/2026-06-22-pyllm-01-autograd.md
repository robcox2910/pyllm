# PyLLM Plan 1 — Autograd Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `Tensor` class that wraps a numpy array and performs reverse-mode automatic differentiation, so `loss.backward()` fills in correct gradients for every value that fed into it.

**Architecture:** A `Tensor` holds `data` (numpy array) and `grad` (numpy array). Every operation returns a new `Tensor` that remembers its parent tensors and a local `_backward` closure describing how to push gradient to those parents. `backward()` topologically sorts the graph and runs the closures in reverse. numpy does the array arithmetic; the gradient logic is entirely ours. Correctness is proven by comparing analytic gradients to finite-difference numerical gradients.

**Tech Stack:** Python 3.14, numpy, pytest, ruff, pyright, managed with `uv`.

## Global Constraints

- **RULE #1 — Child-friendly docs about EVERYTHING.** Every class and function
  gets a docstring a 12-year-old could follow (analogy first, jargon later), and
  this plan is not done until its concept doc (`docs/concepts/autograd.md`)
  exists. This rule outranks all others.
- **Python 3.14**, managed with `uv`. Run everything via `uv run`.
- **Dependencies:** `numpy` only (plus dev/test/docs tooling). Nothing else.
- **TDD:** write the failing test first, watch it fail, then implement.
- **ruff** + **pyright** must stay clean after every task.
- **No `TYPE_CHECKING`, no `from __future__ import annotations`.** (Python 3.14 evaluates annotations lazily, so referring to `Tensor` inside its own class body is fine.)
- All `Tensor` data is stored as `np.float64`.
- Frequent commits: one commit per task minimum.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/pyllm/__init__.py`
- Create: `src/pyllm/autograd/__init__.py`
- Create: `README.md`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `pyllm` package and a working `uv run pytest` / `uv run ruff` / `uv run pyright` toolchain.

- [ ] **Step 1: Write the failing test**

Create `tests/test_smoke.py`:

```python
import pyllm


def test_package_imports():
    assert pyllm.__name__ == "pyllm"
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "pyllm"
dynamic = ["version"]
description = "An educational large language model built from scratch in Python"
readme = "README.md"
requires-python = ">=3.14"
license = {text = "MIT"}
authors = [{ name = "Rob Cox" }]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Education",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.14",
    "Topic :: Education",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
keywords = ["llm", "gpt", "transformer", "autograd", "education", "learning"]
dependencies = ["numpy>=2.1"]

[project.urls]
Homepage = "https://github.com/robcox2910/pyllm"
Repository = "https://github.com/robcox2910/pyllm"
Issues = "https://github.com/robcox2910/pyllm/issues"
Documentation = "https://robcox2910.github.io/pyllm/"

[project.optional-dependencies]
dev = [
    "pytest>=8.3.4",
    "pytest-cov>=6.0.0",
    "ruff>=0.9.3",
    "pre-commit>=4.0.1",
    "pyright>=1.1.400",
    "commitizen>=4.4.0",
    "pip-audit>=2.10.0",
]
docs = ["mkdocs-material>=9.5"]

[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.hooks.vcs]
version-file = "src/pyllm/_version.py"

[tool.hatch.build.targets.wheel]
packages = ["src/pyllm"]

[tool.ruff]
line-length = 88
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pyright]
include = ["src", "tests"]
pythonVersion = "3.14"
typeCheckingMode = "basic"
```

- [ ] **Step 3: Create the package files**

Create `src/pyllm/__init__.py`:

```python
"""PyLLM: an educational large language model built from scratch."""
```

Create `src/pyllm/autograd/__init__.py`:

```python
"""The autograd engine: tensors that remember how to compute their gradients."""
```

Create `README.md`:

```markdown
# PyLLM

An educational large language model built from scratch in Python.

Part of the "from scratch in Python" series. Built incrementally with TDD,
every concept explained with analogies a 12-year-old can follow.

## Quick Start

```bash
uv sync --all-extras
uv run pytest
```
```

- [ ] **Step 4: Sync and run the test**

Run:
```bash
uv sync --all-extras
uv run pytest tests/test_smoke.py -v
```
Expected: PASS (1 passed).

- [ ] **Step 5: Verify tooling is clean**

Run:
```bash
uv run ruff check .
uv run pyright src tests
```
Expected: ruff reports "All checks passed!"; pyright reports 0 errors.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: scaffold pyllm package and toolchain"
```

---

### Task 2: Tensor core (construction, shape, repr)

**Files:**
- Create: `src/pyllm/autograd/tensor.py`
- Test: `tests/autograd/test_tensor.py`

**Interfaces:**
- Consumes: numpy.
- Produces: `Tensor(data)` where `data` is any array-like; attributes `Tensor.data` (np.ndarray float64), `Tensor.grad` (np.ndarray of zeros, same shape), `Tensor.shape` (tuple). Constructor signature `Tensor(data, _children=(), _op="")`.

- [ ] **Step 1: Write the failing test**

Create `tests/autograd/test_tensor.py`:

```python
import numpy as np

from pyllm.autograd.tensor import Tensor


def test_wraps_data_as_float64():
    t = Tensor([1, 2, 3])
    assert t.data.dtype == np.float64
    assert t.data.tolist() == [1.0, 2.0, 3.0]


def test_grad_starts_at_zero_same_shape():
    t = Tensor([[1, 2], [3, 4]])
    assert t.grad.shape == (2, 2)
    assert np.all(t.grad == 0.0)


def test_shape_property():
    assert Tensor([[1, 2, 3]]).shape == (1, 3)


def test_repr_mentions_tensor():
    assert "Tensor" in repr(Tensor([1.0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_tensor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.autograd.tensor'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/autograd/tensor.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_tensor.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tensor core (data, grad, shape, repr)"
```

---

### Task 3: Addition + backward (with broadcasting)

**Files:**
- Modify: `src/pyllm/autograd/tensor.py`
- Test: `tests/autograd/test_tensor.py`

**Interfaces:**
- Consumes: `Tensor` from Task 2.
- Produces: `Tensor.__add__` / `__radd__` (accepts `Tensor` or scalar); `Tensor.backward()` which seeds the called tensor's grad with ones and runs `_backward` closures in reverse topological order. A module-level helper `_unbroadcast(grad, shape)` that reverses numpy broadcasting by summing.

- [ ] **Step 1: Write the failing test**

Add to `tests/autograd/test_tensor.py`:

```python
def test_add_forward():
    out = Tensor([1.0, 2.0]) + Tensor([10.0, 20.0])
    assert out.data.tolist() == [11.0, 22.0]


def test_add_backward_passes_grad_to_both():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    out = a + b
    out.backward()
    assert a.grad.tolist() == [1.0, 1.0]
    assert b.grad.tolist() == [1.0, 1.0]


def test_add_scalar_on_right():
    out = Tensor([1.0, 2.0]) + 5.0
    assert out.data.tolist() == [6.0, 7.0]


def test_add_broadcasting_reduces_grad():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
    b = Tensor([10.0, 20.0])              # shape (2,) broadcasts
    out = a + b
    out.backward()
    # b is used in both rows, so its grad sums across rows.
    assert b.grad.tolist() == [2.0, 2.0]
    assert a.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_tensor.py -k add -v`
Expected: FAIL with `TypeError: unsupported operand type(s)` / `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

Add the helper at the top of `src/pyllm/autograd/tensor.py` (after the imports, before the class):

```python
def _unbroadcast(grad, shape):
    """Sum `grad` back down to `shape`, reversing numpy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad
```

Add these methods to the `Tensor` class:

```python
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_tensor.py -k add -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tensor addition and reverse-mode backward"
```

---

### Task 4: Multiplication

**Files:**
- Modify: `src/pyllm/autograd/tensor.py`
- Test: `tests/autograd/test_tensor.py`

**Interfaces:**
- Consumes: `Tensor`, `_unbroadcast`, `backward` from Task 3.
- Produces: `Tensor.__mul__` / `__rmul__` (elementwise, broadcasting-aware).

- [ ] **Step 1: Write the failing test**

Add to `tests/autograd/test_tensor.py`:

```python
def test_mul_forward():
    out = Tensor([2.0, 3.0]) * Tensor([4.0, 5.0])
    assert out.data.tolist() == [8.0, 15.0]


def test_mul_backward_uses_other_operand():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    out = a * b
    out.backward()
    assert a.grad.tolist() == [4.0, 5.0]  # d(a*b)/da = b
    assert b.grad.tolist() == [2.0, 3.0]  # d(a*b)/db = a


def test_mul_scalar_on_right():
    out = Tensor([2.0, 3.0]) * 10.0
    assert out.data.tolist() == [20.0, 30.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_tensor.py -k mul -v`
Expected: FAIL with `TypeError: unsupported operand type(s) for *`.

- [ ] **Step 3: Write minimal implementation**

Add to the `Tensor` class:

```python
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_tensor.py -k mul -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tensor multiplication"
```

---

### Task 5: Matrix multiplication

**Files:**
- Modify: `src/pyllm/autograd/tensor.py`
- Test: `tests/autograd/test_tensor.py`

**Interfaces:**
- Consumes: `Tensor`, `backward`.
- Produces: `Tensor.__matmul__` for 2-D (and higher) arrays. Gradient uses `out.grad @ other.dataᵀ` and `self.dataᵀ @ out.grad`, transposing the last two axes via `swapaxes(-1, -2)`. (Batch dimensions are assumed to match; broadcasting over batch dims is out of scope for this task.)

- [ ] **Step 1: Write the failing test**

Add to `tests/autograd/test_tensor.py`:

```python
def test_matmul_forward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    out = a @ b
    assert out.data.tolist() == [[19.0, 22.0], [43.0, 50.0]]


def test_matmul_backward_shapes_and_values():
    a = Tensor([[1.0, 2.0, 3.0]])      # (1, 3)
    b = Tensor([[1.0], [1.0], [1.0]])  # (3, 1)
    out = a @ b                         # (1, 1)
    out.backward()
    # d(a@b)/da = out.grad @ b.T ; with out.grad = ones((1,1)) -> b.T = [[1,1,1]]
    assert a.grad.tolist() == [[1.0, 1.0, 1.0]]
    # d(a@b)/db = a.T @ out.grad -> [[1],[2],[3]]
    assert b.grad.tolist() == [[1.0], [2.0], [3.0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_tensor.py -k matmul -v`
Expected: FAIL with `TypeError: unsupported operand type(s) for @`.

- [ ] **Step 3: Write minimal implementation**

Add to the `Tensor` class:

```python
    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.swapaxes(-1, -2)
            other.grad += self.data.swapaxes(-1, -2) @ out.grad

        out._backward = _backward
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_tensor.py -k matmul -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tensor matrix multiplication"
```

---

### Task 6: Reductions (sum, mean)

**Files:**
- Modify: `src/pyllm/autograd/tensor.py`
- Test: `tests/autograd/test_tensor.py`

**Interfaces:**
- Consumes: `Tensor`, `backward`.
- Produces: `Tensor.sum(axis=None, keepdims=False)` and `Tensor.mean(axis=None, keepdims=False)`. `axis` is `None` or a single int.

- [ ] **Step 1: Write the failing test**

Add to `tests/autograd/test_tensor.py`:

```python
def test_sum_all_forward_and_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    out = a.sum()
    assert out.data.tolist() == 10.0
    out.backward()
    assert a.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_sum_axis_backward_broadcasts():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    out = a.sum(axis=0)  # -> [4.0, 6.0]
    assert out.data.tolist() == [4.0, 6.0]
    out.backward()
    assert a.grad.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_mean_all_backward():
    a = Tensor([2.0, 4.0, 6.0, 8.0])
    out = a.mean()
    assert out.data.tolist() == 5.0
    out.backward()
    assert a.grad.tolist() == [0.25, 0.25, 0.25, 0.25]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_tensor.py -k "sum or mean" -v`
Expected: FAIL with `AttributeError: 'Tensor' object has no attribute 'sum'`.

- [ ] **Step 3: Write minimal implementation**

Add to the `Tensor` class:

```python
    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")
        count = self.data.size if axis is None else self.data.shape[axis]

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            self.grad += np.ones_like(self.data) * grad / count

        out._backward = _backward
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_tensor.py -k "sum or mean" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tensor sum and mean reductions"
```

---

### Task 7: Activations (relu, exp, log, tanh)

**Files:**
- Modify: `src/pyllm/autograd/tensor.py`
- Test: `tests/autograd/test_tensor.py`

**Interfaces:**
- Consumes: `Tensor`, `backward`.
- Produces: `Tensor.relu()`, `Tensor.exp()`, `Tensor.log()`, `Tensor.tanh()` — all elementwise.

- [ ] **Step 1: Write the failing test**

Add to `tests/autograd/test_tensor.py`:

```python
def test_relu_forward_and_backward():
    a = Tensor([-2.0, 0.0, 3.0])
    out = a.relu()
    assert out.data.tolist() == [0.0, 0.0, 3.0]
    out.backward()
    assert a.grad.tolist() == [0.0, 0.0, 1.0]


def test_exp_backward_is_exp():
    a = Tensor([0.0, 1.0])
    out = a.exp()
    out.backward()
    assert np.allclose(a.grad, np.exp([0.0, 1.0]))


def test_log_backward_is_reciprocal():
    a = Tensor([1.0, 2.0, 4.0])
    out = a.log()
    out.backward()
    assert np.allclose(a.grad, [1.0, 0.5, 0.25])


def test_tanh_backward():
    a = Tensor([0.0])
    out = a.tanh()
    out.backward()
    assert np.allclose(a.grad, [1.0])  # 1 - tanh(0)^2 = 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_tensor.py -k "relu or exp or log or tanh" -v`
Expected: FAIL with `AttributeError: 'Tensor' object has no attribute 'relu'`.

- [ ] **Step 3: Write minimal implementation**

Add to the `Tensor` class:

```python
    def relu(self):
        out = Tensor(np.maximum(0.0, self.data), (self,), "relu")

        def _backward():
            self.grad += (out.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_tensor.py -k "relu or exp or log or tanh" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tensor activations (relu, exp, log, tanh)"
```

---

### Task 8: Derived ops (neg, sub, pow, truediv)

**Files:**
- Modify: `src/pyllm/autograd/tensor.py`
- Test: `tests/autograd/test_tensor.py`

**Interfaces:**
- Consumes: `Tensor.__mul__`, `__add__`.
- Produces: `__neg__`, `__sub__`, `__rsub__`, `__pow__` (scalar exponent only), `__truediv__`, `__rtruediv__`. `__pow__` is the only new primitive (with its own `_backward`); the rest are composed from existing ops so their gradients flow automatically.

- [ ] **Step 1: Write the failing test**

Add to `tests/autograd/test_tensor.py`:

```python
def test_neg_and_sub():
    a = Tensor([5.0, 7.0])
    b = Tensor([2.0, 3.0])
    out = a - b
    assert out.data.tolist() == [3.0, 4.0]
    out.backward()
    assert a.grad.tolist() == [1.0, 1.0]
    assert b.grad.tolist() == [-1.0, -1.0]


def test_pow_backward():
    a = Tensor([2.0, 3.0])
    out = a ** 2
    assert out.data.tolist() == [4.0, 9.0]
    out.backward()
    assert a.grad.tolist() == [4.0, 6.0]  # d(x^2)/dx = 2x


def test_truediv_backward():
    a = Tensor([6.0])
    b = Tensor([2.0])
    out = a / b
    assert out.data.tolist() == [3.0]
    out.backward()
    assert np.allclose(a.grad, [0.5])     # 1/b
    assert np.allclose(b.grad, [-1.5])    # -a/b^2


def test_rsub_and_rtruediv():
    a = Tensor([2.0])
    assert (10.0 - a).data.tolist() == [8.0]
    assert (6.0 / a).data.tolist() == [3.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_tensor.py -k "neg or sub or pow or truediv" -v`
Expected: FAIL with `TypeError` on `-`, `**`, or `/`.

- [ ] **Step 3: Write minimal implementation**

Add to the `Tensor` class:

```python
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Tensor ** power supports int/float exponents only")
        out = Tensor(self.data ** power, (self,), f"**{power}")

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return (self ** -1) * other
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_tensor.py -k "neg or sub or pow or truediv" -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tensor neg, sub, pow, truediv"
```

---

### Task 9: Numerical gradient checking

**Files:**
- Create: `src/pyllm/autograd/gradcheck.py`
- Test: `tests/autograd/test_gradcheck.py`

**Interfaces:**
- Consumes: `Tensor`.
- Produces: `numerical_grad(make_output, wrt, eps=1e-6) -> np.ndarray` — given a zero-argument function `make_output` that builds and returns an output `Tensor`, and the `Tensor` `wrt` to differentiate with respect to, returns the finite-difference gradient of `sum(make_output().data)` w.r.t. each element of `wrt.data`. This is the oracle used to prove `backward()` is correct.

- [ ] **Step 1: Write the failing test**

Create `tests/autograd/test_gradcheck.py`:

```python
import numpy as np

from pyllm.autograd.gradcheck import numerical_grad
from pyllm.autograd.tensor import Tensor


def _check(make_output, *inputs):
    """Assert analytic grads from backward() match numerical grads."""
    out = make_output()
    out.backward()
    for t in inputs:
        approx = numerical_grad(make_output, t)
        assert np.allclose(t.grad, approx, atol=1e-4), (
            f"grad mismatch:\nanalytic={t.grad}\nnumerical={approx}"
        )


def test_gradcheck_add_mul():
    a = Tensor([1.5, -2.0, 3.0])
    b = Tensor([0.5, 4.0, -1.0])
    _check(lambda: (a * b + a).sum(), a, b)


def test_gradcheck_matmul_relu():
    a = Tensor([[1.0, -2.0], [3.0, 0.5]])
    b = Tensor([[2.0, 1.0], [-1.0, 3.0]])
    _check(lambda: (a @ b).relu().sum(), a, b)


def test_gradcheck_div_log_exp():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([2.0, 4.0, 1.0])
    _check(lambda: ((a / b).exp() + a.log()).sum(), a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_gradcheck.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.autograd.gradcheck'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/autograd/gradcheck.py`:

```python
import numpy as np


def numerical_grad(make_output, wrt, eps=1e-6):
    """Finite-difference gradient of sum(make_output().data) w.r.t. `wrt`.

    `make_output` is a function taking no arguments that builds and returns an
    output Tensor using `wrt`. We nudge each element of `wrt.data` up and down by
    `eps` and measure how the summed output changes -- the classic definition of
    a derivative. This is our independent check that backward() is correct.
    """
    grad = np.zeros_like(wrt.data)
    it = np.nditer(wrt.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = wrt.data[idx]

        wrt.data[idx] = original + eps
        plus = float(make_output().data.sum())

        wrt.data[idx] = original - eps
        minus = float(make_output().data.sum())

        wrt.data[idx] = original
        grad[idx] = (plus - minus) / (2.0 * eps)
        it.iternext()
    return grad
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_gradcheck.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add numerical gradient checking and verify autograd"
```

---

### Task 10: Prove it learns — end-to-end gradient descent

**Files:**
- Modify: `src/pyllm/autograd/tensor.py`
- Modify: `src/pyllm/autograd/__init__.py`
- Test: `tests/autograd/test_learning.py`

**Interfaces:**
- Consumes: all `Tensor` ops.
- Produces: `Tensor.zero_grad()` (resets `grad` to zeros so the same parameters can be reused across training steps); `pyllm.autograd` re-exports `Tensor`. Confirms the engine can actually drive a loss down via manual gradient descent.

- [ ] **Step 1: Write the failing test**

Create `tests/autograd/test_learning.py`:

```python
import numpy as np

from pyllm.autograd import Tensor


def test_zero_grad_resets():
    a = Tensor([1.0, 2.0])
    (a.sum()).backward()
    assert a.grad.tolist() == [1.0, 1.0]
    a.zero_grad()
    assert a.grad.tolist() == [0.0, 0.0]


def test_linear_regression_converges():
    # Learn y = 2x + 1 from data, using only the autograd engine.
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1.0, 1.0, size=(20, 1))
    ys = 2.0 * xs + 1.0

    w = Tensor([[0.0]])
    b = Tensor([0.0])
    x = Tensor(xs)
    target = Tensor(ys)

    lr = 0.1
    losses = []
    for _ in range(200):
        w.zero_grad()
        b.zero_grad()
        pred = x @ w + b
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        w.data -= lr * w.grad
        b.data -= lr * b.grad
        losses.append(float(loss.data))

    assert losses[-1] < 1e-3
    assert np.isclose(w.data[0, 0], 2.0, atol=0.05)
    assert np.isclose(b.data[0], 1.0, atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_learning.py -v`
Expected: FAIL — `ImportError: cannot import name 'Tensor' from 'pyllm.autograd'` (and `zero_grad` missing).

- [ ] **Step 3: Write minimal implementation**

Add to the `Tensor` class in `src/pyllm/autograd/tensor.py`:

```python
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
```

Replace the contents of `src/pyllm/autograd/__init__.py`:

```python
"""The autograd engine: tensors that remember how to compute their gradients."""

from pyllm.autograd.tensor import Tensor

__all__ = ["Tensor"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_learning.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full suite and tooling**

Run:
```bash
uv run pytest -v
uv run ruff check .
uv run pyright src tests
```
Expected: all tests pass; ruff clean; pyright 0 errors.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: add zero_grad and prove autograd learns via regression"
```

---

### Task 11: Child-friendly concept doc (RULE #1)

**Files:**
- Create: `docs/concepts/autograd.md`
- Test: `tests/test_docs.py`

**Interfaces:**
- Consumes: nothing (documentation).
- Produces: a standalone kid-friendly explanation of the autograd engine. The test only guards that the doc exists and covers the key ideas, so it can't be silently skipped.

- [ ] **Step 1: Write the failing test**

Create `tests/test_docs.py`:

```python
from pathlib import Path


def test_autograd_concept_doc_exists_and_covers_key_ideas():
    doc = Path("docs/concepts/autograd.md")
    assert doc.exists(), "RULE #1: every plan ships a kid-friendly concept doc"
    text = doc.read_text().lower()
    # The big ideas a child should walk away understanding.
    for idea in ["breadcrumb", "gradient", "backward", "analogy"]:
        assert idea in text, f"concept doc should explain '{idea}'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docs.py -v`
Expected: FAIL with `AssertionError: RULE #1: every plan ships a kid-friendly concept doc`.

- [ ] **Step 3: Write the concept doc**

Create `docs/concepts/autograd.md`:

```markdown
# The Breadcrumb Trail (Autograd)

> **Analogy:** Imagine walking through a forest and dropping a breadcrumb at
> every step. If you take a wrong turn, you can follow the breadcrumbs *backward*
> to find exactly where you went wrong — and fix it. That is exactly how our
> neural network learns. This trick is called **autograd** (automatic gradients).

## The problem

A neural network is just a big pile of numbers (we call them **weights**). When
it makes a guess, it's usually a bit wrong at first. To get better, it needs to
know: *"If I nudge this one number up a tiny bit, does my answer get better or
worse?"* That "does it get better or worse, and by how much" is called a
**gradient**.

A real network has *millions* of numbers. Working out every gradient by hand
would take forever. So we make the computer do it automatically.

## How the breadcrumbs work

Every time we do a piece of maths (add, multiply, etc.), our `Tensor` quietly
writes down:

1. **who its parents were** (which tensors went into it), and
2. **how to pass the blame backward** to those parents.

That "write down how to pass the blame backward" note is the breadcrumb.

When we finally measure how wrong we were (the **loss**), we call
`loss.backward()`. The computer follows every breadcrumb in reverse — from the
final answer all the way back to each weight — adding up exactly how much each
number was responsible for the mistake. That responsibility is stored in
`.grad`.

## A tiny worked example

```python
a = Tensor([2.0])
b = Tensor([3.0])
c = a * b        # c is 6, and remembers "a and b made me"
c.backward()     # follow the breadcrumbs back
# a.grad is 3  -> "if a goes up by 1, c goes up by 3" (because b = 3)
# b.grad is 2  -> "if b goes up by 1, c goes up by 2" (because a = 2)
```

## How do we *know* it's right?

We cheat-check it the slow, obvious way: nudge a number up a tiny bit, see how
much the answer changed, and compare. That's **numerical gradient checking**
(`numerical_grad`). If the fast breadcrumb answer matches the slow nudge answer,
we trust the breadcrumbs.

## Why this matters

This one idea — leaving a trail so you can walk back and learn from mistakes — is
the engine inside *every* modern AI, including the big ones. Everything else in
PyLLM (attention, transformers, the whole GPT) is built on top of these
breadcrumbs.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_docs.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: add kid-friendly autograd concept doc (the breadcrumb trail)"
```

---

## Self-Review

**Spec coverage (Plan 1 portion):** The autograd engine (spec §4 "Key design decisions", §7 "Numerical gradient-checking") is fully covered: Tensor wrapping numpy (Task 2), the operation set needed by every layer — add/mul/matmul/sum/mean/activations/pow/div (Tasks 3–8), gradient checking as the correctness oracle (Task 9), and an end-to-end learning proof (Task 10), and the RULE #1 kid-friendly concept doc enforced by a test (Task 11). numpy-as-backend-only and no-future-annotations constraints are honored throughout, and every class/function carries an analogy-first docstring. The remaining spec sections (nn layers, tokenizers, models, training, generation, Pebble flagship, PyStack integration, docs) are covered by Plans 2–5 in the roadmap.

**Placeholder scan:** No TBDs, no "add error handling", no "similar to Task N". Every code step shows complete code; every test step shows the full test.

**Type consistency:** `Tensor(data, _children=(), _op="")` constructor is used consistently by every op. `_unbroadcast(grad, shape)`, `numerical_grad(make_output, wrt, eps)`, `backward()`, and `zero_grad()` signatures match across their definition and call sites. `softmax`/`cross_entropy` are intentionally deferred to Plan 2 (they belong with the nn building blocks), not referenced here.
