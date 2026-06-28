# PyLLM Plan 2 — Neural-Net Building Blocks + Tokenizers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the neural-network layers (Linear, Embedding, LayerNorm, Dropout, softmax, cross-entropy, gelu, self-attention, multi-head attention, the Transformer block) and the tokenizers (char-level + from-scratch BPE) on top of Plan 1's autograd `Tensor`, each unit-tested.

**Architecture:** Everything is built on `pyllm.autograd.Tensor`. Layers subclass a tiny `Module` base that auto-collects parameters. Operations that need a custom backward but aren't single Tensor primitives (softmax, cross-entropy, embedding lookup, concat) are written as *functionals* that construct an autograd node directly (make an output `Tensor`, set its `_backward`). One new Tensor primitive — `transpose` — is added for attention. numpy is the math backend; every gradient is still ours.

**Tech Stack:** Python 3.14, numpy, pytest, ruff, pyright, managed with `uv`. Builds on branch from Plan 1 (merged to `main`).

## Global Constraints

- **RULE #1 — Child-friendly docs about EVERYTHING.** Every public class/function gets an analogy-first docstring (analogy before jargon). This plan is not done until its concept docs (`docs/concepts/tokens.md`, `embeddings.md`, `attention.md`) exist. This rule outranks all others.
- **Python 3.14**, managed with `uv`. Run everything via `uv run`.
- **Dependencies:** `numpy` only (plus dev/test/docs tooling). Nothing else.
- **TDD:** write the failing test first, watch it fail, then implement.
- **ruff** + **pyright** must stay clean after every task.
- **No `TYPE_CHECKING`, no `from __future__ import annotations`.**
- **Learnable weights are `Tensor` attributes; constant buffers (e.g. attention masks) are plain numpy arrays** — so `Module.parameters()` never trains a buffer.
- **Determinism:** layers that initialize or sample randomly accept an optional `rng` (a `numpy.random.Generator`); tests pass a seeded `np.random.default_rng(0)`.
- Frequent commits: one per task minimum.

## Plan 1 interfaces this plan consumes

- `from pyllm.autograd import Tensor` — ops: `+ - * / ** @`, `.sum(axis=None, keepdims=False)`, `.mean(axis=None, keepdims=False)`, `.relu() .exp() .log() .tanh()`, `.backward()`, `.zero_grad()`, `.shape`, `.data` (numpy float64, a copy), `.grad`.
- `from pyllm.autograd.gradcheck import numerical_grad` — `numerical_grad(make_output, wrt, eps=1e-6)` returns the finite-difference gradient of `sum(make_output().data)` w.r.t. `wrt`. Used to verify every new backward.

## File structure

```
src/pyllm/
  autograd/tensor.py        MODIFY: add Tensor.transpose()
  nn/
    __init__.py             re-exports the public layer/functional API
    module.py               Module base class (parameter collection)
    functional.py           softmax, cross_entropy, gelu, embedding, concat
    linear.py               Linear
    embedding.py            Embedding
    normalization.py        LayerNorm
    dropout.py              Dropout
    attention.py            Head, MultiHeadAttention
    transformer.py          FeedForward, TransformerBlock
  tokenizer/
    __init__.py             re-exports CharTokenizer, BPETokenizer
    char.py                 CharTokenizer
    bpe.py                  BPETokenizer
tests/
  autograd/test_transpose.py
  nn/test_module.py test_linear.py test_embedding.py test_functional.py
     test_normalization.py test_dropout.py test_attention.py test_transformer.py
  tokenizer/test_char.py test_bpe.py
  test_docs.py              MODIFY: guard the three new concept docs
docs/concepts/tokens.md embeddings.md attention.md
```

**Gradient-check pattern used throughout the nn tests:** every backward is verified
against finite differences. Each such test defines a local closure `make_output()`
that rebuilds the expression, calls `out.backward()` once, then asserts
`np.allclose(param.grad, numerical_grad(make_output, param), atol=1e-4)` for each
input/parameter. The full pattern is written out inline in each task's test — do
not factor it into a shared module; tests stay self-contained.

---

### Task 1: Tensor.transpose() primitive

**Files:**
- Modify: `src/pyllm/autograd/tensor.py`
- Test: `tests/autograd/test_transpose.py`

**Interfaces:**
- Consumes: `Tensor`.
- Produces: `Tensor.transpose()` — swaps the last two axes (`swapaxes(-1, -2)`), with backward that swaps them back. Needed by attention to compute `q @ k.transpose()`.

- [ ] **Step 1: Write the failing test**

Create `tests/autograd/test_transpose.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad


def test_transpose_forward_swaps_last_two_axes():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    assert t.transpose().shape == (3, 2)
    assert t.transpose().data.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_transpose_batched():
    t = Tensor(np.ones((4, 2, 3)))  # (B, T, D)
    assert t.transpose().shape == (4, 3, 2)


def test_transpose_backward():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = (a.transpose() * 2.0).sum()
    out.backward()
    approx = numerical_grad(lambda: (a.transpose() * 2.0).sum(), a)
    assert np.allclose(a.grad, approx, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_transpose.py -v`
Expected: FAIL with `AttributeError: 'Tensor' object has no attribute 'transpose'`.

- [ ] **Step 3: Write minimal implementation**

Add this method to the `Tensor` class in `src/pyllm/autograd/tensor.py`:

```python
    def transpose(self):
        """Flip a table on its side — swap the last two axes (rows <-> columns).

        Like turning a class register so the names run across the top instead of
        down the side. Attention uses this to line up "questions" against
        "keys". Breadcrumb rule: gradient just gets flipped back the same way.
        """
        out = Tensor(self.data.swapaxes(-1, -2), (self,), "transpose")

        def _backward():
            self.grad += out.grad.swapaxes(-1, -2)

        out._backward = _backward
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_transpose.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tensor.transpose() primitive for attention"
```

---

### Task 2: Module base class

**Files:**
- Create: `src/pyllm/nn/__init__.py`
- Create: `src/pyllm/nn/module.py`
- Test: `tests/nn/test_module.py`

**Interfaces:**
- Consumes: `Tensor`.
- Produces: `Module` with `parameters()` (recursively collects `Tensor` attributes, sub-`Module` attributes, and `Tensor`/`Module` items inside list/tuple attributes), `zero_grad()` (zeroes every parameter), and `__call__(*args, **kwargs)` delegating to `self.forward(...)`. Subclasses define `forward`.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_module.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.module import Module


class _Tiny(Module):
    def __init__(self):
        self.w = Tensor([1.0, 2.0])
        self.b = Tensor([0.0])
        self.not_a_param = 42

    def forward(self, x):
        return x + self.b


class _Nested(Module):
    def __init__(self):
        self.layer = _Tiny()
        self.heads = [_Tiny(), _Tiny()]


def test_parameters_collects_only_tensors():
    params = _Tiny().parameters()
    assert len(params) == 2
    assert all(isinstance(p, Tensor) for p in params)


def test_parameters_recurses_into_submodules_and_lists():
    # 1 submodule (2) + 2 heads (2 each) = 6 tensors
    assert len(_Nested().parameters()) == 6


def test_zero_grad_zeros_all_params():
    m = _Tiny()
    for p in m.parameters():
        p.grad += 5.0
    m.zero_grad()
    assert all(np.all(p.grad == 0.0) for p in m.parameters())


def test_call_delegates_to_forward():
    m = _Tiny()
    out = m(Tensor([10.0]))
    assert out.data.tolist() == [10.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_module.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.nn'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/nn/__init__.py`:

```python
"""Neural-net building blocks, all built on the autograd Tensor."""
```

Create `src/pyllm/nn/module.py`:

```python
from pyllm.autograd import Tensor


class Module:
    """A reusable Lego brick of a neural network.

    A Module holds some learnable numbers (its *parameters*) and knows how to
    turn an input into an output (`forward`). Bricks can hold other bricks, so a
    whole network is just one big brick made of smaller ones. `parameters()`
    walks the whole tower and hands back every learnable number so the optimizer
    can nudge them all.
    """

    def parameters(self):
        """Collect every learnable Tensor in this module and its children."""
        found = []
        for value in self.__dict__.values():
            if isinstance(value, Tensor):
                found.append(value)
            elif isinstance(value, Module):
                found.extend(value.parameters())
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, Module):
                        found.extend(item.parameters())
                    elif isinstance(item, Tensor):
                        found.append(item)
        return found

    def zero_grad(self):
        """Wipe the slate clean before the next learning step."""
        for param in self.parameters():
            param.zero_grad()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_module.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add nn.Module base class with parameter collection"
```

---

### Task 3: Linear layer

**Files:**
- Create: `src/pyllm/nn/linear.py`
- Test: `tests/nn/test_linear.py`

**Interfaces:**
- Consumes: `Tensor`, `Module`.
- Produces: `Linear(in_features, out_features, bias=True, rng=None)` with `.weight` Tensor `(in_features, out_features)` and `.bias` Tensor `(out_features,)` or `None`. `forward(x)` returns `x @ weight (+ bias)`. Works on `(in_features,)`, `(N, in_features)`, and batched `(B, T, in_features)` inputs (relies on Plan 1's batched matmul + broadcast bias). Weights initialized uniform in `±1/sqrt(in_features)`; bias zeros.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_linear.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad
from pyllm.nn.linear import Linear


def test_linear_output_shape_batched():
    layer = Linear(4, 3, rng=np.random.default_rng(0))
    x = Tensor(np.ones((2, 5, 4)))  # (B, T, in)
    assert layer(x).shape == (2, 5, 3)


def test_linear_has_two_parameters():
    layer = Linear(4, 3, rng=np.random.default_rng(0))
    assert len(layer.parameters()) == 2  # weight + bias


def test_linear_no_bias():
    layer = Linear(4, 3, bias=False, rng=np.random.default_rng(0))
    assert layer.bias is None
    assert len(layer.parameters()) == 1


def test_linear_gradients_check():
    layer = Linear(3, 2, rng=np.random.default_rng(0))
    x = Tensor([[1.0, -2.0, 0.5]])

    def make_output():
        return layer(x).sum()

    out = make_output()
    out.backward()
    for p in (layer.weight, layer.bias, x):
        assert np.allclose(p.grad, numerical_grad(make_output, p), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_linear.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.nn.linear'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/nn/linear.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_linear.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add nn.Linear layer"
```

---

### Task 4: softmax and embedding functionals

**Files:**
- Create: `src/pyllm/nn/functional.py`
- Test: `tests/nn/test_functional.py`

**Interfaces:**
- Consumes: `Tensor`.
- Produces:
  - `softmax(t, axis=-1)` — numerically stable (subtracts the max), returns a `Tensor` of probabilities summing to 1 along `axis`, with the correct softmax backward.
  - `embedding(weight, ids)` — `weight` is a `Tensor (num_embeddings, dim)`, `ids` is an int array of any shape `S`; returns a `Tensor` of shape `S + (dim,)` (the looked-up rows), with backward that scatter-adds gradient back into the used rows.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_functional.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad
from pyllm.nn.functional import embedding, softmax


def test_softmax_rows_sum_to_one():
    out = softmax(Tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]]), axis=-1)
    assert np.allclose(out.data.sum(axis=-1), [1.0, 1.0])


def test_softmax_is_stable_for_large_values():
    out = softmax(Tensor([[1000.0, 1000.0]]), axis=-1)
    assert np.allclose(out.data, [[0.5, 0.5]])  # no overflow


def test_softmax_gradients_check():
    a = Tensor([[0.5, -1.0, 2.0]])

    def make_output():
        # weight the probabilities so the gradient is non-trivial
        return (softmax(a, axis=-1) * Tensor([[1.0, 2.0, 3.0]])).sum()

    out = make_output()
    out.backward()
    assert np.allclose(a.grad, numerical_grad(make_output, a), atol=1e-4)


def test_embedding_looks_up_rows():
    weight = Tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    out = embedding(weight, np.array([2, 0, 1]))
    assert out.data.tolist() == [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]


def test_embedding_backward_scatters_to_used_rows():
    weight = Tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    out = embedding(weight, np.array([0, 0, 2]))  # row 0 used twice
    out.sum().backward()
    # row 0 gets gradient 2 (used twice), row 1 zero, row 2 one
    assert weight.grad.tolist() == [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_functional.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.nn.functional'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/nn/functional.py`:

```python
import numpy as np

from pyllm.autograd import Tensor


def softmax(t, axis=-1):
    """Turn a row of scores into a row of probabilities that add up to 1.

    Like sharing a cake by how much each person shouted for it: louder scores get
    a bigger slice, but every slice is positive and the whole cake is shared. We
    subtract the biggest score first so the numbers never blow up (`exp` of a
    huge number overflows). Breadcrumb rule: each probability nudges its score and
    gently pushes the others down (that's the `probs * (g - sum(g*probs))` line).
    """
    shifted = t.data - t.data.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=axis, keepdims=True)
    out = Tensor(probs, (t,), "softmax")

    def _backward():
        weighted = (out.grad * probs).sum(axis=axis, keepdims=True)
        t.grad += probs * (out.grad - weighted)

    out._backward = _backward
    return out


def embedding(weight, ids):
    """Look up a row of numbers for each id — a dictionary from id to vector.

    `weight` is a table with one row per possible token. `embedding` just grabs
    the rows named by `ids` (like looking up several words in a picture
    dictionary). Breadcrumb rule: when learning, every place a row was used adds
    its blame back onto that one row (`np.add.at` does the adding-up).
    """
    ids = np.asarray(ids)
    dim = weight.data.shape[1]
    out = Tensor(weight.data[ids], (weight,), "embedding")

    def _backward():
        grad = np.zeros_like(weight.data)
        np.add.at(grad, ids.reshape(-1), out.grad.reshape(-1, dim))
        weight.grad += grad

    out._backward = _backward
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_functional.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add softmax and embedding functionals"
```

---

### Task 5: cross_entropy functional

**Files:**
- Modify: `src/pyllm/nn/functional.py`
- Test: `tests/nn/test_functional.py`

**Interfaces:**
- Consumes: `Tensor`.
- Produces: `cross_entropy(logits, targets)` — `logits` is a `Tensor` whose last axis is the vocabulary (shape `(..., V)`); `targets` is an int array of the leading shape `(...)`. Returns a scalar `Tensor` (the mean negative-log-likelihood). Numerically stable (log-sum-exp); backward is `(softmax - one_hot) / N`.

- [ ] **Step 1: Write the failing test**

Add to `tests/nn/test_functional.py`:

```python
from pyllm.nn.functional import cross_entropy


def test_cross_entropy_perfect_prediction_is_near_zero():
    # huge logit on the correct class -> loss ~ 0
    logits = Tensor([[100.0, 0.0, 0.0]])
    loss = cross_entropy(logits, np.array([0]))
    assert loss.data < 1e-3


def test_cross_entropy_uniform_logits_equals_log_v():
    logits = Tensor([[0.0, 0.0, 0.0, 0.0]])  # V = 4
    loss = cross_entropy(logits, np.array([2]))
    assert np.isclose(loss.data, np.log(4))


def test_cross_entropy_gradients_check():
    logits = Tensor([[0.5, -1.0, 2.0], [1.0, 0.0, -0.5]])
    targets = np.array([2, 0])

    def make_output():
        return cross_entropy(logits, targets)

    out = make_output()
    out.backward()
    assert np.allclose(logits.grad, numerical_grad(make_output, logits), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_functional.py -k cross_entropy -v`
Expected: FAIL with `ImportError: cannot import name 'cross_entropy'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/pyllm/nn/functional.py`:

```python
def cross_entropy(logits, targets):
    """Score how surprised the model was by the right answer (lower = better).

    For each example the model gives a score to every possible next token. We
    turn those into probabilities (softmax) and ask: "what probability did you
    give the *correct* token?" If it was confident and right, the surprise is
    near zero; if it was confident and wrong, the surprise is huge. We average
    the surprise over all examples. Breadcrumb rule: push the predicted
    probabilities toward the true answer -- gradient is `(softmax - one_hot) / N`.
    """
    targets = np.asarray(targets)
    flat_logits = logits.data.reshape(-1, logits.data.shape[-1])
    flat_targets = targets.reshape(-1)
    n = flat_logits.shape[0]

    shifted = flat_logits - flat_logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    correct = probs[np.arange(n), flat_targets]
    loss_value = -np.log(correct).mean()
    out = Tensor(loss_value, (logits,), "cross_entropy")

    def _backward():
        grad = probs.copy()
        grad[np.arange(n), flat_targets] -= 1.0
        grad = grad / n * out.grad
        logits.grad += grad.reshape(logits.data.shape)

    out._backward = _backward
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_functional.py -k cross_entropy -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add cross_entropy loss functional"
```

---

### Task 6: gelu activation functional

**Files:**
- Modify: `src/pyllm/nn/functional.py`
- Test: `tests/nn/test_functional.py`

**Interfaces:**
- Consumes: `Tensor` ops (built from `*`, `**`, `tanh`, `+`, so the gradient flows automatically — no manual backward).
- Produces: `gelu(t)` — the tanh approximation of GELU, a smooth activation used in Transformers.

- [ ] **Step 1: Write the failing test**

Add to `tests/nn/test_functional.py`:

```python
from pyllm.nn.functional import gelu


def test_gelu_zero_maps_to_zero():
    assert np.isclose(gelu(Tensor([0.0])).data[0], 0.0)


def test_gelu_is_close_to_identity_for_large_positive():
    assert np.isclose(gelu(Tensor([5.0])).data[0], 5.0, atol=1e-2)


def test_gelu_squashes_large_negative_toward_zero():
    assert abs(gelu(Tensor([-5.0])).data[0]) < 1e-2


def test_gelu_gradients_check():
    a = Tensor([-1.0, 0.3, 2.0])

    def make_output():
        return gelu(a).sum()

    out = make_output()
    out.backward()
    assert np.allclose(a.grad, numerical_grad(make_output, a), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_functional.py -k gelu -v`
Expected: FAIL with `ImportError: cannot import name 'gelu'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/pyllm/nn/functional.py`:

```python
def gelu(t):
    """A soft on/off switch for signals — like a dimmer, not a hard light switch.

    `relu` slams negatives to exactly zero. `gelu` is gentler: it lets most of a
    positive number through, fades negatives smoothly toward zero, and bends
    softly around zero. This smoothness helps Transformers learn. It's built
    purely from `*`, `**`, `+` and `tanh`, so its gradient comes for free from
    the autograd engine.
    """
    c = np.sqrt(2.0 / np.pi)
    inner = (t + (t ** 3) * 0.044715) * c
    return (t * 0.5) * (inner.tanh() + 1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_functional.py -k gelu -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add gelu activation functional"
```

---

### Task 7: concat functional

**Files:**
- Modify: `src/pyllm/nn/functional.py`
- Test: `tests/nn/test_functional.py`

**Interfaces:**
- Consumes: `Tensor`.
- Produces: `concat(tensors, axis=-1)` — joins a list of Tensors along `axis`; backward splits the gradient back to each piece by its size.

- [ ] **Step 1: Write the failing test**

Add to `tests/nn/test_functional.py`:

```python
from pyllm.nn.functional import concat


def test_concat_joins_along_last_axis():
    out = concat([Tensor([[1.0, 2.0]]), Tensor([[3.0]])], axis=-1)
    assert out.data.tolist() == [[1.0, 2.0, 3.0]]


def test_concat_backward_splits_gradient():
    a = Tensor([[1.0, 2.0]])
    b = Tensor([[3.0]])

    def make_output():
        return (concat([a, b], axis=-1) * Tensor([[10.0, 20.0, 30.0]])).sum()

    out = make_output()
    out.backward()
    assert np.allclose(a.grad, numerical_grad(make_output, a), atol=1e-4)
    assert np.allclose(b.grad, numerical_grad(make_output, b), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_functional.py -k concat -v`
Expected: FAIL with `ImportError: cannot import name 'concat'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/pyllm/nn/functional.py`:

```python
def concat(tensors, axis=-1):
    """Stick several tables side by side into one wider table.

    Multi-head attention gives each "head" its own small report; `concat` lays
    the reports next to each other to make one big report. Breadcrumb rule: when
    blame comes back, each head only gets the slice of blame that belongs to its
    own columns.
    """
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(data, tuple(tensors), "concat")

    def _backward():
        sizes = [t.data.shape[axis] for t in tensors]
        split_points = np.cumsum(sizes)[:-1]
        pieces = np.split(out.grad, split_points, axis=axis)
        for tensor, piece in zip(tensors, pieces):
            tensor.grad += piece

    out._backward = _backward
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_functional.py -k concat -v`
Expected: PASS (2 passed). Also run the whole functional file: `uv run pytest tests/nn/test_functional.py -v` → all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add concat functional"
```

---

### Task 8: LayerNorm

**Files:**
- Create: `src/pyllm/nn/normalization.py`
- Test: `tests/nn/test_normalization.py`

**Interfaces:**
- Consumes: `Tensor` ops (`mean`, `-`, `**`, `/`, `*`, `+`), `Module`.
- Produces: `LayerNorm(dim, eps=1e-5)` with learnable `.gamma` `(dim,)` (init ones) and `.beta` `(dim,)` (init zeros). `forward(x)` normalizes over the last axis to mean 0 / variance 1, then scales by gamma and shifts by beta. Built entirely from existing Tensor ops, so gradients are automatic.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_normalization.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad
from pyllm.nn.normalization import LayerNorm


def test_layernorm_outputs_zero_mean_unit_var():
    ln = LayerNorm(4)
    x = Tensor([[1.0, 2.0, 3.0, 10.0]])
    out = ln(x).data
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.var(axis=-1), 1.0, atol=1e-3)


def test_layernorm_has_two_parameters():
    assert len(LayerNorm(4).parameters()) == 2


def test_layernorm_gradients_check():
    ln = LayerNorm(3)
    x = Tensor([[1.0, -2.0, 0.5], [3.0, 0.0, 1.0]])

    def make_output():
        return ln(x).sum()

    out = make_output()
    out.backward()
    for p in (ln.gamma, ln.beta, x):
        assert np.allclose(p.grad, numerical_grad(make_output, p), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_normalization.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.nn.normalization'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/nn/normalization.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_normalization.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add nn.LayerNorm"
```

---

### Task 9: Dropout

**Files:**
- Create: `src/pyllm/nn/dropout.py`
- Test: `tests/nn/test_dropout.py`

**Interfaces:**
- Consumes: `Tensor`, `Module`.
- Produces: `Dropout(p=0.1, rng=None)` with attribute `.training` (default `True`). When training and `p > 0`, multiplies the input by a random keep-mask scaled by `1/(1-p)` (inverted dropout). When `not training` or `p == 0`, returns the input unchanged. Has no parameters.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_dropout.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.dropout import Dropout


def test_dropout_eval_is_identity():
    d = Dropout(p=0.5, rng=np.random.default_rng(0))
    d.training = False
    x = Tensor([1.0, 2.0, 3.0, 4.0])
    assert d(x).data.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_dropout_has_no_parameters():
    assert Dropout(p=0.5).parameters() == []


def test_dropout_train_zeros_some_and_scales_rest():
    d = Dropout(p=0.5, rng=np.random.default_rng(0))
    x = Tensor(np.ones(1000))
    out = d(x).data
    # kept entries are scaled to 1/(1-0.5) = 2.0; dropped are 0.0
    assert set(np.unique(out)).issubset({0.0, 2.0})
    # roughly half are kept (allow slack)
    assert 350 < np.count_nonzero(out) < 650


def test_dropout_preserves_expected_value_roughly():
    d = Dropout(p=0.2, rng=np.random.default_rng(1))
    x = Tensor(np.ones(10000))
    assert np.isclose(d(x).data.mean(), 1.0, atol=0.05)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_dropout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.nn.dropout'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/nn/dropout.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_dropout.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add nn.Dropout"
```

---

### Task 10: Embedding module

**Files:**
- Create: `src/pyllm/nn/embedding.py`
- Test: `tests/nn/test_embedding.py`

**Interfaces:**
- Consumes: `Tensor`, `Module`, `functional.embedding`.
- Produces: `Embedding(num_embeddings, dim, rng=None)` with learnable `.weight` `(num_embeddings, dim)` (init normal × 0.02). `forward(ids)` returns the looked-up rows via the `embedding` functional.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_embedding.py`:

```python
import numpy as np

from pyllm.nn.embedding import Embedding


def test_embedding_output_shape():
    emb = Embedding(10, 4, rng=np.random.default_rng(0))
    ids = np.array([[1, 2, 3], [4, 5, 6]])  # (B, T)
    assert emb(ids).shape == (2, 3, 4)


def test_embedding_has_one_parameter():
    emb = Embedding(10, 4, rng=np.random.default_rng(0))
    assert len(emb.parameters()) == 1
    assert emb.weight.shape == (10, 4)


def test_embedding_trains_only_used_rows():
    emb = Embedding(5, 2, rng=np.random.default_rng(0))
    emb(np.array([1, 1, 3])).sum().backward()
    grad = emb.weight.grad
    assert np.all(grad[0] == 0.0) and np.all(grad[2] == 0.0) and np.all(grad[4] == 0.0)
    assert np.all(grad[1] != 0.0) and np.all(grad[3] != 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_embedding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.nn.embedding'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/nn/embedding.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_embedding.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add nn.Embedding module"
```

---

### Task 11: Single-head causal self-attention

**Files:**
- Create: `src/pyllm/nn/attention.py`
- Test: `tests/nn/test_attention.py`

**Interfaces:**
- Consumes: `Tensor`, `Module`, `Linear`, `functional.softmax`, `Tensor.transpose`.
- Produces: `Head(embed_dim, head_size, block_size, rng=None)` with `.key`, `.query`, `.value` as bias-free `Linear(embed_dim, head_size)` and a numpy causal mask buffer `.mask` of shape `(block_size, block_size)` (0 on/below the diagonal, `-1e9` above). `forward(x)` for `x` shape `(B, T, embed_dim)` returns `(B, T, head_size)` using scaled dot-product attention with the causal mask applied before softmax.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_attention.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.attention import Head


def test_head_output_shape():
    head = Head(embed_dim=8, head_size=4, block_size=16, rng=np.random.default_rng(0))
    x = Tensor(np.ones((2, 5, 8)))  # (B, T, embed_dim)
    assert head(x).shape == (2, 5, 4)


def test_head_is_causal():
    # Changing the LAST time step must not change outputs at earlier steps.
    head = Head(embed_dim=8, head_size=4, block_size=16, rng=np.random.default_rng(0))
    rng = np.random.default_rng(1)
    base = rng.normal(size=(1, 5, 8))
    changed = base.copy()
    changed[0, -1, :] += 10.0  # disturb only the final position

    out_base = head(Tensor(base)).data
    out_changed = head(Tensor(changed)).data
    # earlier positions (0..3) are unchanged; only the last may differ
    assert np.allclose(out_base[0, :4], out_changed[0, :4], atol=1e-8)


def test_head_parameters_exclude_mask():
    head = Head(embed_dim=8, head_size=4, block_size=16, rng=np.random.default_rng(0))
    # 3 bias-free Linear layers, 1 weight each -> 3 params (mask is a buffer)
    assert len(head.parameters()) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_attention.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.nn.attention'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/nn/attention.py`:

```python
import numpy as np

from pyllm.nn.functional import softmax
from pyllm.nn.linear import Linear
from pyllm.nn.module import Module


class Head(Module):
    """One "attention" head: each token decides which earlier tokens to listen to.

    Reading a sentence, the word "it" looks back to find what "it" refers to.
    A head does that: every token makes a *query* ("what am I looking for?"),
    every token offers a *key* ("here's what I am"), and tokens whose keys match
    the query get listened to most. The *value* is the information actually
    passed along. A causal mask hides the future, so a token can only look
    backward -- you can't peek at words you haven't read yet.
    """

    def __init__(self, embed_dim, head_size, block_size, rng=None):
        self.key = Linear(embed_dim, head_size, bias=False, rng=rng)
        self.query = Linear(embed_dim, head_size, bias=False, rng=rng)
        self.value = Linear(embed_dim, head_size, bias=False, rng=rng)
        self.head_size = head_size
        # Buffer (NOT a parameter): -1e9 above the diagonal blocks the future.
        allowed = np.tril(np.ones((block_size, block_size)))
        self.mask = np.where(allowed == 0, -1e9, 0.0)

    def forward(self, x):
        seq_len = x.shape[1]
        q = self.query(x)            # (B, T, head_size)
        k = self.key(x)
        v = self.value(x)
        scores = (q @ k.transpose()) / np.sqrt(self.head_size)  # (B, T, T)
        scores = scores + self.mask[:seq_len, :seq_len]
        weights = softmax(scores, axis=-1)
        return weights @ v           # (B, T, head_size)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_attention.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add single-head causal self-attention"
```

---

### Task 12: Multi-head attention

**Files:**
- Modify: `src/pyllm/nn/attention.py`
- Test: `tests/nn/test_attention.py`

**Interfaces:**
- Consumes: `Head`, `Linear`, `functional.concat`, `Module`.
- Produces: `MultiHeadAttention(embed_dim, num_heads, block_size, rng=None)` (requires `embed_dim % num_heads == 0`); holds `.heads` (a list of `num_heads` `Head`s each of size `embed_dim // num_heads`) and an output projection `.proj = Linear(embed_dim, embed_dim)`. `forward(x)` runs every head, concatenates along the last axis, and projects back to `embed_dim`.

- [ ] **Step 1: Write the failing test**

Add to `tests/nn/test_attention.py`:

```python
from pyllm.nn.attention import MultiHeadAttention


def test_multihead_output_shape():
    mha = MultiHeadAttention(embed_dim=8, num_heads=2, block_size=16,
                             rng=np.random.default_rng(0))
    x = Tensor(np.ones((2, 5, 8)))
    assert mha(x).shape == (2, 5, 8)


def test_multihead_requires_divisible_dim():
    import pytest
    with pytest.raises(AssertionError):
        MultiHeadAttention(embed_dim=8, num_heads=3, block_size=16)


def test_multihead_collects_all_head_and_proj_params():
    mha = MultiHeadAttention(embed_dim=8, num_heads=2, block_size=16,
                             rng=np.random.default_rng(0))
    # 2 heads x 3 params + proj (weight + bias) = 6 + 2 = 8
    assert len(mha.parameters()) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_attention.py -k multihead -v`
Expected: FAIL with `ImportError: cannot import name 'MultiHeadAttention'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/pyllm/nn/attention.py`:

```python
from pyllm.nn.functional import concat


class MultiHeadAttention(Module):
    """Several attention heads working in parallel, then combined.

    One head can only track one kind of relationship at a time. Like a panel of
    readers where one watches for "who did what", another for "when", another for
    "where" -- each head looks for a different pattern. We run them all at once,
    lay their reports side by side (`concat`), and blend them with a final
    Linear so the network can mix what every head found.
    """

    def __init__(self, embed_dim, num_heads, block_size, rng=None):
        assert embed_dim % num_heads == 0, "embed_dim must divide evenly by num_heads"
        head_size = embed_dim // num_heads
        self.heads = [
            Head(embed_dim, head_size, block_size, rng=rng) for _ in range(num_heads)
        ]
        self.proj = Linear(embed_dim, embed_dim, rng=rng)

    def forward(self, x):
        combined = concat([head(x) for head in self.heads], axis=-1)
        return self.proj(combined)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_attention.py -k multihead -v`
Expected: PASS (3 passed). Then run the whole file: `uv run pytest tests/nn/test_attention.py -v`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add multi-head attention"
```

---

### Task 13: FeedForward + TransformerBlock

**Files:**
- Create: `src/pyllm/nn/transformer.py`
- Test: `tests/nn/test_transformer.py`

**Interfaces:**
- Consumes: `Linear`, `functional.gelu`, `LayerNorm`, `MultiHeadAttention`, `Module`.
- Produces:
  - `FeedForward(embed_dim, rng=None)` — `Linear(embed_dim, 4*embed_dim)` → `gelu` → `Linear(4*embed_dim, embed_dim)`.
  - `TransformerBlock(embed_dim, num_heads, block_size, rng=None)` — pre-norm residual block: `x = x + attn(ln1(x))`, then `x = x + ffn(ln2(x))`. Output shape equals input shape.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_transformer.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.nn.transformer import FeedForward, TransformerBlock


def test_feedforward_preserves_shape():
    ff = FeedForward(8, rng=np.random.default_rng(0))
    x = Tensor(np.ones((2, 5, 8)))
    assert ff(x).shape == (2, 5, 8)


def test_transformer_block_preserves_shape():
    block = TransformerBlock(embed_dim=8, num_heads=2, block_size=16,
                             rng=np.random.default_rng(0))
    x = Tensor(np.ones((2, 5, 8)))
    assert block(x).shape == (2, 5, 8)


def test_transformer_block_is_causal():
    block = TransformerBlock(embed_dim=8, num_heads=2, block_size=16,
                             rng=np.random.default_rng(0))
    rng = np.random.default_rng(2)
    base = rng.normal(size=(1, 6, 8))
    changed = base.copy()
    changed[0, -1, :] += 10.0
    out_base = block(Tensor(base)).data
    out_changed = block(Tensor(changed)).data
    assert np.allclose(out_base[0, :5], out_changed[0, :5], atol=1e-8)


def test_transformer_block_gradients_flow_to_input():
    block = TransformerBlock(embed_dim=8, num_heads=2, block_size=16,
                             rng=np.random.default_rng(0))
    x = Tensor(np.ones((1, 4, 8)))
    block(x).sum().backward()
    assert np.any(x.grad != 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_transformer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.nn.transformer'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/nn/transformer.py`:

```python
from pyllm.nn.attention import MultiHeadAttention
from pyllm.nn.functional import gelu
from pyllm.nn.linear import Linear
from pyllm.nn.module import Module
from pyllm.nn.normalization import LayerNorm


class FeedForward(Module):
    """A little "think it over" step applied to each position on its own.

    After attention gathers information from other tokens, each token does some
    private thinking: expand to a bigger scratch space (4x), apply a smooth
    activation (`gelu`), then shrink back. It's where a lot of the model's
    per-token reasoning happens.
    """

    def __init__(self, embed_dim, rng=None):
        self.fc1 = Linear(embed_dim, 4 * embed_dim, rng=rng)
        self.fc2 = Linear(4 * embed_dim, embed_dim, rng=rng)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class TransformerBlock(Module):
    """The repeatable Lego brick of a GPT: communicate, then think.

    Two steps, each wrapped in a *residual* connection (we add the step's result
    back onto the input, like keeping your original notes and only adding edits):
    1. tokens talk to each other (multi-head attention),
    2. each token thinks privately (feed-forward).
    LayerNorm tidies the numbers before each step. Stack many of these bricks and
    you get a real language model.
    """

    def __init__(self, embed_dim, num_heads, block_size, rng=None):
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size, rng=rng)
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, rng=rng)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_transformer.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add FeedForward and TransformerBlock"
```

---

### Task 14: nn package public API

**Files:**
- Modify: `src/pyllm/nn/__init__.py`
- Test: `tests/nn/test_public_api.py`

**Interfaces:**
- Consumes: all nn modules.
- Produces: `pyllm.nn` re-exports `Module`, `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `Head`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock`, and the functionals `softmax`, `cross_entropy`, `gelu`, `embedding`, `concat`.

- [ ] **Step 1: Write the failing test**

Create `tests/nn/test_public_api.py`:

```python
def test_public_api_exports():
    import pyllm.nn as nn

    expected = [
        "Module", "Linear", "Embedding", "LayerNorm", "Dropout",
        "Head", "MultiHeadAttention", "FeedForward", "TransformerBlock",
        "softmax", "cross_entropy", "gelu", "embedding", "concat",
    ]
    for name in expected:
        assert hasattr(nn, name), f"pyllm.nn is missing {name}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/nn/test_public_api.py -v`
Expected: FAIL with `AssertionError: pyllm.nn is missing Module`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/pyllm/nn/__init__.py`:

```python
"""Neural-net building blocks, all built on the autograd Tensor."""

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/nn/test_public_api.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: expose nn public API"
```

---

### Task 15: Char tokenizer

**Files:**
- Create: `src/pyllm/tokenizer/__init__.py`
- Create: `src/pyllm/tokenizer/char.py`
- Test: `tests/tokenizer/test_char.py`

**Interfaces:**
- Consumes: nothing (pure Python).
- Produces: `CharTokenizer(text)` — builds a sorted vocabulary of the unique characters in `text`. `.vocab_size` (int property), `.encode(text) -> list[int]`, `.decode(ids) -> str`. Encode/decode are exact inverses for any text using only known characters.

- [ ] **Step 1: Write the failing test**

Create `tests/tokenizer/test_char.py`:

```python
from pyllm.tokenizer.char import CharTokenizer


def test_vocab_size_counts_unique_chars():
    tok = CharTokenizer("hello")  # h, e, l, o
    assert tok.vocab_size == 4


def test_encode_decode_roundtrip():
    tok = CharTokenizer("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"


def test_encode_returns_ints_in_range():
    tok = CharTokenizer("abc")
    ids = tok.encode("cab")
    assert all(isinstance(i, int) for i in ids)
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_vocab_is_sorted_and_stable():
    # sorted unique chars -> 'a','b','c' => a=0, b=1, c=2
    tok = CharTokenizer("cba")
    assert tok.encode("abc") == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tokenizer/test_char.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.tokenizer'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/tokenizer/__init__.py`:

```python
"""Tokenizers: turn text into numbers the model can read, and back again."""
```

Create `src/pyllm/tokenizer/char.py`:

```python
class CharTokenizer:
    """The simplest possible tokenizer: one number per character.

    Imagine giving every letter a locker number: a=0, b=1, c=2... To "encode" a
    word you write down its locker numbers; to "decode" you read the letters back
    out. Simple and lossless, but the model has to spell everything out one
    letter at a time -- which is why we also build a smarter BPE tokenizer later.
    """

    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {char: index for index, char in enumerate(chars)}
        self.itos = {index: char for char, index in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        return [self.stoi[char] for char in text]

    def decode(self, ids):
        return "".join(self.itos[index] for index in ids)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tokenizer/test_char.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add CharTokenizer"
```

---

### Task 16: BPE tokenizer

**Files:**
- Create: `src/pyllm/tokenizer/bpe.py`
- Modify: `src/pyllm/tokenizer/__init__.py`
- Test: `tests/tokenizer/test_bpe.py`

**Interfaces:**
- Consumes: nothing (pure Python).
- Produces: `BPETokenizer()` with:
  - `.train(text, num_merges)` — starts from single characters and repeatedly merges the most frequent adjacent pair into a new token, `num_merges` times. Records merges in order.
  - `.vocab_size` (int property) — number of base chars + number of merges actually performed.
  - `.encode(text) -> list[int]` — splits into chars, then greedily applies the learned merges in the order they were learned.
  - `.decode(ids) -> str` — concatenates the token strings.
  Encode/decode roundtrip exactly; after training, encoding the training text yields fewer tokens than the character count.

- [ ] **Step 1: Write the failing test**

Create `tests/tokenizer/test_bpe.py`:

```python
from pyllm.tokenizer.bpe import BPETokenizer


def test_bpe_roundtrip():
    tok = BPETokenizer()
    tok.train("ababab abc", num_merges=3)
    assert tok.decode(tok.encode("ababab")) == "ababab"


def test_bpe_merges_reduce_token_count():
    text = "abababababab"
    tok = BPETokenizer()
    tok.train(text, num_merges=2)
    # "ab" should merge, so encoding is much shorter than 12 chars
    assert len(tok.encode(text)) < len(text)


def test_bpe_vocab_grows_by_merges():
    text = "ababab"  # base chars: a, b  => 2
    tok = BPETokenizer()
    tok.train(text, num_merges=1)
    assert tok.vocab_size == 3  # a, b, and merged "ab"


def test_bpe_learns_most_frequent_pair_first():
    tok = BPETokenizer()
    tok.train("ababab", num_merges=1)
    # the first (and only) merge should be the pair ('a', 'b')
    assert tok.merges[0] == ("a", "b")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tokenizer/test_bpe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.tokenizer.bpe'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/tokenizer/bpe.py`:

```python
from collections import Counter


class BPETokenizer:
    """A smarter tokenizer that learns common chunks, like "th" or "ing".

    Reading letter-by-letter is slow. People read in chunks. BPE (Byte Pair
    Encoding) learns chunks automatically: look at the text, find the two
    neighbours that appear together most often, glue them into one new token,
    and repeat. After a while common pieces like "ab" or "the" become single
    tokens, so the model reads in bigger, smarter gulps.
    """

    def __init__(self):
        # merges[i] = the (left, right) pair learned at step i, in order.
        self.merges = []
        self.stoi = {}
        self.itos = {}

    def _rebuild_vocab(self, base_chars):
        tokens = list(base_chars)
        for left, right in self.merges:
            tokens.append(left + right)
        self.stoi = {token: index for index, token in enumerate(tokens)}
        self.itos = {index: token for token, index in self.stoi.items()}

    def train(self, text, num_merges):
        base_chars = sorted(set(text))
        symbols = list(text)  # each element is a token string, starting as chars
        self.merges = []
        for _ in range(num_merges):
            pairs = Counter(zip(symbols, symbols[1:]))
            if not pairs:
                break
            (left, right), count = pairs.most_common(1)[0]
            if count < 2:
                break  # nothing worth merging
            self.merges.append((left, right))
            symbols = _merge_pair(symbols, left, right)
        self._rebuild_vocab(base_chars)

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        symbols = list(text)
        for left, right in self.merges:
            symbols = _merge_pair(symbols, left, right)
        return [self.stoi[symbol] for symbol in symbols]

    def decode(self, ids):
        return "".join(self.itos[index] for index in ids)


def _merge_pair(symbols, left, right):
    """Walk the list and glue every adjacent (left, right) into one token."""
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
            merged.append(left + right)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged
```

Update `src/pyllm/tokenizer/__init__.py`:

```python
"""Tokenizers: turn text into numbers the model can read, and back again."""

from pyllm.tokenizer.bpe import BPETokenizer
from pyllm.tokenizer.char import CharTokenizer

__all__ = ["CharTokenizer", "BPETokenizer"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/tokenizer/test_bpe.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add from-scratch BPE tokenizer"
```

---

### Task 17: Child-friendly concept docs (RULE #1)

**Files:**
- Create: `docs/concepts/tokens.md`
- Create: `docs/concepts/embeddings.md`
- Create: `docs/concepts/attention.md`
- Modify: `tests/test_docs.py`

**Interfaces:**
- Consumes: nothing (documentation).
- Produces: three standalone kid-friendly concept docs, guarded by tests so they can't be skipped.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_docs.py`:

```python
def test_plan2_concept_docs_exist_and_cover_key_ideas():
    from pathlib import Path

    checks = {
        "docs/concepts/tokens.md": ["token", "chunk", "analogy"],
        "docs/concepts/embeddings.md": ["embedding", "meaning", "analogy"],
        "docs/concepts/attention.md": ["attention", "look back", "analogy"],
    }
    for path, ideas in checks.items():
        doc = Path(path)
        assert doc.exists(), f"RULE #1: missing concept doc {path}"
        text = doc.read_text().lower()
        for idea in ideas:
            assert idea in text, f"{path} should explain '{idea}'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docs.py::test_plan2_concept_docs_exist_and_cover_key_ideas -v`
Expected: FAIL with `AssertionError: RULE #1: missing concept doc docs/concepts/tokens.md`.

- [ ] **Step 3: Write the concept docs**

Create `docs/concepts/tokens.md`:

```markdown
# What Is a Token? (Chopping Text into Lego Bricks)

> **Analogy:** A model can't read letters the way you do. First we chop text into
> small pieces called **tokens** -- like snapping a sentence into Lego bricks --
> and give every brick a number. The model only ever sees the numbers.

## Two ways to chop

**Character tokens** (`CharTokenizer`): every single letter is its own brick.
`"cat"` becomes `c, a, t` -> `[2, 0, 19]`. Simple and never fails, but the model
has to spell out everything one letter at a time.

**BPE tokens** (`BPETokenizer`): we let the computer *learn* common chunks. It
keeps gluing together the two neighbours that show up together most often, so
pieces like `"th"`, `"ing"`, or `"ab"` become single bricks. Now the model reads
in bigger, smarter gulps.

## Encode and decode

- **Encode** = text -> numbers (chop into bricks, write down their numbers).
- **Decode** = numbers -> text (look the bricks back up and glue them together).

They are perfect opposites: `decode(encode("hello")) == "hello"`.

## Why it matters

Tokens are the model's alphabet. Every other part of PyLLM -- embeddings,
attention, the whole GPT -- works on these token numbers, never on raw letters.
```

Create `docs/concepts/embeddings.md`:

```markdown
# What Is an Embedding? (A Map of Meaning)

> **Analogy:** An embedding gives every token its own spot on a giant "map of
> meaning". Tokens that mean similar things end up close together, the way
> similar towns sit near each other on a real map.

## From a number to a vector

A token id like `42` is just a label -- it tells us nothing about meaning. An
`Embedding` is a big lookup table with one row of numbers per token. We swap each
id for its row (its **vector**). That little list of numbers is the token's
position on the map.

```
id 42  -->  [ 0.13, -0.88, 0.42, ... ]
```

## The map is *learned*

At the start every row is random -- the map is nonsense. As the model trains,
tokens that get used in similar ways drift toward each other. Nobody tells the
model "cat and dog are similar"; it discovers it, and the map arranges itself.

## Why it matters

Only the rows the model actually uses get nudged when learning (look at
`embedding`'s breadcrumb rule). Embeddings turn cold id numbers into rich
vectors that attention and the Transformer can actually reason about.
```

Create `docs/concepts/attention.md`:

```markdown
# Attention (Re-reading a Sentence to Understand It)

> **Analogy:** When you read "The trophy didn't fit in the case because *it* was
> too big," you instantly **look back** to decide what "it" means. Attention is
> the model doing exactly that: each word looks back at earlier words and decides
> which ones matter right now.

## Query, key, value

Every token produces three things:
- a **query** -- "what am I looking for?"
- a **key** -- "here's what I am, in case you're looking for me",
- a **value** -- "here's the information I'll hand over if you pick me".

A token compares its query against every earlier token's key. Good matches get a
high score; we turn the scores into shares with **softmax** and use them to take
a weighted blend of the **values**. That blend is what the token learned by
looking back.

## Looking back only (the causal mask)

A token may only look at itself and tokens *before* it -- you can't use words you
haven't read yet. We enforce this with a **causal mask** that blocks the future
(it adds a huge negative number to future scores so softmax gives them ~0 share).

## Many heads

One head tracks one kind of relationship. **Multi-head attention** runs several
in parallel -- one might follow "who did what", another "when" -- then blends
their reports. Stack attention + a little per-token thinking (feed-forward) and
you have a **Transformer block**, the repeatable brick of a GPT.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_docs.py -v`
Expected: PASS (all doc tests, including Plan 1's).

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
git commit -m "docs: add kid-friendly concept docs for tokens, embeddings, attention"
```

---

## Self-Review

**Spec coverage (Plan 2 portion of the design spec):** Covers spec §4's `nn/` (Module, Linear, Embedding, LayerNorm, Dropout, softmax, cross-entropy, gelu, self-attention head, multi-head attention, TransformerBlock — Tasks 2–14) and `tokenizer/` (char + BPE — Tasks 15–16). `Tensor.transpose` (Task 1) is the one new autograd primitive the attention math needs. RULE #1 concept docs for tokens/embeddings/attention land in Task 17, guarded by a test. The optimizer, the model ladder, training loop, and generation are intentionally deferred to Plan 3; corpora and checkpoints to Plans 3–4.

**Placeholder scan:** No TBDs, no "add error handling", no "similar to Task N". Every code step shows complete code; every test step shows the full test. The `_gradcheck`/`make_output` pattern is written out inline in each test that uses it.

**Type consistency:** `Module.parameters()`/`zero_grad()`/`__call__` are used consistently by every layer. `Linear(in_features, out_features, bias=True, rng=None)` with `.weight`/`.bias`; `Embedding(num_embeddings, dim, rng=None)` with `.weight`; `Head(embed_dim, head_size, block_size, rng=None)`; `MultiHeadAttention(embed_dim, num_heads, block_size, rng=None)` with `.heads`/`.proj`; `TransformerBlock(embed_dim, num_heads, block_size, rng=None)`. Functionals `softmax(t, axis=-1)`, `cross_entropy(logits, targets)`, `gelu(t)`, `embedding(weight, ids)`, `concat(tensors, axis=-1)` match between definition (Tasks 4–7) and every call site (Tasks 8–13). `CharTokenizer(text)` / `BPETokenizer().train(text, num_merges)` with `.vocab_size`/`.encode`/`.decode` are consistent between Tasks 15–16 and their tests. The new `Tensor.transpose()` (Task 1) is the only Tensor signature consumed by attention (Task 11) and matches.
