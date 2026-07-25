# PyLLM Plan 3 — Model Ladder + Training + Generation (Pokémon end-to-end) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the Plan 2 nn building blocks into three trainable language models (`Bigram`, `MLP`, `GPT`), add a from-scratch optimizer (SGD + Adam), batching, and a training loop; add sampling (greedy/temperature/top-k) and checkpoint save/load; ship a bundled Pokémon corpus + a tiny pre-trained checkpoint that generates the moment you launch the `pyllm` CLI.

**Architecture:** Every model exposes `forward(idx) -> logits` and a `.mode` attribute (`"sequence"` for Bigram/GPT which predict a next token at *every* position, `"single"` for the MLP which predicts *one* next token from a fixed window). One `get_batch` produces `(x, y)` in either shape; one `train` loop drives any model with `cross_entropy` (which already flattens leading dims). Generation walks one token at a time off the model's `.data` (no gradients). Checkpoints are numpy `.npz` files holding each parameter plus a small JSON config + the tokenizer's vocab, so `load()` rebuilds the model with `build_model(config)` and reassigns weights in `parameters()` order.

**Tech Stack:** Python 3.14, numpy, pytest, ruff, pyright, managed with `uv`. Builds on Plan 2 (`main`).

## Global Constraints

- **RULE #1 — Child-friendly docs about EVERYTHING.** Every public class/function gets an analogy-first docstring (analogy before jargon). This plan is not done until its concept docs (`docs/concepts/how-a-model-learns.md`, `sampling.md`, `transformer.md`) exist. This rule outranks all others.
- **Python 3.14**, managed with `uv`. Run everything via `uv run`.
- **Dependencies:** `numpy` only (plus dev/test/docs tooling). Nothing else.
- **TDD:** write the failing test first, watch it fail, then implement.
- **ruff** + **pyright** must stay clean after every task.
- **No `TYPE_CHECKING`, no `from __future__ import annotations`.**
- **Learnable weights are `Tensor` attributes; constant buffers (masks, position ids) are plain numpy** — so `Module.parameters()` never trains a buffer.
- **Determinism:** anything that initializes or samples randomly accepts an optional `rng` (`numpy.random.Generator`); tests pass a seeded `np.random.default_rng(0)`.
- Frequent commits: one per task minimum.

## Plan 1 + Plan 2 interfaces this plan consumes

- `from pyllm.autograd import Tensor` — ops `+ - * / ** @`, `.sum/.mean(axis, keepdims)`, `.relu/.exp/.log/.tanh/.transpose()`, `.backward()`, `.zero_grad()`, `.shape`, `.data` (numpy float64), `.grad`.
- `from pyllm.nn import Module, Linear, Embedding, LayerNorm, Dropout, MultiHeadAttention, TransformerBlock, softmax, cross_entropy, gelu, embedding, concat`.
  - `cross_entropy(logits, targets)` accepts logits `(..., V)` and int targets of the leading shape `(...)`, returns a scalar mean-NLL `Tensor`. Works for both `(B, T, V)` vs `(B, T)` and `(B, V)` vs `(B,)`.
  - `embedding(weight, ids)` — `weight` is a `Tensor (num, dim)`, `ids` an int array of shape `S`; returns `Tensor` of shape `S + (dim,)`.
  - `Module.parameters()` dedupes by id; `Module.train(mode=True)` / `Module.eval()` recurse and set `.training`.
- `from pyllm.tokenizer import CharTokenizer` — `CharTokenizer(text)` builds a **sorted-unique-char** vocab; `.vocab_size`, `.encode(text) -> list[int]`, `.decode(ids) -> str`. Because the vocab is `sorted(set(text))`, feeding the joined sorted vocab back into `CharTokenizer` reproduces identical `stoi`/`itos` — this is how checkpoints reload the tokenizer.

## File structure

```
src/pyllm/
  autograd/tensor.py        MODIFY: add Tensor.reshape()
  models/
    __init__.py             re-exports Bigram, MLP, GPT, build_model
    bigram.py               Bigram
    mlp.py                  MLP
    gpt.py                  GPT
  training/
    __init__.py             re-exports SGD, Adam, get_batch, train
    optim.py                SGD, Adam
    data.py                 get_batch
    loop.py                 train
  generate.py               sample_next, generate
  checkpoint.py             save, load
  data/
    __init__.py             load_corpus (reads bundled .txt), CHECKPOINT paths
    pokemon_corpus.txt      bundled corpus (committed)
    pokemon.npz             tiny trained checkpoint (committed, built in Task 15)
  cli.py                    main(): REPL + train + tokenize
scripts/
  train_pokemon.py          build-time: trains + writes src/pyllm/data/pokemon.npz
tests/
  models/test_bigram.py test_mlp.py test_gpt.py
  training/test_optim.py test_data.py test_loop.py
  test_generate.py test_checkpoint.py test_overfit.py test_cli.py
  test_docs.py              MODIFY: guard the three new concept docs
docs/concepts/how-a-model-learns.md sampling.md transformer.md
pyproject.toml              MODIFY: [project.scripts] + bundle data artifacts
```

**Model contract (shared by all three):**
- `model(idx)` where `idx` is an int numpy array; returns a logits `Tensor`.
- `.mode`: `"sequence"` → `idx (B, T)`, logits `(B, T, V)`; `"single"` → `idx (B, block_size)`, logits `(B, V)`.
- `.block_size`: int. Generation crops context to the last `block_size` tokens; batching draws windows of this width.
- `.vocab_size`: int.
- `.config()` → a plain dict with a `"kind"` key + hyperparameters, enough for `build_model` to rebuild it.

---

### Task 1: Tensor.reshape() primitive

**Files:**
- Modify: `src/pyllm/autograd/tensor.py`
- Test: `tests/autograd/test_reshape.py`

**Interfaces:**
- Consumes: `Tensor`.
- Produces: `Tensor.reshape(shape)` — returns a Tensor viewing the same data in a new shape; backward reshapes the gradient back. Needed by the MLP to flatten `(B, block_size, embed_dim)` → `(B, block_size*embed_dim)`.

- [ ] **Step 1: Write the failing test**

Create `tests/autograd/test_reshape.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad


def test_reshape_forward_changes_shape_not_data():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
    out = t.reshape((3, 2))
    assert out.shape == (3, 2)
    assert out.data.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_reshape_flattens_last_two_axes():
    t = Tensor(np.ones((4, 2, 3)))  # (B, T, D)
    assert t.reshape((4, 6)).shape == (4, 6)


def test_reshape_backward():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def make_output():
        return (a.reshape((6,)) * 2.0).sum()

    out = make_output()
    out.backward()
    assert np.allclose(a.grad, numerical_grad(make_output, a), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/autograd/test_reshape.py -v`
Expected: FAIL with `AttributeError: 'Tensor' object has no attribute 'reshape'`.

- [ ] **Step 3: Write minimal implementation**

Add this method to the `Tensor` class in `src/pyllm/autograd/tensor.py` (put it right after `transpose`):

```python
    def reshape(self, shape):
        """Rearrange the same numbers into a differently shaped box.

        Like tipping a tray of 6 buns into a 2x3 arrangement or a 3x2 one --
        it's the exact same buns, just laid out differently. The MLP uses this
        to lay each position's little embeddings out in one long row before the
        first Linear layer. Breadcrumb rule: the gradient just gets folded back
        into the original shape.
        """
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/autograd/test_reshape.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Tensor.reshape() primitive"
```

---

### Task 2: Bigram model

**Files:**
- Create: `src/pyllm/models/__init__.py`
- Create: `src/pyllm/models/bigram.py`
- Test: `tests/models/test_bigram.py`

**Interfaces:**
- Consumes: `Tensor`, `Module`, `Embedding`.
- Produces: `Bigram(vocab_size, block_size=8, rng=None)`. `.mode == "sequence"`, `.block_size`, `.vocab_size`. Holds `.token_embedding = Embedding(vocab_size, vocab_size)` — each row is directly the logits for the next token. `forward(idx)` for int `idx (B, T)` returns logits `(B, T, vocab_size)`. `.config()` → `{"kind": "bigram", "vocab_size": ..., "block_size": ...}`.

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_bigram.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.models.bigram import Bigram
from pyllm.nn import cross_entropy


def test_bigram_logits_shape():
    model = Bigram(vocab_size=5, rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3], [0, 4, 1]])  # (B, T)
    assert model(idx).shape == (2, 3, 5)


def test_bigram_mode_and_metadata():
    model = Bigram(vocab_size=5, block_size=8, rng=np.random.default_rng(0))
    assert model.mode == "sequence"
    assert model.block_size == 8
    assert model.vocab_size == 5


def test_bigram_config_roundtrips_kind():
    cfg = Bigram(vocab_size=7, block_size=4, rng=np.random.default_rng(0)).config()
    assert cfg == {"kind": "bigram", "vocab_size": 7, "block_size": 4}


def test_bigram_has_one_parameter():
    model = Bigram(vocab_size=5, rng=np.random.default_rng(0))
    assert len(model.parameters()) == 1  # just the embedding table


def test_bigram_loss_and_backward_touch_the_table():
    model = Bigram(vocab_size=5, rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3]])
    targets = np.array([[2, 3, 4]])
    loss = cross_entropy(model(idx), targets)
    loss.backward()
    assert isinstance(loss, Tensor)
    assert np.any(model.token_embedding.weight.grad != 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_bigram.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.models'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/models/__init__.py`:

```python
"""Language models on the teaching ladder, assembled from nn building blocks."""
```

Create `src/pyllm/models/bigram.py`:

```python
from pyllm.nn import Embedding
from pyllm.nn import Module


class Bigram(Module):
    """The simplest language model: "given this letter, what usually comes next?"

    It's a giant lookup table. For every possible current character it stores a
    row of scores -- one score per possible *next* character. There is no
    thinking and no memory of anything before the current letter; it just reads
    off the row. It learns which next-letters are common (after "q" comes "u"),
    which is already enough to babble vaguely word-shaped text.
    """

    def __init__(self, vocab_size, block_size=8, rng=None):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.mode = "sequence"
        # Each row IS the next-token scores for that token id.
        self.token_embedding = Embedding(vocab_size, vocab_size, rng=rng)

    def forward(self, idx):
        return self.token_embedding(idx)  # (B, T, vocab_size)

    def config(self):
        """The recipe card needed to rebuild an identical (untrained) model."""
        return {
            "kind": "bigram",
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_bigram.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Bigram model"
```

---

### Task 3: MLP model

**Files:**
- Create: `src/pyllm/models/mlp.py`
- Test: `tests/models/test_mlp.py`

**Interfaces:**
- Consumes: `Tensor`, `Module`, `Embedding`, `Linear`, `gelu`, `Tensor.reshape`.
- Produces: `MLP(vocab_size, block_size=3, embed_dim=16, hidden_dim=64, rng=None)`. `.mode == "single"`. Holds `.token_embedding = Embedding(vocab_size, embed_dim)`, `.fc1 = Linear(block_size*embed_dim, hidden_dim)`, `.fc2 = Linear(hidden_dim, vocab_size)`. `forward(idx)` for int `idx (B, block_size)` returns logits `(B, vocab_size)`: embed → flatten → `fc1` → `gelu` → `fc2`. `.config()` → `{"kind": "mlp", "vocab_size", "block_size", "embed_dim", "hidden_dim"}`.

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_mlp.py`:

```python
import numpy as np

from pyllm.models.mlp import MLP
from pyllm.nn import cross_entropy


def test_mlp_logits_shape_is_one_prediction_per_window():
    model = MLP(vocab_size=5, block_size=3, embed_dim=8, hidden_dim=16,
                rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3], [0, 4, 1]])  # (B, block_size)
    assert model(idx).shape == (2, 5)  # one next-token distribution per row


def test_mlp_mode_and_metadata():
    model = MLP(vocab_size=5, block_size=3, rng=np.random.default_rng(0))
    assert model.mode == "single"
    assert model.block_size == 3
    assert model.vocab_size == 5


def test_mlp_config():
    cfg = MLP(vocab_size=5, block_size=3, embed_dim=8, hidden_dim=16,
              rng=np.random.default_rng(0)).config()
    assert cfg == {
        "kind": "mlp", "vocab_size": 5, "block_size": 3,
        "embed_dim": 8, "hidden_dim": 16,
    }


def test_mlp_has_embedding_and_two_linears():
    model = MLP(vocab_size=5, block_size=3, embed_dim=8, hidden_dim=16,
                rng=np.random.default_rng(0))
    # embedding(1) + fc1(weight+bias) + fc2(weight+bias) = 5
    assert len(model.parameters()) == 5


def test_mlp_learns_gradients_flow():
    model = MLP(vocab_size=5, block_size=3, embed_dim=8, hidden_dim=16,
                rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3]])
    targets = np.array([4])  # (B,) single next token
    loss = cross_entropy(model(idx), targets)
    loss.backward()
    assert np.any(model.fc1.weight.grad != 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_mlp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.models.mlp'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/models/mlp.py`:

```python
from pyllm.nn import Embedding
from pyllm.nn import Linear
from pyllm.nn import Module
from pyllm.nn import gelu


class MLP(Module):
    """A small neural net that reads the last few characters, then guesses the next.

    The bigram only ever looks at *one* letter. This model looks at a little
    window of the last `block_size` letters at once. It turns each letter into a
    short list of numbers (an embedding), lays those lists out in one long row,
    and passes them through a tiny "thinking" layer (a hidden Linear + a smooth
    `gelu` switch) before guessing the next letter. Wider context = smarter
    guesses than the bigram.
    """

    def __init__(self, vocab_size, block_size=3, embed_dim=16, hidden_dim=64,
                 rng=None):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.mode = "single"
        self.token_embedding = Embedding(vocab_size, embed_dim, rng=rng)
        self.fc1 = Linear(block_size * embed_dim, hidden_dim, rng=rng)
        self.fc2 = Linear(hidden_dim, vocab_size, rng=rng)

    def forward(self, idx):
        batch = idx.shape[0]
        emb = self.token_embedding(idx)  # (B, block_size, embed_dim)
        flat = emb.reshape((batch, self.block_size * self.embed_dim))
        hidden = gelu(self.fc1(flat))
        return self.fc2(hidden)  # (B, vocab_size)

    def config(self):
        """The recipe card needed to rebuild an identical (untrained) model."""
        return {
            "kind": "mlp",
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_mlp.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add MLP model"
```

---

### Task 4: GPT model

**Files:**
- Create: `src/pyllm/models/gpt.py`
- Test: `tests/models/test_gpt.py`

**Interfaces:**
- Consumes: `Tensor`, `Module`, `Embedding`, `LayerNorm`, `Linear`, `TransformerBlock`, `embedding`.
- Produces: `GPT(vocab_size, block_size=8, embed_dim=32, num_heads=4, num_layers=2, rng=None)`. `.mode == "sequence"`. Holds `.token_embedding = Embedding(vocab_size, embed_dim)`, `.position_embedding = Embedding(block_size, embed_dim)`, `.blocks = [TransformerBlock(embed_dim, num_heads, block_size, rng) for _ in range(num_layers)]`, `.ln_f = LayerNorm(embed_dim)`, `.head = Linear(embed_dim, vocab_size)`. `forward(idx)` for int `idx (B, T)` (with `T <= block_size`) returns logits `(B, T, vocab_size)`: `tok + pos → blocks → ln_f → head`. `.config()` → `{"kind": "gpt", "vocab_size", "block_size", "embed_dim", "num_heads", "num_layers"}`.

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_gpt.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.models.gpt import GPT
from pyllm.nn import cross_entropy


def test_gpt_logits_shape():
    model = GPT(vocab_size=6, block_size=8, embed_dim=16, num_heads=2,
                num_layers=2, rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3, 4]])  # (B, T), T <= block_size
    assert model(idx).shape == (1, 4, 6)


def test_gpt_mode_and_config():
    model = GPT(vocab_size=6, block_size=8, embed_dim=16, num_heads=2,
                num_layers=2, rng=np.random.default_rng(0))
    assert model.mode == "sequence"
    assert model.config() == {
        "kind": "gpt", "vocab_size": 6, "block_size": 8,
        "embed_dim": 16, "num_heads": 2, "num_layers": 2,
    }


def test_gpt_is_causal():
    # Disturbing the LAST position must not change earlier-position logits.
    model = GPT(vocab_size=6, block_size=8, embed_dim=16, num_heads=2,
                num_layers=2, rng=np.random.default_rng(0))
    base = np.array([[1, 2, 3, 4, 5]])
    changed = base.copy()
    changed[0, -1] = 0  # change only the final token
    out_base = model(base).data
    out_changed = model(changed).data
    assert np.allclose(out_base[0, :4], out_changed[0, :4], atol=1e-8)


def test_gpt_trains_end_to_end_gradient_flow():
    model = GPT(vocab_size=6, block_size=8, embed_dim=16, num_heads=2,
                num_layers=2, rng=np.random.default_rng(0))
    idx = np.array([[1, 2, 3, 4]])
    targets = np.array([[2, 3, 4, 5]])
    loss = cross_entropy(model(idx), targets)
    loss.backward()
    assert isinstance(loss, Tensor)
    assert np.any(model.token_embedding.weight.grad != 0.0)
    assert np.any(model.head.weight.grad != 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_gpt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.models.gpt'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/models/gpt.py`:

```python
import numpy as np

from pyllm.nn import Embedding
from pyllm.nn import LayerNorm
from pyllm.nn import Linear
from pyllm.nn import Module
from pyllm.nn import TransformerBlock
from pyllm.nn import embedding


class GPT(Module):
    """The real thing: a stack of Transformer blocks that reads a whole window.

    Each character becomes a vector (token embedding), and because "the cat sat"
    is not "sat the cat", we also add a vector for *where* it sits (position
    embedding). Then a tower of Transformer blocks lets every position gather
    clues from the positions before it (attention) and think them over
    (feed-forward). A final tidy-up (LayerNorm) and a last Linear turn each
    position into scores for the next character. Stack more blocks -> smarter.
    """

    def __init__(self, vocab_size, block_size=8, embed_dim=32, num_heads=4,
                 num_layers=2, rng=None):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mode = "sequence"
        self.token_embedding = Embedding(vocab_size, embed_dim, rng=rng)
        self.position_embedding = Embedding(block_size, embed_dim, rng=rng)
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, block_size, rng=rng)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, vocab_size, rng=rng)

    def forward(self, idx):
        seq_len = idx.shape[1]
        tok = self.token_embedding(idx)  # (B, T, embed_dim)
        pos = embedding(self.position_embedding.weight, np.arange(seq_len))
        x = tok + pos  # (T, embed_dim) broadcasts over the batch
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)

    def config(self):
        """The recipe card needed to rebuild an identical (untrained) model."""
        return {
            "kind": "gpt",
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_gpt.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add GPT model"
```

---

### Task 5: models public API + build_model factory

**Files:**
- Modify: `src/pyllm/models/__init__.py`
- Test: `tests/models/test_build_model.py`

**Interfaces:**
- Consumes: `Bigram`, `MLP`, `GPT`.
- Produces: `pyllm.models` re-exports `Bigram`, `MLP`, `GPT`, and `build_model(config, rng=None)` which reads `config["kind"]` and constructs the matching model from the remaining config keys. Round-trips `model.config()` back to an identical-shape model.

- [ ] **Step 1: Write the failing test**

Create `tests/models/test_build_model.py`:

```python
import numpy as np

from pyllm.models import GPT
from pyllm.models import build_model


def test_build_model_roundtrips_every_kind():
    rng = np.random.default_rng(0)
    originals = [
        ("bigram", {"kind": "bigram", "vocab_size": 5, "block_size": 8}),
        ("mlp", {"kind": "mlp", "vocab_size": 5, "block_size": 3,
                 "embed_dim": 8, "hidden_dim": 16}),
        ("gpt", {"kind": "gpt", "vocab_size": 5, "block_size": 8,
                 "embed_dim": 16, "num_heads": 2, "num_layers": 2}),
    ]
    for _kind, cfg in originals:
        model = build_model(cfg, rng=rng)
        assert model.config() == cfg


def test_build_model_produces_working_gpt():
    cfg = GPT(vocab_size=5, block_size=8, embed_dim=16, num_heads=2,
              num_layers=1, rng=np.random.default_rng(0)).config()
    model = build_model(cfg, rng=np.random.default_rng(0))
    assert model(np.array([[1, 2, 3]])).shape == (1, 3, 5)


def test_build_model_rejects_unknown_kind():
    import pytest
    with pytest.raises(ValueError):
        build_model({"kind": "quantum"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/models/test_build_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_model'`.

- [ ] **Step 3: Write minimal implementation**

Replace `src/pyllm/models/__init__.py`:

```python
"""Language models on the teaching ladder, assembled from nn building blocks."""

from pyllm.models.bigram import Bigram
from pyllm.models.gpt import GPT
from pyllm.models.mlp import MLP

__all__ = ["Bigram", "MLP", "GPT", "build_model"]


def build_model(config, rng=None):
    """Rebuild a model from its recipe card (`config()` output).

    A saved checkpoint stores only the *numbers* (weights) plus this little
    recipe card. To bring the model back to life we first build the right empty
    shape from the card, then pour the saved numbers in (that second step lives
    in `checkpoint.load`). `kind` picks which model to build.
    """
    kind = config["kind"]
    if kind == "bigram":
        return Bigram(config["vocab_size"], block_size=config["block_size"],
                      rng=rng)
    if kind == "mlp":
        return MLP(config["vocab_size"], block_size=config["block_size"],
                   embed_dim=config["embed_dim"],
                   hidden_dim=config["hidden_dim"], rng=rng)
    if kind == "gpt":
        return GPT(config["vocab_size"], block_size=config["block_size"],
                   embed_dim=config["embed_dim"], num_heads=config["num_heads"],
                   num_layers=config["num_layers"], rng=rng)
    raise ValueError(f"unknown model kind: {kind!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/models/test_build_model.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: expose models public API + build_model factory"
```

---

### Task 6: SGD optimizer

**Files:**
- Create: `src/pyllm/training/__init__.py`
- Create: `src/pyllm/training/optim.py`
- Test: `tests/training/test_optim.py`

**Interfaces:**
- Consumes: `Tensor` (its `.data` and `.grad` numpy arrays).
- Produces: `SGD(parameters, lr=0.1)` with `.step()` (each `p.data -= lr * p.grad`) and `.zero_grad()` (each `p.grad[:] = 0`). `parameters` is the list from `model.parameters()`.

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_optim.py`:

```python
import numpy as np

from pyllm.autograd import Tensor
from pyllm.training.optim import SGD


def test_sgd_step_moves_downhill():
    p = Tensor([10.0])
    p.grad = np.array([2.0])  # loss increases with p -> step should decrease p
    SGD([p], lr=0.5).step()
    assert np.isclose(p.data[0], 10.0 - 0.5 * 2.0)  # 9.0


def test_sgd_minimizes_a_simple_quadratic():
    # minimize (x - 3)^2 by gradient descent; should approach x = 3
    x = Tensor([0.0])
    opt = SGD([x], lr=0.1)
    for _ in range(200):
        opt.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        opt.step()
    assert np.isclose(x.data[0], 3.0, atol=1e-2)


def test_sgd_zero_grad_clears_gradients():
    p = Tensor([1.0])
    p.grad = np.array([5.0])
    SGD([p]).zero_grad()
    assert np.all(p.grad == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/training/test_optim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.training'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/training/__init__.py`:

```python
"""From-scratch optimizers, batching, and the training loop."""
```

Create `src/pyllm/training/optim.py`:

```python
import numpy as np


class SGD:
    """Walk downhill in the direction that lowers the loss fastest.

    Imagine standing on a foggy hill wanting to reach the bottom. You can't see
    far, but you can feel which way is downhill under your feet (that's the
    gradient). SGD takes a small step that way, over and over. `lr` (learning
    rate) is your step size: too big and you overshoot the valley, too small and
    you crawl.
    """

    def __init__(self, parameters, lr=0.1):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        """Take one downhill step for every parameter."""
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        """Erase the slope readings before measuring the next ones."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/training/test_optim.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add SGD optimizer"
```

---

### Task 7: Adam optimizer

**Files:**
- Modify: `src/pyllm/training/optim.py`
- Test: `tests/training/test_optim.py`

**Interfaces:**
- Consumes: `Tensor`.
- Produces: `Adam(parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)` with `.step()` (bias-corrected first/second moment update) and `.zero_grad()`. Keeps per-parameter running averages keyed by list position.

- [ ] **Step 1: Write the failing test**

Add to `tests/training/test_optim.py`:

```python
from pyllm.training.optim import Adam


def test_adam_minimizes_a_simple_quadratic():
    x = Tensor([0.0])
    opt = Adam([x], lr=0.1)
    for _ in range(500):
        opt.zero_grad()
        loss = (x - 3.0) ** 2
        loss.backward()
        opt.step()
    assert np.isclose(x.data[0], 3.0, atol=1e-2)


def test_adam_first_step_size_is_about_lr():
    # On the very first step Adam's update magnitude is ~lr regardless of grad scale.
    x = Tensor([0.0])
    x.grad = np.array([1000.0])
    Adam([x], lr=0.1).step()
    assert np.isclose(abs(x.data[0]), 0.1, atol=1e-6)


def test_adam_zero_grad_clears_gradients():
    p = Tensor([1.0])
    p.grad = np.array([5.0])
    Adam([p]).zero_grad()
    assert np.all(p.grad == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/training/test_optim.py -k adam -v`
Expected: FAIL with `ImportError: cannot import name 'Adam'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/pyllm/training/optim.py`:

```python
class Adam:
    """A smarter downhill walker that adapts its step size per parameter.

    Plain SGD uses one step size for everything. Adam remembers two things for
    each parameter: the recent *average* slope (momentum -- keep rolling the way
    you've been going) and the recent *size* of the slope (so it takes smaller
    steps where the ground is steep and bigger steps where it's flat). This lets
    it learn faster and more steadily, which is why almost every real network
    uses it. `betas` say how long each memory lasts; `eps` avoids divide-by-zero.
    """

    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self):
        """Take one adaptive step for every parameter."""
        self.t += 1
        for i, p in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Erase the slope readings before measuring the next ones."""
        for p in self.parameters:
            p.grad = np.zeros_like(p.data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/training/test_optim.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Adam optimizer"
```

---

### Task 8: get_batch batching

**Files:**
- Create: `src/pyllm/training/data.py`
- Test: `tests/training/test_data.py`

**Interfaces:**
- Consumes: numpy only.
- Produces: `get_batch(data, block_size, batch_size, mode="sequence", rng=None)`. `data` is a 1-D int numpy array of token ids. Returns `(x, y)` int numpy arrays:
  - `mode="sequence"`: `x (batch_size, block_size)`, `y (batch_size, block_size)` where `y` is `x` shifted one step into the future (next token at every position).
  - `mode="single"`: `x (batch_size, block_size)`, `y (batch_size,)` where `y` is the single token following each window.

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_data.py`:

```python
import numpy as np

from pyllm.training.data import get_batch


def test_sequence_batch_shapes_and_shift():
    data = np.arange(20)
    x, y = get_batch(data, block_size=4, batch_size=3, mode="sequence",
                     rng=np.random.default_rng(0))
    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    # y is x shifted forward by one everywhere
    for row in range(3):
        assert np.array_equal(y[row], x[row] + 1)  # data is 0,1,2,... so +1


def test_single_batch_shapes_and_target():
    data = np.arange(20)
    x, y = get_batch(data, block_size=4, batch_size=3, mode="single",
                     rng=np.random.default_rng(0))
    assert x.shape == (3, 4)
    assert y.shape == (3,)
    # target is the token right after the window (window ends at x[:, -1])
    for row in range(3):
        assert y[row] == x[row, -1] + 1


def test_batch_is_deterministic_with_seed():
    data = np.arange(50)
    a = get_batch(data, 4, 5, rng=np.random.default_rng(7))
    b = get_batch(data, 4, 5, rng=np.random.default_rng(7))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_batch_ids_are_integers():
    data = np.arange(20)
    x, y = get_batch(data, 4, 2, rng=np.random.default_rng(0))
    assert np.issubdtype(x.dtype, np.integer)
    assert np.issubdtype(y.dtype, np.integer)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/training/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.training.data'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/training/data.py`:

```python
import numpy as np


def get_batch(data, block_size, batch_size, mode="sequence", rng=None):
    """Grab a handful of random windows of text to learn from this step.

    Training on the whole corpus at once would be slow, so each step we grab a
    small random sample -- a "batch" -- of little windows cut from the text.
    `x` is what the model reads; `y` is the answer it should predict.

    - "sequence" (bigram, GPT): the model predicts the next token at *every*
      position, so `y` is just `x` slid one character to the right.
    - "single" (MLP): the model reads a whole window and predicts the *one*
      character that comes right after it.
    """
    if rng is None:
        rng = np.random.default_rng()
    data = np.asarray(data)
    max_start = len(data) - block_size - 1
    starts = rng.integers(0, max_start + 1, size=batch_size)
    x = np.stack([data[s:s + block_size] for s in starts])
    if mode == "sequence":
        y = np.stack([data[s + 1:s + 1 + block_size] for s in starts])
    else:  # "single"
        y = np.array([data[s + block_size] for s in starts])
    return x.astype(np.int64), y.astype(np.int64)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/training/test_data.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add get_batch batching"
```

---

### Task 9: training loop

**Files:**
- Create: `src/pyllm/training/loop.py`
- Modify: `src/pyllm/training/__init__.py`
- Test: `tests/training/test_loop.py`

**Interfaces:**
- Consumes: `cross_entropy`, `SGD`, `Adam`, `get_batch`, a model with `.mode`, `.block_size`, `.parameters()`, `.train()`.
- Produces:
  - `train(model, data, steps=1000, batch_size=32, lr=1e-3, optimizer="adam", rng=None, log_every=0, log=print)` — runs `steps` training steps and returns a `list[float]` of per-step losses. Uses `model.block_size` and `model.mode` for batching. `optimizer` is `"adam"` or `"sgd"`. When `log_every > 0`, calls `log(f"step {i}: loss {loss:.4f}")` every `log_every` steps.
  - `src/pyllm/training/__init__.py` re-exports `SGD`, `Adam`, `get_batch`, `train`.

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_loop.py`:

```python
import numpy as np

from pyllm.models import Bigram
from pyllm.models import MLP
from pyllm.training import train


def test_train_returns_loss_per_step():
    rng = np.random.default_rng(0)
    data = np.array([0, 1, 2, 3] * 50)
    model = Bigram(vocab_size=4, block_size=4, rng=rng)
    losses = train(model, data, steps=5, batch_size=8, rng=rng)
    assert len(losses) == 5
    assert all(isinstance(x, float) for x in losses)


def test_train_drives_loss_down_on_repetitive_data():
    rng = np.random.default_rng(0)
    data = np.array([0, 1, 2, 3] * 200)  # perfectly predictable cycle
    model = Bigram(vocab_size=4, block_size=4, rng=rng)
    losses = train(model, data, steps=300, batch_size=16, lr=0.5,
                   optimizer="sgd", rng=rng)
    assert losses[-1] < losses[0] * 0.5  # loss at least halved


def test_train_works_for_single_mode_model():
    rng = np.random.default_rng(0)
    data = np.array([0, 1, 2, 3] * 200)
    model = MLP(vocab_size=4, block_size=3, embed_dim=8, hidden_dim=16, rng=rng)
    losses = train(model, data, steps=100, batch_size=16, rng=rng)
    assert losses[-1] < losses[0]


def test_train_log_every_calls_log(capsys):
    rng = np.random.default_rng(0)
    data = np.array([0, 1, 2, 3] * 50)
    model = Bigram(vocab_size=4, block_size=4, rng=rng)
    messages = []
    train(model, data, steps=4, batch_size=8, rng=rng, log_every=2,
          log=messages.append)
    assert len(messages) == 2  # steps 0 and 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/training/test_loop.py -v`
Expected: FAIL with `ImportError: cannot import name 'train'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/training/loop.py`:

```python
from pyllm.nn import cross_entropy
from pyllm.training.data import get_batch
from pyllm.training.optim import SGD
from pyllm.training.optim import Adam


def train(model, data, steps=1000, batch_size=32, lr=1e-3, optimizer="adam",
          rng=None, log_every=0, log=print):
    """Teach a model by showing it batches over and over and nudging its weights.

    This is the whole heartbeat of learning, repeated `steps` times:
    1. grab a random batch of windows (`get_batch`),
    2. let the model guess (`forward`),
    3. measure how wrong it was (`cross_entropy` -- the loss),
    4. trace the blame back to every weight (`loss.backward()`),
    5. nudge every weight a little to be less wrong (`optimizer.step()`).
    Do that enough times and the model gets good at predicting the next token.
    Returns the loss at each step so you can watch it fall.
    """
    model.train()
    params = model.parameters()
    opt = Adam(params, lr=lr) if optimizer == "adam" else SGD(params, lr=lr)
    losses = []
    for step in range(steps):
        x, y = get_batch(data, model.block_size, batch_size, mode=model.mode,
                         rng=rng)
        logits = model(x)
        loss = cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_value = float(loss.data)
        losses.append(loss_value)
        if log_every and step % log_every == 0:
            log(f"step {step}: loss {loss_value:.4f}")
    return losses
```

Update `src/pyllm/training/__init__.py`:

```python
"""From-scratch optimizers, batching, and the training loop."""

from pyllm.training.data import get_batch
from pyllm.training.loop import train
from pyllm.training.optim import SGD
from pyllm.training.optim import Adam

__all__ = ["SGD", "Adam", "get_batch", "train"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/training/test_loop.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add training loop"
```

---

### Task 10: overfit-a-tiny-batch end-to-end test

**Files:**
- Test: `tests/test_overfit.py`

**Interfaces:**
- Consumes: `CharTokenizer`, `GPT`, `train`. No new production code — this is the decisive smoke test of the whole stack (tokenizer → model → loss → optimizer). A model that cannot memorize a tiny corpus is broken.

- [ ] **Step 1: Write the failing test**

Create `tests/test_overfit.py`:

```python
import numpy as np

from pyllm.models import GPT
from pyllm.tokenizer import CharTokenizer
from pyllm.training import train


def test_gpt_can_overfit_a_tiny_corpus():
    # A GPT with enough steps must drive the loss near zero on a tiny, fixed text.
    text = "pikachu bulbasaur charmander squirtle"
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text))
    rng = np.random.default_rng(0)
    model = GPT(vocab_size=tok.vocab_size, block_size=8, embed_dim=32,
                num_heads=4, num_layers=2, rng=rng)
    losses = train(model, data, steps=400, batch_size=16, lr=1e-2, rng=rng)
    assert losses[-1] < 0.3, f"expected near-zero loss, got {losses[-1]}"
```

- [ ] **Step 2: Run test to verify it fails/passes**

Run: `uv run pytest tests/test_overfit.py -v`
Expected: PASS (this validates the whole stack). If it FAILS, the bug is in an earlier task — debug there, do not weaken the assertion.

- [ ] **Step 3: (no implementation — verification task)**

If the loss does not drop below 0.3, use `superpowers:systematic-debugging`: check gradient signs (SGD/Adam), that `model.train()` doesn't disable learning, and that `get_batch` shifts targets correctly. Increase `steps` only if it is genuinely still descending; do not lower the bar.

- [ ] **Step 4: Confirm the full suite is still green**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: add overfit-a-tiny-corpus end-to-end smoke test"
```

---

### Task 11: sampling — sample_next (greedy / temperature / top-k)

**Files:**
- Create: `src/pyllm/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: numpy only.
- Produces: `sample_next(logits_row, temperature=1.0, top_k=None, rng=None) -> int`. `logits_row` is a 1-D numpy array of length `vocab_size`.
  - `temperature == 0` → greedy (`argmax`).
  - otherwise divide logits by `temperature`, optionally keep only the `top_k` highest (mask the rest to `-inf`), softmax, sample one index with `rng`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_generate.py`:

```python
import numpy as np

from pyllm.generate import sample_next


def test_greedy_picks_argmax():
    logits = np.array([0.1, 5.0, -2.0, 0.3])
    assert sample_next(logits, temperature=0.0) == 1


def test_top_k_1_is_effectively_greedy():
    logits = np.array([0.1, 5.0, -2.0, 0.3])
    rng = np.random.default_rng(0)
    picks = {sample_next(logits, temperature=1.0, top_k=1, rng=rng)
             for _ in range(20)}
    assert picks == {1}  # only the single top token is ever allowed


def test_low_temperature_concentrates_on_top_token():
    logits = np.array([0.0, 2.0, 0.0])
    rng = np.random.default_rng(0)
    picks = [sample_next(logits, temperature=0.1, rng=rng) for _ in range(200)]
    assert picks.count(1) > 190  # nearly always the peak


def test_sampling_stays_in_vocab_range():
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    rng = np.random.default_rng(1)
    for _ in range(50):
        idx = sample_next(logits, temperature=1.0, rng=rng)
        assert 0 <= idx < 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.generate'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/generate.py`:

```python
import numpy as np


def sample_next(logits_row, temperature=1.0, top_k=None, rng=None):
    """Pick the next token from the model's scores -- boldly or cautiously.

    The model hands back a score for every possible next character. We turn those
    into a dice with weighted sides and roll it.
    - `temperature` is the boldness dial: 0 always takes the single best guess
      (safe but repetitive); 1 rolls fairly by the model's confidence; higher is
      wilder and more surprising.
    - `top_k` says "only consider the k most likely characters" so the roll never
      lands on something absurd.
    """
    if rng is None:
        rng = np.random.default_rng()
    logits = np.asarray(logits_row, dtype=np.float64)
    if temperature == 0.0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k is not None:
        keep = np.argsort(logits)[-top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add sample_next (greedy/temperature/top-k)"
```

---

### Task 12: generate() — autoregressive text generation

**Files:**
- Modify: `src/pyllm/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: `sample_next`, a model with `.mode`, `.block_size`, and `forward`; a tokenizer with `.encode`/`.decode`.
- Produces: `generate(model, tokenizer, prompt="", max_new_tokens=100, temperature=1.0, top_k=None, rng=None) -> str`. Encodes `prompt`, then repeatedly: crops the running context to the last `block_size` tokens (left-padding with token 0 for `"single"` models when short), runs the model, reads the **last position's** logits, samples the next id, appends it. Returns the decoded full string (prompt + generated).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_generate.py`:

```python
from pyllm.generate import generate
from pyllm.models import GPT
from pyllm.models import MLP
from pyllm.tokenizer import CharTokenizer


def test_generate_returns_prompt_plus_new_tokens():
    text = "abcde"
    tok = CharTokenizer(text)
    model = GPT(vocab_size=tok.vocab_size, block_size=8, embed_dim=8,
                num_heads=2, num_layers=1, rng=np.random.default_rng(0))
    out = generate(model, tok, prompt="a", max_new_tokens=5,
                   rng=np.random.default_rng(0))
    assert out.startswith("a")
    assert len(out) == 1 + 5  # prompt char + 5 generated


def test_generate_only_emits_known_characters():
    text = "pokemon"
    tok = CharTokenizer(text)
    model = GPT(vocab_size=tok.vocab_size, block_size=8, embed_dim=8,
                num_heads=2, num_layers=1, rng=np.random.default_rng(0))
    out = generate(model, tok, prompt="p", max_new_tokens=20,
                   rng=np.random.default_rng(1))
    assert set(out).issubset(set(text))


def test_generate_works_for_single_mode_model_with_short_prompt():
    text = "abcdef"
    tok = CharTokenizer(text)
    model = MLP(vocab_size=tok.vocab_size, block_size=3, embed_dim=8,
                hidden_dim=16, rng=np.random.default_rng(0))
    # empty prompt must still work (left-pad the window)
    out = generate(model, tok, prompt="", max_new_tokens=4,
                   rng=np.random.default_rng(0))
    assert len(out) == 4
    assert set(out).issubset(set(text))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generate.py -k generate_returns -v`
Expected: FAIL with `ImportError: cannot import name 'generate'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/pyllm/generate.py`:

```python
def generate(model, tokenizer, prompt="", max_new_tokens=100, temperature=1.0,
             top_k=None, rng=None):
    """Dream up new text one character at a time, feeding each guess back in.

    Start with the prompt, ask the model "what comes next?", roll for a character
    (`sample_next`), stick it on the end, and repeat. Each new character becomes
    part of the context for the next guess -- that feedback loop is why a model
    trained on Pokémon names can invent brand-new ones that *feel* like Pokémon.
    We only ever look at the last `block_size` characters (the model's memory).
    """
    if rng is None:
        rng = np.random.default_rng()
    model.eval()
    context = list(tokenizer.encode(prompt))
    for _ in range(max_new_tokens):
        window = context[-model.block_size:]
        if model.mode == "single":
            # MLP needs exactly block_size tokens; left-pad short windows.
            if len(window) < model.block_size:
                window = [0] * (model.block_size - len(window)) + window
        x = np.array([window])
        logits = model(x).data
        last_row = logits[0, -1] if logits.ndim == 3 else logits[0]
        context.append(sample_next(last_row, temperature=temperature,
                                    top_k=top_k, rng=rng))
    return tokenizer.decode(context)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generate.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add autoregressive generate()"
```

---

### Task 13: checkpoint save/load

**Files:**
- Create: `src/pyllm/checkpoint.py`
- Test: `tests/test_checkpoint.py`

**Interfaces:**
- Consumes: `build_model`, `CharTokenizer`, a model with `.parameters()` and `.config()`.
- Produces:
  - `save(path, model, tokenizer)` — writes a numpy `.npz` holding `param_0..param_{n-1}` (each `p.data`), `config` (a JSON string of `model.config()`), and `vocab` (a JSON string of the tokenizer's sorted chars, i.e. `list(tokenizer.stoi)`).
  - `load(path, rng=None) -> (model, tokenizer)` — reads the `.npz`, rebuilds the model via `build_model(config, rng)`, assigns saved arrays into `model.parameters()` in order, and reconstructs the `CharTokenizer` from the stored vocab (feeding the joined sorted chars reproduces identical `stoi`/`itos`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_checkpoint.py`:

```python
import numpy as np

from pyllm.checkpoint import load
from pyllm.checkpoint import save
from pyllm.generate import generate
from pyllm.models import GPT
from pyllm.tokenizer import CharTokenizer


def test_save_load_roundtrips_config_and_weights(tmp_path):
    text = "pikachu"
    tok = CharTokenizer(text)
    model = GPT(vocab_size=tok.vocab_size, block_size=8, embed_dim=8,
                num_heads=2, num_layers=1, rng=np.random.default_rng(0))
    path = tmp_path / "model.npz"
    save(path, model, tok)
    reloaded, reloaded_tok = load(path)
    assert reloaded.config() == model.config()
    for a, b in zip(model.parameters(), reloaded.parameters()):
        assert np.allclose(a.data, b.data)


def test_reloaded_tokenizer_matches(tmp_path):
    text = "bulbasaur"
    tok = CharTokenizer(text)
    model = GPT(vocab_size=tok.vocab_size, block_size=8, embed_dim=8,
                num_heads=2, num_layers=1, rng=np.random.default_rng(0))
    path = tmp_path / "m.npz"
    save(path, model, tok)
    _, reloaded_tok = load(path)
    assert reloaded_tok.stoi == tok.stoi
    assert reloaded_tok.decode(reloaded_tok.encode("bulba")) == "bulba"


def test_reloaded_model_generates_identically(tmp_path):
    text = "charmander"
    tok = CharTokenizer(text)
    model = GPT(vocab_size=tok.vocab_size, block_size=8, embed_dim=8,
                num_heads=2, num_layers=1, rng=np.random.default_rng(0))
    path = tmp_path / "m.npz"
    save(path, model, tok)
    reloaded, reloaded_tok = load(path)
    a = generate(model, tok, prompt="c", max_new_tokens=10,
                 rng=np.random.default_rng(3))
    b = generate(reloaded, reloaded_tok, prompt="c", max_new_tokens=10,
                 rng=np.random.default_rng(3))
    assert a == b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.checkpoint'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/checkpoint.py`:

```python
import json

import numpy as np

from pyllm.models import build_model
from pyllm.tokenizer import CharTokenizer


def save(path, model, tokenizer):
    """Freeze a trained model to a file so it can wake up exactly as it was.

    A checkpoint is like a photograph of the model's brain. We store three
    things: every learned number (the weights), the recipe card that says how
    the model is shaped (`config`), and the tokenizer's alphabet so text maps to
    the same numbers next time. Later `load` uses the recipe to build an empty
    model and pours the saved numbers back in.
    """
    arrays = {f"param_{i}": p.data for i, p in enumerate(model.parameters())}
    arrays["config"] = np.array(json.dumps(model.config()))
    arrays["vocab"] = np.array(json.dumps(list(tokenizer.stoi)))
    np.savez(path, **arrays)


def load(path, rng=None):
    """Wake a saved model back up: rebuild its shape, pour its numbers back in.

    The reverse of `save`. We read the recipe card, build a fresh empty model of
    the right shape, then copy each saved weight into place in the same order
    `parameters()` lists them. The tokenizer is rebuilt from the saved alphabet
    -- because our alphabet is always sorted, handing the sorted letters back to
    `CharTokenizer` reproduces the exact same letter<->number mapping.
    """
    data = np.load(path, allow_pickle=False)
    config = json.loads(str(data["config"]))
    model = build_model(config, rng=rng)
    for i, p in enumerate(model.parameters()):
        p.data = data[f"param_{i}"]
    vocab = json.loads(str(data["vocab"]))
    tokenizer = CharTokenizer("".join(vocab))
    return model, tokenizer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_checkpoint.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add checkpoint save/load"
```

---

### Task 14: Pokémon corpus + loader

**Files:**
- Create: `src/pyllm/data/__init__.py`
- Create: `src/pyllm/data/pokemon_corpus.txt`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `pathlib`.
- Produces:
  - `src/pyllm/data/pokemon_corpus.txt` — one lowercase Pokémon name per line (the Gen-1 roster below is enough to train a tiny char model).
  - `pyllm.data.CORPUS_DIR` (a `Path` to `src/pyllm/data/`), `pyllm.data.POKEMON_CHECKPOINT` (a `Path` to `pokemon.npz`, created in Task 15), and `load_corpus(name="pokemon") -> str` which reads `<name>_corpus.txt` and returns its text.

- [ ] **Step 1: Write the failing test**

Create `tests/test_corpus.py`:

```python
from pyllm.data import load_corpus


def test_pokemon_corpus_loads_nonempty_text():
    text = load_corpus("pokemon")
    assert isinstance(text, str)
    assert len(text) > 500


def test_pokemon_corpus_has_many_names():
    names = [line for line in load_corpus("pokemon").splitlines() if line]
    assert len(names) >= 100


def test_pokemon_corpus_is_lowercase_letters_and_newlines():
    text = load_corpus("pokemon")
    assert set(text) <= set("abcdefghijklmnopqrstuvwxyz\n .-'")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.data'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/data/pokemon_corpus.txt` with one name per line (Gen 1, lowercase). Paste exactly:

```
bulbasaur
ivysaur
venusaur
charmander
charmeleon
charizard
squirtle
wartortle
blastoise
caterpie
metapod
butterfree
weedle
kakuna
beedrill
pidgey
pidgeotto
pidgeot
rattata
raticate
spearow
fearow
ekans
arbok
pikachu
raichu
sandshrew
sandslash
nidoran
nidorina
nidoqueen
nidorino
nidoking
clefairy
clefable
vulpix
ninetales
jigglypuff
wigglytuff
zubat
golbat
oddish
gloom
vileplume
paras
parasect
venonat
venomoth
diglett
dugtrio
meowth
persian
psyduck
golduck
mankey
primeape
growlithe
arcanine
poliwag
poliwhirl
poliwrath
abra
kadabra
alakazam
machop
machoke
machamp
bellsprout
weepinbell
victreebel
tentacool
tentacruel
geodude
graveler
golem
ponyta
rapidash
slowpoke
slowbro
magnemite
magneton
farfetchd
doduo
dodrio
seel
dewgong
grimer
muk
shellder
cloyster
gastly
haunter
gengar
onix
drowzee
hypno
krabby
kingler
voltorb
electrode
exeggcute
exeggutor
cubone
marowak
hitmonlee
hitmonchan
lickitung
koffing
weezing
rhyhorn
rhydon
chansey
tangela
kangaskhan
horsea
seadra
goldeen
seaking
staryu
starmie
scyther
jynx
electabuzz
magmar
pinsir
tauros
magikarp
gyarados
lapras
ditto
eevee
vaporeon
jolteon
flareon
porygon
omanyte
omastar
kabuto
kabutops
aerodactyl
snorlax
articuno
zapdos
moltres
dratini
dragonair
dragonite
mewtwo
mew
```

Create `src/pyllm/data/__init__.py`:

```python
"""Bundled training corpora and the shipped Pokémon checkpoint."""

from pathlib import Path

CORPUS_DIR = Path(__file__).parent
POKEMON_CHECKPOINT = CORPUS_DIR / "pokemon.npz"


def load_corpus(name="pokemon"):
    """Read one of the bundled text corpora and hand back its text.

    A corpus is just a big text file we train on. The Pokémon one is a list of
    real Pokémon names -- small enough to train on a laptop in minutes, but full
    of the sounds and shapes that make a name feel Pokémon-ish, so the model can
    learn to dream up new ones.
    """
    return (CORPUS_DIR / f"{name}_corpus.txt").read_text(encoding="utf-8")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: bundle Pokémon corpus + loader"
```

---

### Task 15: train + bundle the tiny Pokémon checkpoint

**Files:**
- Create: `scripts/train_pokemon.py`
- Create (build artifact, committed): `src/pyllm/data/pokemon.npz`
- Test: `tests/test_bundled_checkpoint.py`

**Interfaces:**
- Consumes: `load_corpus`, `CharTokenizer`, `GPT`, `train`, `save`, `POKEMON_CHECKPOINT`.
- Produces:
  - `scripts/train_pokemon.py` — a build-time script (`uv run python scripts/train_pokemon.py`) that trains a tiny GPT on the Pokémon corpus with a fixed seed and writes `src/pyllm/data/pokemon.npz`. Defines `build_and_train(rng)` returning `(model, tokenizer)` so the test can reuse it cheaply.
  - the committed `pokemon.npz` so `pyllm` generates on launch without training.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bundled_checkpoint.py`:

```python
import numpy as np

from pyllm.checkpoint import load
from pyllm.data import POKEMON_CHECKPOINT
from pyllm.generate import generate


def test_bundled_pokemon_checkpoint_exists_and_generates():
    assert POKEMON_CHECKPOINT.exists(), "run: uv run python scripts/train_pokemon.py"
    model, tok = load(POKEMON_CHECKPOINT)
    out = generate(model, tok, prompt="", max_new_tokens=40,
                   rng=np.random.default_rng(0))
    assert isinstance(out, str) and len(out) > 0
    assert set(out).issubset(set(tok.stoi))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bundled_checkpoint.py -v`
Expected: FAIL — `pokemon.npz` does not exist yet.

- [ ] **Step 3: Write the training script and build the checkpoint**

Create `scripts/train_pokemon.py`:

```python
"""Build-time: train a tiny GPT on the Pokémon corpus and save the checkpoint.

Run once with `uv run python scripts/train_pokemon.py`; the resulting
`src/pyllm/data/pokemon.npz` is committed so the shipped CLI generates instantly.
"""

import numpy as np

from pyllm.checkpoint import save
from pyllm.data import POKEMON_CHECKPOINT
from pyllm.data import load_corpus
from pyllm.models import GPT
from pyllm.tokenizer import CharTokenizer
from pyllm.training import train


def build_and_train(rng):
    """Train a small Pokémon GPT and return (model, tokenizer)."""
    text = load_corpus("pokemon")
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text))
    model = GPT(vocab_size=tokenizer.vocab_size, block_size=16, embed_dim=64,
                num_heads=4, num_layers=3, rng=rng)
    train(model, data, steps=3000, batch_size=32, lr=3e-3, rng=rng,
          log_every=200)
    return model, tokenizer


def main():
    rng = np.random.default_rng(1234)
    model, tokenizer = build_and_train(rng)
    save(POKEMON_CHECKPOINT, model, tokenizer)
    print(f"saved checkpoint to {POKEMON_CHECKPOINT}")


if __name__ == "__main__":
    main()
```

Then build the artifact:

```bash
uv run python scripts/train_pokemon.py
```

Expected: loss prints falling from ~3+ toward ~1.5 or below, then `saved checkpoint to .../pokemon.npz`. Sanity-check the output looks Pokémon-ish:

```bash
uv run pyllm --max-new-tokens 60 2>/dev/null || true   # (CLI arrives in Task 16)
```

(If training is too slow or loss stalls, tune `steps`/`lr`/`embed_dim` — the target is recognizably name-shaped babble, not perfection. Record final numbers in the commit message.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bundled_checkpoint.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit (including the .npz artifact)**

```bash
git add -A
git commit -m "feat: train and bundle tiny Pokémon GPT checkpoint"
```

---

### Task 16: CLI — pyllm REPL + train + tokenize

**Files:**
- Create: `src/pyllm/cli.py`
- Modify: `pyproject.toml` (add `[project.scripts]` + ensure data files ship)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `argparse`, `load` (checkpoint), `POKEMON_CHECKPOINT`, `load_corpus`, `CharTokenizer`, all three models, `train`, `save`, `generate`, `sample_next`.
- Produces `src/pyllm/cli.py` with helper functions (each takes parsed args + an optional `out=print`, returns `int` exit code) and `main(argv=None)`:
  - `run_generate(args, out=print)` — default command: load the bundled checkpoint (or `args.checkpoint`), `generate` with `args.temperature`/`args.top_k`/`args.max_new_tokens`/`args.prompt`, print result.
  - `run_train(args, out=print)` — build a model of `args.model` on `args.corpus` (a file path or the bundled name), train `args.steps`, `save` to `args.out`.
  - `run_tokenize(args, out=print)` — build a `CharTokenizer` from the bundled corpus (or `args.corpus`), print each character of `args.text` alongside its id.
  - `main(argv=None)` — argparse with subcommands `train` and `tokenize`; **no subcommand runs `run_generate`**. Returns the helper's exit code.
- Adds `[project.scripts] pyllm = "pyllm.cli:main"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
from pyllm.cli import main


def test_tokenize_prints_ids(capsys):
    code = main(["tokenize", "--text", "abc"])
    out = capsys.readouterr().out
    assert code == 0
    # each character appears next to a number
    assert "a" in out and "b" in out and "c" in out
    assert any(ch.isdigit() for ch in out)


def test_default_command_generates_from_bundled_checkpoint(capsys):
    code = main(["--max-new-tokens", "20", "--seed", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert len(out.strip()) > 0


def test_train_writes_a_checkpoint(tmp_path):
    dest = tmp_path / "my.npz"
    code = main(["train", "--model", "bigram", "--steps", "5",
                 "--out", str(dest), "--seed", "0"])
    assert code == 0
    assert dest.exists()


def test_generate_is_reproducible_with_seed(capsys):
    main(["--max-new-tokens", "30", "--seed", "42"])
    first = capsys.readouterr().out
    main(["--max-new-tokens", "30", "--seed", "42"])
    second = capsys.readouterr().out
    assert first == second
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.cli'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/cli.py`:

```python
import argparse

import numpy as np

from pyllm.checkpoint import load
from pyllm.checkpoint import save
from pyllm.data import POKEMON_CHECKPOINT
from pyllm.data import load_corpus
from pyllm.generate import generate
from pyllm.models import Bigram
from pyllm.models import GPT
from pyllm.models import MLP
from pyllm.tokenizer import CharTokenizer
from pyllm.training import train


def _read_corpus(source):
    """A corpus arg is either a bundled name ('pokemon') or a path to a file."""
    from pathlib import Path
    path = Path(source)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return load_corpus(source)


def run_generate(args, out=print):
    """Default command: dream up new text from a trained checkpoint."""
    rng = np.random.default_rng(args.seed)
    checkpoint = args.checkpoint if args.checkpoint else POKEMON_CHECKPOINT
    model, tokenizer = load(checkpoint)
    text = generate(model, tokenizer, prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_k=args.top_k, rng=rng)
    out(text)
    return 0


def run_train(args, out=print):
    """Train a model on a corpus and save a checkpoint."""
    rng = np.random.default_rng(args.seed)
    text = _read_corpus(args.corpus)
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text))
    if args.model == "bigram":
        model = Bigram(tokenizer.vocab_size, block_size=8, rng=rng)
    elif args.model == "mlp":
        model = MLP(tokenizer.vocab_size, block_size=3, rng=rng)
    else:
        model = GPT(tokenizer.vocab_size, block_size=16, embed_dim=64,
                    num_heads=4, num_layers=3, rng=rng)
    losses = train(model, data, steps=args.steps, batch_size=args.batch_size,
                   lr=args.lr, rng=rng, log_every=max(1, args.steps // 10),
                   log=out)
    save(args.out, model, tokenizer)
    out(f"final loss {losses[-1]:.4f}; saved checkpoint to {args.out}")
    return 0


def run_tokenize(args, out=print):
    """Show how a piece of text is chopped into token ids (a teaching demo)."""
    text = _read_corpus(args.corpus) if args.corpus else load_corpus("pokemon")
    tokenizer = CharTokenizer(text)
    for char in args.text:
        shown = "\\n" if char == "\n" else char
        out(f"{shown!r} -> {tokenizer.encode(char)[0]}")
    return 0


def main(argv=None):
    """Entry point for `pyllm`: no subcommand generates; `train`/`tokenize` too."""
    parser = argparse.ArgumentParser(prog="pyllm",
                                     description="A tiny LLM you can read.")
    parser.set_defaults(func=run_generate)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)

    sub = parser.add_subparsers()

    p_train = sub.add_parser("train", help="train your own model")
    p_train.add_argument("--model", choices=["bigram", "mlp", "gpt"],
                         default="gpt")
    p_train.add_argument("--corpus", default="pokemon")
    p_train.add_argument("--steps", type=int, default=1000)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=3e-3)
    p_train.add_argument("--out", default="model.npz")
    p_train.add_argument("--seed", type=int, default=None)
    p_train.set_defaults(func=run_train)

    p_tok = sub.add_parser("tokenize", help="show how text becomes tokens")
    p_tok.add_argument("--text", required=True)
    p_tok.add_argument("--corpus", default=None)
    p_tok.set_defaults(func=run_tokenize)

    args = parser.parse_args(argv)
    return args.func(args)
```

Modify `pyproject.toml` — add after the `[project.urls]` block (a new top-level table):

```toml
[project.scripts]
pyllm = "pyllm.cli:main"
```

And ensure the bundled data files ship in the wheel — add under the build config (after the existing `[tool.hatch.build.targets.wheel]` `packages` line):

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/pyllm/data/pokemon_corpus.txt" = "pyllm/data/pokemon_corpus.txt"
"src/pyllm/data/pokemon.npz" = "pyllm/data/pokemon.npz"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (4 passed). Also try it for real:

```bash
uv run pyllm --max-new-tokens 60 --seed 0
uv run pyllm tokenize --text "pika"
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add pyllm CLI (generate REPL + train + tokenize)"
```

---

### Task 17: concept docs (RULE #1) + doc guard

**Files:**
- Create: `docs/concepts/how-a-model-learns.md`
- Create: `docs/concepts/sampling.md`
- Create: `docs/concepts/transformer.md`
- Modify: `tests/test_docs.py`

**Interfaces:**
- Consumes: nothing (prose + a test guard).
- Produces: three kid-friendly concept docs and a test asserting they exist and cover their key ideas. **This plan is NOT done until this task passes** (RULE #1).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_docs.py`:

```python
def test_plan3_concept_docs_exist_and_cover_key_ideas():
    from pathlib import Path

    checks = {
        "docs/concepts/how-a-model-learns.md": ["loss", "gradient", "analogy"],
        "docs/concepts/sampling.md": ["temperature", "top-k", "analogy"],
        "docs/concepts/transformer.md": ["attention", "block", "analogy"],
    }
    for path, ideas in checks.items():
        doc = Path(path)
        assert doc.exists(), f"RULE #1: missing concept doc {path}"
        text = doc.read_text().lower()
        for idea in ideas:
            assert idea in text, f"{path} should explain '{idea}'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docs.py::test_plan3_concept_docs_exist_and_cover_key_ideas -v`
Expected: FAIL (docs don't exist).

- [ ] **Step 3: Write the concept docs**

Create `docs/concepts/how-a-model-learns.md`:

```markdown
# How a model learns (the guessing game)

Imagine a game where you try to guess the next letter of a Pokémon name, over
and over. Every time you guess, a friend tells you how wrong you were. If you
keep score and adjust after every round, you slowly get better. That is exactly
how our model learns.

## The four steps, on repeat

1. **Guess.** The model reads some letters and gives a score to every possible
   next letter.
2. **Measure the surprise (the *loss*).** We look at how much probability the
   model gave the *correct* next letter. Confident and right → tiny loss.
   Confident and wrong → huge loss. This number is the **loss**: lower is better.
3. **Trace the blame (the *gradient*).** We walk backward through every step the
   model took and work out, for each internal dial, "which way should I turn you,
   and how much, to make the loss smaller?" That direction-and-amount is the
   **gradient** (see the breadcrumb-trail analogy in `autograd.md`).
4. **Nudge the dials.** The optimizer turns every dial a tiny step in the
   blame-reducing direction.

Repeat thousands of times and the loss falls: the model has learned.

## A tiny worked example

Train a bigram on `"abcabcabc..."`. At first it guesses randomly (loss ≈ log of
the vocab size). After a few hundred steps it has noticed "after `a` comes `b`,
after `b` comes `c`" and the loss drops close to zero — it has *memorized the
pattern*.

## Why does this matter?

Every AI that "learns from data" — from Pokémon-name dreamers to giant chatbots —
is doing this same loop: guess, measure surprise, trace blame, nudge. The only
differences are how big the model is and how much text it reads.
```

Create `docs/concepts/sampling.md`:

```markdown
# Sampling: how the model chooses what to say next

The model never hands you a single answer. It hands you a *score for every
possible next letter*. Turning those scores into an actual choice is called
**sampling**, and how boldly we choose changes the whole personality of the
output.

## The dice analogy

Think of the scores as a weighted dice: letters the model likes have bigger
sides. **Sampling** is rolling that dice.

- **Temperature** is the boldness dial.
  - `temperature = 0`: never roll — always take the single best guess. Safe, but
    it repeats itself and gets stuck.
  - `temperature = 1`: roll fairly, trusting the model's confidence.
  - `temperature > 1`: flatten the dice so even unlikely letters get a chance —
    wilder, more surprising, sometimes nonsense.
- **top-k** is a guardrail: "only allow the k most likely letters onto the dice."
  It stops the wild rolls from picking something absurd while still letting the
  model be creative among sensible options.

## A tiny worked example

Scores favour `u` after `q`. At `temperature = 0.2` you'll almost always get
`qu...`. Crank it to `temperature = 1.5` and you might see `qx` or `qe` — rarer,
riskier names. `top-k = 3` keeps the surprises to the three best letters only.

## Why does this matter?

Sampling is the knob between *boring-but-correct* and *creative-but-chaotic*.
Every chatbot you've used has a temperature setting under the hood; picking it
well is the difference between a dull answer and a delightful one.
```

Create `docs/concepts/transformer.md`:

```markdown
# How a Transformer is built (stacking the pieces)

A **Transformer** is the engine inside modern language models — including ours.
It sounds fancy, but it's built from a few simple bricks we've already met,
stacked up like Lego.

## The bricks

1. **Token embedding** — turn each letter into a little list of numbers (its
   "meaning"). See `embeddings.md`.
2. **Position embedding** — also add a number for *where* the letter sits,
   because "sat the cat" and "the cat sat" use the same letters in a different
   order.
3. **Attention** — every position looks back at earlier positions and pulls in
   the clues that matter. See `attention.md` ("re-reading a sentence").
4. **Feed-forward** — each position then does a little private thinking on what
   it gathered.
5. **LayerNorm + residuals** — tidy the numbers before each step, and always
   keep the original notes and only *add* edits (a residual connection), so deep
   stacks stay stable.

## The repeatable block

Steps 3–5 together make one **Transformer block**: *communicate* (attention),
then *think* (feed-forward), each wrapped so nothing gets lost. Stack a few
blocks and finish with one last layer that turns each position into scores for
the next letter. That's a **GPT**.

## A tiny worked example

Our bundled Pokémon model is a real GPT with a handful of blocks. Reading
`pi`, the token+position embeddings say *what* and *where*, attention notices
"names that start `pi` often continue `pika...`", the feed-forward sharpens the
hunch, and the final layer scores `k` highest. Roll the dice (see `sampling.md`)
and out comes another letter.

## Why does this matter?

This exact structure — embeddings, stacked attention+feed-forward blocks — is
what powers the biggest AI models in the world. Ours is tiny, but it is the
*same machine*, and you can read every line of it.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_docs.py -v`
Expected: PASS (all doc guards green).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: add kid-friendly concept docs for learning, sampling, transformers"
```

---

## Final verification (run before declaring Plan 3 done)

- [ ] `uv run pytest -q` — the entire suite is green.
- [ ] `uv run ruff check .` — clean.
- [ ] `uv run ruff format --check .` — clean (or run `uv run ruff format .` and recommit).
- [ ] `uv run pyright` — clean.
- [ ] `uv run pyllm --seed 0 --max-new-tokens 60` — prints Pokémon-ish names instantly.
- [ ] `uv run pyllm train --model bigram --steps 50 --out /tmp/b.npz` — trains and saves.
- [ ] `uv run pyllm tokenize --text "pikachu"` — shows char→id mapping.
- [ ] Update `README.md` with an example session and confirm the roadmap's Plan 3 row is satisfied (models, optimizer, training loop, sampling, checkpoints, CLI, bundled Pokémon corpus + checkpoint).
```
