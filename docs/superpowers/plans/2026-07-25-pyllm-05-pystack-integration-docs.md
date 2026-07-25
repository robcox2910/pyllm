# PyLLM Plan 5 — PyStack Integration + Docs Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the loop of the series — let any Pebble program running inside PyStack `import "llm"` and ask PyLLM to write text (including *more Pebble*), and add PyLLM as the 11th project ("the brain") to PyStack. Plus final docs polish: a clean public generation API, a `docs/concepts/` index, and a README with the full "from scratch in Python" series table.

**Architecture:** The `import "llm"` module is realized as a **PyStack plugin**, exactly like every other sibling module (`crypto`, `git`, `net`, …). PyStack owns the integration layer: a plugin's `pebble_stdlib()` returns a `StdlibModule` that PyStack injects into Pebble's global `STDLIB_MODULES` at boot. This means **no change to pebble-lang core** and **no circular dependency** (pebble-lang never imports pyllm; pyllm optionally imports pebble-lang for scoring; pystack depends on both). The plugin calls a small, cached public API added to pyllm (`pyllm.api`). Work spans two repos: **pyllm** (Part A) and **pystack** (Part B).

**Tech Stack:** Python 3.14, numpy, pytest, ruff, pyright, uv. pyllm builds on Plan 4 (merged to `main`). pystack depends on the sibling projects via `[tool.uv.sources]` path entries.

## Global Constraints

- **RULE #1 — Child-friendly docs about EVERYTHING.** Every public class/function gets an analogy-first docstring. This plan is not done until the docs polish (Part A, Task A3) is complete. This rule outranks all others.
- **Python 3.14**, managed with `uv`. Run everything via `uv run`.
- **Dependencies:** pyllm's shipped runtime stays numpy-only. The pystack plugin imports pyllm; that dependency lives in the **pystack** repo, not pyllm.
- **TDD:** write the failing test first, watch it fail, then implement.
- **ruff** + **pyright** must stay clean after every task (in whichever repo the task touches).
- **No `TYPE_CHECKING`, no `from __future__ import annotations`.**
- **Two repos, two branches, two PRs.** Part A commits to pyllm branch `plan-05-pystack-integration`. Part B commits to a new pystack branch `add-pyllm-plugin`. Do not mix a repo's changes into the other's commits.
- Frequent commits: one per task minimum.

## Interfaces this plan consumes

**pyllm (existing):**
- `from pyllm.checkpoint import load` → `(model, tokenizer)`.
- `from pyllm.generate import generate` → `generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, rng) -> str`.
- `from pyllm.data import PEBBLE_CHECKPOINT, POKEMON_CHECKPOINT` (bundled `.npz` paths).

**pystack (verified):**
- `from pystack.plugins.base import Plugin, PluginInfo, ShellCommand, pebble_handler`. Hooks: `info() -> PluginInfo`, `pebble_stdlib() -> StdlibModule | None`, `pebble_module_name() -> str`, `shell_commands() -> list[ShellCommand]`.
- `from pebble.stdlib import StdlibModule` — `StdlibModule(functions={name: (arity, handler)}, constants={})`; a handler is `Callable[[list[Value]], Value]` where a Pebble string is a Python `str`.
- `@pebble_handler` wraps a handler so exceptions become `"error: <msg>"` strings.
- Plugins are registered in `src/pystack/environment.py` `_register_all_plugins()` (a hardcoded list) and, on register, their stdlib is injected via `STDLIB_MODULES[plugin.pebble_module_name()] = plugin.pebble_stdlib()`.
- `from pystack.environment import PyStackEnvironment` — `env.run_pebble_source(source) -> str` (captured stdout); `env.shutdown()`.
- pystack `pyproject.toml`: siblings listed in `[project].dependencies` + `[tool.uv.sources]` path entries; `requires-python = ">=3.14"`.

---

## Part A — pyllm repo (branch `plan-05-pystack-integration`)

### Task A1: Public generation API (`pyllm.api`)

**Files:**
- Create: `src/pyllm/api.py`
- Modify: `src/pyllm/__init__.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `load`, `generate`, `PEBBLE_CHECKPOINT`, `POKEMON_CHECKPOINT`.
- Produces:
  - `generate_pebble(prompt="let ", max_new_tokens=120, temperature=0.7, seed=None) -> str` — load the bundled Pebble checkpoint (cached) and generate.
  - `generate_pokemon(prompt="", max_new_tokens=60, temperature=0.8, seed=None) -> str` — same for the Pokémon checkpoint.
  - Both cache the loaded `(model, tokenizer)` per checkpoint path so repeated calls don't reload. Re-exported from `pyllm` as `pyllm.generate_pebble` / `pyllm.generate_pokemon`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_api.py`:

```python
from pyllm.api import generate_pebble
from pyllm.api import generate_pokemon


def test_generate_pokemon_returns_known_charset():
    out = generate_pokemon(prompt="", max_new_tokens=30, seed=0)
    assert isinstance(out, str) and len(out) > 0


def test_generate_pebble_starts_with_prompt():
    out = generate_pebble(prompt="let ", max_new_tokens=20, seed=0)
    assert out.startswith("let ")


def test_generation_is_reproducible_with_seed():
    a = generate_pebble(prompt="let ", max_new_tokens=30, seed=7)
    b = generate_pebble(prompt="let ", max_new_tokens=30, seed=7)
    assert a == b


def test_checkpoint_is_cached_not_reloaded():
    import pyllm.api as api

    generate_pokemon(prompt="", max_new_tokens=5, seed=1)
    from pyllm.data import POKEMON_CHECKPOINT
    assert str(POKEMON_CHECKPOINT) in api._CACHE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.api'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/api.py`:

```python
"""A friendly one-call way to make PyLLM write something.

Loading a model from disk and wiring up sampling every time would be a chore, so
this is the "just give me some text" front door: call `generate_pokemon()` to
dream up creatures, or `generate_pebble()` to write Pebble code. The trained
brain is loaded once and kept warm (cached) so the second call is instant. This
is also the door PyStack knocks on to let Pebble programs use PyLLM.
"""

import numpy as np

from pyllm.checkpoint import load
from pyllm.data import PEBBLE_CHECKPOINT
from pyllm.data import POKEMON_CHECKPOINT
from pyllm.generate import generate

_CACHE = {}


def _load_cached(path):
    """Load a checkpoint once and remember it, so later calls are instant."""
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = load(path)
    return _CACHE[key]


def generate_pokemon(prompt="", max_new_tokens=60, temperature=0.8, seed=None):
    """Dream up new Pokémon-ish names using the bundled Pokémon brain."""
    model, tokenizer = _load_cached(POKEMON_CHECKPOINT)
    return generate(model, tokenizer, prompt=prompt,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                    rng=np.random.default_rng(seed))


def generate_pebble(prompt="let ", max_new_tokens=120, temperature=0.7, seed=None):
    """Write Pebble-flavoured code using the bundled Pebble brain."""
    model, tokenizer = _load_cached(PEBBLE_CHECKPOINT)
    return generate(model, tokenizer, prompt=prompt,
                    max_new_tokens=max_new_tokens, temperature=temperature,
                    rng=np.random.default_rng(seed))
```

Update `src/pyllm/__init__.py`:

```python
"""PyLLM: an educational large language model built from scratch."""

from pyllm.api import generate_pebble
from pyllm.api import generate_pokemon

__all__ = ["generate_pebble", "generate_pokemon"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add pyllm.api public generation helpers (cached)"
```

---

### Task A2: docs/concepts index

**Files:**
- Create: `docs/concepts/README.md`
- Test: `tests/test_docs.py` (MODIFY)

**Interfaces:**
- Produces: an index page linking all eight concept docs in reading order, and a test guard that the index exists and links every concept doc.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_docs.py`:

```python
def test_concepts_index_links_every_doc():
    from pathlib import Path

    index = Path("docs/concepts/README.md")
    assert index.exists(), "concepts index page must exist"
    text = index.read_text()
    for doc in [
        "autograd.md", "tokens.md", "embeddings.md", "attention.md",
        "how-a-model-learns.md", "sampling.md", "transformer.md",
        "grow-your-own-data.md",
    ]:
        assert doc in text, f"concepts index should link {doc}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docs.py::test_concepts_index_links_every_doc -v`
Expected: FAIL (index doesn't exist).

- [ ] **Step 3: Write the index**

Create `docs/concepts/README.md`:

```markdown
# PyLLM concepts, explained for a curious 12-year-old

These pages explain every big idea in PyLLM using real-world analogies — no jargon
first. Read them roughly in this order; each builds on the last.

1. [The breadcrumb trail (autograd)](autograd.md) — how a network remembers what
   it did so it can learn from mistakes.
2. [What is a token?](tokens.md) — chopping text into Lego bricks.
3. [What is an embedding?](embeddings.md) — giving each token a place on a map of
   meaning.
4. [Attention is re-reading a sentence](attention.md) — how a token decides which
   earlier words matter.
5. [How a Transformer is built](transformer.md) — stacking the pieces into a GPT.
6. [How a model learns](how-a-model-learns.md) — loss, gradients, the training loop.
7. [Sampling](sampling.md) — how the model chooses what to say next (temperature,
   top-k).
8. [When there's no data, grow your own](grow-your-own-data.md) — the Pebble
   corpus: harvest, generate, and grade.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_docs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: add concepts index page"
```

---

### Task A3: README series table + polish

**Files:**
- Modify: `README.md`

**Interfaces:**
- Produces: a "from scratch in Python" series table in the README's "Related projects" section (PyLLM = the brain, project #11), a link to the concepts index, and the `import "llm"` PyStack example. No test (prose), but run the doc suite to confirm nothing regressed.

- [ ] **Step 1: Replace the stub "Related projects" section**

In `README.md`, replace the `## Related projects` section with a proper series table. Use the authoritative project list from PyStack's README "Related Projects" table (`../pystack/README.md`) — read it first and mirror the rows, adding PyLLM. Format:

```markdown
## Related projects — "from scratch in Python"

PyLLM is **the brain** of a family of educational projects, each building a real
system from scratch to teach how it works. They're tied together by
**[PyStack](https://github.com/robcox2910/pystack)**, where a Pebble program can
even `import "llm"` and ask PyLLM to write more code.

| Project | What it teaches | Repository |
|---------|-----------------|------------|
| PyOS | Operating systems | [robcox2910/py-os](https://github.com/robcox2910/py-os) |
| Pebble | Compilers & languages | [robcox2910/pebble-lang](https://github.com/robcox2910/pebble-lang) |
| … | … | … |
| **PyLLM** | **Language models (the brain)** | **[robcox2910/pyllm](https://github.com/robcox2910/pyllm)** |
| PyStack | Full-stack integration | [robcox2910/pystack](https://github.com/robcox2910/pystack) |
```

(Fill the `…` rows verbatim from PyStack's table so the two stay consistent.)

- [ ] **Step 2: Add the `import "llm"` example**

Add a short subsection under the flagship or related-projects section:

```markdown
### Inside PyStack: Pebble asks PyLLM to write code

Once PyLLM is installed as a PyStack plugin, any Pebble program can call it:

```pebble
import "llm"
let code = llm_generate("a function that adds two numbers")
print(code)
```
```

- [ ] **Step 3: Link the concepts index**

Update the "Learn the ideas" section to point at [`docs/concepts/`](docs/concepts/README.md) and mention all eight topics including grow-your-own-data.

- [ ] **Step 4: Verify docs suite + commit**

Run: `uv run pytest tests/test_docs.py -v`
Expected: PASS.

```bash
git add -A
git commit -m "docs: add series table, import-llm example, concepts link to README"
```

- [ ] **Step 5: Final pyllm gate**

Run: `uv run pytest -q && uv run ruff check . && uv run pyright`
Expected: all green (scope ruff/format to Plan 5 files if pre-existing files are noisy).

Then push and open the pyllm PR:

```bash
git push -u origin plan-05-pystack-integration
gh pr create --base main --title "Plan 5 (pyllm side): public API + docs polish" \
  --body "Public pyllm.api generation helpers, concepts index, README series table. Pairs with the PyStack LLMPlugin PR."
```

---

## Part B — pystack repo (branch `add-pyllm-plugin`)

> All Part B tasks run in `/Users/robcox/development/python/pystack`. Create the branch first:
> `git -C ../pystack checkout -b add-pyllm-plugin`

### Task B1: Depend on pyllm

**Files:**
- Modify: `../pystack/pyproject.toml`
- Test: `../pystack/tests/test_llm_plugin.py` (created in B2; here just wire the dep)

**Interfaces:**
- Produces: pystack depends on `pyllm` via a path source, so `import pyllm` works inside pystack.

- [ ] **Step 1: Add the dependency**

In `../pystack/pyproject.toml`, add `"pyllm"` to `[project].dependencies` and a source entry:

```toml
[tool.uv.sources]
# ... existing entries ...
pyllm = { path = "../pyllm", editable = true }
```

- [ ] **Step 2: Install**

Run: `uv sync --all-extras` (in `../pystack`)
Expected: installs `pyllm` (editable) from `../pyllm`.

- [ ] **Step 3: Verify import + commit**

Run: `cd ../pystack && uv run python -c "import pyllm; print(pyllm.generate_pokemon(max_new_tokens=5, seed=0))"`
Expected: prints a short string.

```bash
git -C ../pystack add -A
git -C ../pystack commit -m "build: depend on pyllm (the brain) via path source"
```

---

### Task B2: LLMPlugin + wiring

**Files:**
- Create: `../pystack/src/pystack/plugins/llm_plugin.py`
- Modify: `../pystack/src/pystack/environment.py`
- Test: `../pystack/tests/test_llm_plugin.py`

**Interfaces:**
- Consumes: `Plugin`, `PluginInfo`, `ShellCommand`, `pebble_handler`, `StdlibModule`, `pyllm.generate_pebble`, `pyllm.generate_pokemon`, `PyStackEnvironment`.
- Produces: `LLMPlugin(Plugin)` exposing the Pebble `"llm"` module with `llm_generate(prompt)` (writes Pebble) and `llm_dream(prompt)` (Pokémon), plus a `dream` shell command. Registered in `environment.py`.

- [ ] **Step 1: Write the failing test**

Create `../pystack/tests/test_llm_plugin.py`:

```python
from pathlib import Path

from pystack.environment import PyStackEnvironment


def test_pebble_can_import_llm_and_generate(tmp_path: Path) -> None:
    env = PyStackEnvironment(db_path=tmp_path)
    try:
        out = env.run_pebble_source(
            'import "llm"\nlet code = llm_generate("add two numbers")\nprint(code)'
        )
        assert out.strip()  # non-empty generated text
    finally:
        env.shutdown()


def test_pebble_can_dream_pokemon(tmp_path: Path) -> None:
    env = PyStackEnvironment(db_path=tmp_path)
    try:
        out = env.run_pebble_source('import "llm"\nprint(llm_dream("pik"))')
        assert out.strip()
    finally:
        env.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run (in `../pystack`): `uv run pytest tests/test_llm_plugin.py -v`
Expected: FAIL — `import "llm"` unresolved (module not registered).

- [ ] **Step 3: Write the plugin**

Create `../pystack/src/pystack/plugins/llm_plugin.py`:

```python
"""PyStack plugin exposing PyLLM (the brain) to Pebble as the ``llm`` module.

This is the loop-closing plugin: it lets a Pebble program `import "llm"` and ask
PyLLM to write text -- including *more Pebble code*. The brain of the series
writing the language of the series. It calls PyLLM's cached public API, so the
model is loaded once and reused.
"""

import pyllm
from pebble.builtins import Value as PebbleValue
from pebble.stdlib import StdlibModule

from pystack.plugins.base import Plugin
from pystack.plugins.base import PluginInfo
from pystack.plugins.base import ShellCommand
from pystack.plugins.base import pebble_handler


@pebble_handler
def _llm_generate(args: list[PebbleValue]) -> PebbleValue:
    """Write Pebble-flavoured code from a prompt string."""
    prompt = str(args[0]) if args else "let "
    return pyllm.generate_pebble(prompt=prompt, max_new_tokens=120)


@pebble_handler
def _llm_dream(args: list[PebbleValue]) -> PebbleValue:
    """Dream up Pokémon-ish names from a prompt string."""
    prompt = str(args[0]) if args else ""
    return pyllm.generate_pokemon(prompt=prompt, max_new_tokens=40)


class LLMPlugin(Plugin):
    """Wires PyLLM into PyStack as the Pebble ``llm`` module (and a shell command)."""

    def info(self) -> PluginInfo:
        return PluginInfo(
            name="PyLLM",
            description="A from-scratch language model -- the brain of the series.",
        )

    def pebble_module_name(self) -> str:
        return "llm"

    def pebble_stdlib(self) -> StdlibModule:
        return StdlibModule(
            functions={
                "llm_generate": (1, _llm_generate),
                "llm_dream": (1, _llm_dream),
            },
            constants={},
        )

    def shell_commands(self) -> list[ShellCommand]:
        return [
            ShellCommand(
                name="dream",
                handler=lambda args: pyllm.generate_pokemon(
                    prompt=" ".join(args), max_new_tokens=40
                ),
                help_text="Dream up new Pokémon names with PyLLM.",
            )
        ]
```

Wire it into `../pystack/src/pystack/environment.py`: add the import alongside the other plugin imports, and append `LLMPlugin()` to the `plugins` list in `_register_all_plugins()`:

```python
from pystack.plugins.llm_plugin import LLMPlugin
```
```python
        plugins: list[Plugin] = [
            CryptoPlugin(),
            # ... existing plugins ...
            ColDBPlugin(),
            LLMPlugin(),
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run (in `../pystack`): `uv run pytest tests/test_llm_plugin.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git -C ../pystack add -A
git -C ../pystack commit -m "feat: add LLMPlugin exposing PyLLM as the Pebble 'llm' module"
```

---

### Task B3: Update PyStack README + registration tests

**Files:**
- Modify: `../pystack/README.md`
- Modify: `../pystack/tests/test_integration_plugins.py`

**Interfaces:**
- Produces: PyLLM added to PyStack's project tables ("the brain"), stale counts updated, and the plugin-registration tests updated (module list gains `"llm"`, plugin count bumped, `"PyLLM"` asserted present).

- [ ] **Step 1: Update the registration tests (write the failing expectation first)**

In `../pystack/tests/test_integration_plugins.py`:
- Add `"llm"` to the `expected_modules` list in `test_all_modules_registered`.
- Bump `expected_count` by 1 in `test_plugins_in_registry` and add `assert "PyLLM" in names`.

Run (in `../pystack`): `uv run pytest tests/test_integration_plugins.py -k "registered or registry" -v`
Expected: these two pass **only after** B2's plugin is wired (it is, since B2 precedes B3) — confirming the count/module list now includes PyLLM.

- [ ] **Step 2: Update the README tables**

In `../pystack/README.md`:
- Add a row to the "School Analogy" table: `| **PyLLM** | Language model | The imagination / the brain |`.
- Add a row to the "Related Projects" table: `| PyLLM | Language models | [robcox2910/pyllm](https://github.com/robcox2910/pyllm) |`.
- Add a row to the plugin table: `| LLMPlugin | `llm` | `dream` |`.
- Update the stale headline count and the "N active plugins" / "N Pebble modules" counts to reflect the added plugin (read the current numbers and increment; reconcile the "TEN" wording).

- [ ] **Step 3: Verify the whole pystack suite + gate**

Run (in `../pystack`):
```bash
uv run pytest -q
uv run ruff check .
uv run pyright
```
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git -C ../pystack add -A
git -C ../pystack commit -m "docs+test: add PyLLM (the brain) to PyStack tables and registration tests"
```

- [ ] **Step 5: Push + PR (pystack)**

```bash
git -C ../pystack push -u origin add-pyllm-plugin
gh -C ../pystack pr create --base main --title "Add PyLLM as the 11th project (the brain): import \"llm\" plugin" \
  --body "Adds an LLMPlugin exposing PyLLM to Pebble as the 'llm' module (llm_generate / llm_dream), a 'dream' shell command, the pyllm path dependency, README rows, and registration tests."
```
(If `gh -C` is unsupported, `cd ../pystack` first.)

---

## Final verification (run before declaring Plan 5 done)

- [ ] **pyllm repo:** `uv run pytest -q`, `ruff check`, `pyright` all green; PR opened.
- [ ] **pystack repo:** `uv run pytest -q`, `ruff check`, `pyright` all green; PR opened.
- [ ] End-to-end proof (in `../pystack`): `uv run python -c "from pystack.environment import PyStackEnvironment; e=PyStackEnvironment(db_path='/tmp/ps'); print(e.run_pebble_source('import \"llm\"\nprint(llm_generate(\"add\"))')); e.shutdown()"` — prints generated Pebble.
- [ ] Roadmap Plan 5 row satisfied: `import "llm"` module (via plugin), PyLLM added as project #11 to PyStack, full `docs/concepts/` with index, README series table + example session. This completes the five-plan roadmap.
```
