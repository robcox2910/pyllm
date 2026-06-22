# PyLLM Implementation Roadmap

> The PyLLM spec (`docs/superpowers/specs/2026-06-22-pyllm-design.md`) is built
> as **five sequential plans**. Each plan produces working, testable software on
> its own and builds on the one before it. Implement them in order.

| # | Plan | Deliverable (what works at the end) |
|---|------|-------------------------------------|
| 1 | **Autograd foundation** — `2026-06-22-pyllm-01-autograd.md` | A `Tensor` class wrapping numpy with reverse-mode autodiff. `loss.backward()` produces correct gradients, verified against finite-difference gradient checking. Project scaffolding (pyproject, package, tooling) done. |
| 2 | **nn building blocks + tokenizers** | `Module` base, `Linear`, `Embedding`, `LayerNorm`, `Dropout`, `softmax`, `cross_entropy`, `gelu`, a self-attention head, multi-head attention, `TransformerBlock` — each unit-tested. Char tokenizer + from-scratch BPE tokenizer. |
| 3 | **Model ladder + training + generation (Pokémon end-to-end)** | `bigram`, `mlp`, `gpt` models; from-scratch SGD + Adam; batching + training loop; sampling (greedy/temperature/top-k); checkpoint save/load; `pyllm` / `pyllm train` / `pyllm tokenize` CLI. Bundled Pokémon corpus + a tiny trained checkpoint that generates on launch. |
| 4 | **Pebble flagship** | `pebble/` program generator (random valid ASTs → pebble-lang formatter), corpus harvester, parser-based validity scorer; bundled Pebble corpus + checkpoint; `pyllm pebble` and `pyllm gen-corpus` CLI commands. |
| 5 | **PyStack integration + docs polish** | Pebble `import "llm"` module (`llm_generate`, checkpoint/sampling helpers); PyLLM added as project #11 ("the brain") to PyStack; full `docs/concepts/` kid-friendly write-ups; README with example session + series table. |

## RULE #1 — Child-friendly documentation about EVERYTHING

**This is the single most important rule of the entire series, above all others.**
Every module, every class, every function, and every concept must be explained so
a curious 12-year-old can follow it, using real-world analogies. This is not a
final polish step — it ships *with* each piece of code.

Concretely, in every plan:

- **Every public class/function gets a docstring written for a child** — what it
  is, using an analogy, before any jargon. (See the autograd docstrings in
  Plan 1 for the tone: "a numpy array that remembers how to compute its own
  gradient", "we nudge each element up and down and measure how the output
  changes".)
- **Every plan ends with a `docs/concepts/<topic>.md` task** — a standalone
  kid-friendly explanation of the big idea that plan introduced, with an analogy,
  a tiny worked example, and a "why does this matter?" section. A plan is **not
  done** until its concept doc exists.
- **Tests double as readable examples** — name them so they read like sentences
  describing the behaviour.

If a reviewer can't understand a module from its docs without reading the code,
the task is not complete.

## Other conventions enforced by every plan

These mirror the rest of the series and are repeated as Global Constraints in
each plan:

- **Python 3.14**, managed with `uv`. Run things via `uv run`.
- **Dependencies:** `numpy` only (plus dev/test/docs tooling). Nothing else.
- **TDD:** every behaviour gets a failing test first.
- **ruff** + **pyright** must stay clean.
- **No `TYPE_CHECKING`, no `from __future__ import annotations`.**
- Frequent, small commits (one per task minimum).

## Concept docs each plan must deliver

- Plan 1: `docs/concepts/autograd.md` — *The breadcrumb trail* (how a network
  remembers what it did so it can learn from mistakes).
- Plan 2: `docs/concepts/tokens.md`, `embeddings.md`, `attention.md` — tokens as
  Lego bricks, embeddings as a map of meaning, attention as re-reading a sentence.
- Plan 3: `docs/concepts/how-a-model-learns.md`, `sampling.md`, `transformer.md`.
- Plan 4: `docs/concepts/grow-your-own-data.md` — harvest, generate, grade.
- Plan 5: full `docs/concepts/` polish + README with example session.

## Grammar-constrained decoding

Deferred stretch goal (spec §6). Not in any of the five plans; add a sixth plan
only after Plan 4 ships and proves valuable.
