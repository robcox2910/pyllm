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

## Conventions enforced by every plan

These mirror the rest of the series and are repeated as Global Constraints in
each plan:

- **Python 3.14**, managed with `uv`. Run things via `uv run`.
- **Dependencies:** `numpy` only (plus dev/test/docs tooling). Nothing else.
- **TDD:** every behaviour gets a failing test first.
- **ruff** + **pyright** must stay clean.
- **No `TYPE_CHECKING`, no `from __future__ import annotations`.**
- **Docs-first:** kid-friendly analogies; concepts documented as features land.
- Frequent, small commits (one per task minimum).

## Grammar-constrained decoding

Deferred stretch goal (spec §6). Not in any of the five plans; add a sixth plan
only after Plan 4 ships and proves valuable.
