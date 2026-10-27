---
title: "Build your own GPT from scratch in Python"
description: "A small GPT-style language model built from scratch in Python and NumPy: autograd, attention, a Transformer, a tokenizer, training and sampling."
---

# PyLLM: Build your own GPT from scratch in Python

An educational large language model built from scratch in Python.

Part of the "from scratch in Python" series. Built incrementally with TDD,
every concept explained with analogies a 12-year-old can follow. The only
dependency is numpy (for fast array math) — the autograd engine, neural-net
layers, attention, the Transformer, the tokenizer, the optimizer, the training
loop, and sampling are all ours.

## Install

```bash
pip install fromscratch-pyllm
```

Or get the whole series at once with `pip install fromscratch-pystack`. To read the code or run the tests, clone it from [GitHub](https://github.com/robcox2910/pyllm).

## Quick Start

```bash
uv sync --all-extras
uv run pytest
```

## Dream up new Pokémon (generates instantly)

PyLLM ships with a tiny pre-trained GPT, so it invents Pokémon-ish names the
moment you launch it — no training wait:

```console
$ uv run pyllm --seed 0 --max-new-tokens 40
machop
machoke
machamp
bellsprout
weepin
```

Turn the boldness dial with `--temperature` (0 = safe/greedy, higher = wilder)
and add a `--top-k` guardrail. Start from a prompt with `--prompt "pi"`.

## Train your own model

Pick a rung on the teaching ladder (`bigram`, `mlp`, or `gpt`) and point it at
any text file (defaults to the bundled Pokémon corpus):

```console
$ uv run pyllm train --model gpt --steps 2000 --seed 0 --out mine.npz
step 0: loss 3.4070
...
final loss 0.2996; saved checkpoint to mine.npz

$ uv run pyllm --checkpoint mine.npz --max-new-tokens 60
```

## The flagship: teach PyLLM to write Pebble

There's no corpus of [Pebble](https://github.com/robcox2910/pebble-lang) code in
the wild — so we *grow our own*. A generator emits **random-but-provably-valid**
Pebble ASTs (100% parse under Pebble's own parser), a harvester scrapes real
snippets from Pebble's docs, and — because we own the language toolchain — we
**grade** the trained model by running its output back through Pebble's real
parser and reporting what percentage parses:

```console
$ uv run pyllm pebble --seed 7 --temperature 0.5 --score
let total = n * (5 + n)
        let a = 6
    }
    let a = result * (6 - result)
...
parse rate: 0.0% of 1 programs
```

The bundled checkpoint is *tiny* (a few-minute char-level GPT), so it writes
Pebble-*flavoured* code and only some of it parses — and that's the whole point:
the parser is an honest, objective judge, so you can watch the score go up as you
train longer or bigger. Almost no LLM project can grade its own output like this.

### Call in the referee: always-valid Pebble

Add `--constrained` and a from-scratch grammar **referee** vetoes every character
that would break Pebble's rules, so the model rolls again until it picks a legal
one. Same tiny brain — but now every program it returns parses:

```console
$ uv run pyllm pebble --seed 7 --constrained --score
let total = result + j * 3

parse rate: 100.0% of 1 programs
```

From Python: `generate_pebble(constrained=True)`. See
[`docs/concepts/the-referee.md`](https://github.com/robcox2910/pyllm/blob/main/docs/concepts/the-referee.md).

Grow a fresh synthetic corpus with `uv run pyllm gen-corpus`. Live scoring needs
the optional `fromscratch-pebble` dependency (`uv sync --all-extras`); generation and
training work without it. See [`docs/concepts/grow-your-own-data.md`](https://github.com/robcox2910/pyllm/blob/main/docs/concepts/grow-your-own-data.md).

## See how text becomes tokens

```console
$ uv run pyllm tokenize --text "pika"
'p' -> 16
'i' -> 9
'k' -> 11
'a' -> 1
```

## The teaching ladder

Each rung is a real, trainable language model that fixes a weakness of the one
below it, so you feel *why* each idea exists:

1. **Bigram** — "what character usually comes next?" A lookup table.
2. **MLP** — a small neural net over the last few characters (embeddings + a
   hidden layer).
3. **GPT** — stacked Transformer blocks (multi-head attention + feed-forward +
   LayerNorm + residuals) with positional information. The real thing.

## Learn the ideas

Kid-friendly explanations live in [`docs/concepts/`](https://github.com/robcox2910/pyllm/blob/main/docs/concepts/README.md):
the breadcrumb trail (autograd), tokens, embeddings, attention, how a Transformer
is built, how a model learns, sampling, and — the flagship lesson — growing your
own data.

## Developing alongside a sibling

PyLLM uses a released version of [Pebble](https://github.com/robcox2910/pebble-lang)
from PyPI (`fromscratch-pebble`, the optional `pebble` extra). To try an
unreleased Pebble change, install your local checkout over the released one:

```bash
uv pip install -e ../pebble-lang
uv run --no-sync pytest
```

Use `uv run --no-sync` while you do this: a plain `uv run` (like `uv sync`)
puts the released version back, which is how you undo the override.

## Releasing

1. Merge the change to `main`.
2. Release it: `gh release create vX.Y.Z --target main --generate-notes`
   (or push a `vX.Y.Z` tag); the release workflow publishes it to PyPI.
   A change to the public API bumps the minor version (`v0.1.0` → `v0.2.0`);
   anything else bumps the patch (`v0.1.0` → `v0.1.1`).
3. Projects that use PyLLM (like [PyStack](https://github.com/robcox2910/pystack))
   upgrade by raising their `fromscratch-pyllm>=X.Y.Z` floor and running
   `uv lock --upgrade-package fromscratch-pyllm`, in a pull request so CI checks the pair.

## PyLLM inside PyStack

PyLLM is **the brain** of a family of educational projects, each building a real
system from scratch to teach how it works. They're tied together by
**[PyStack](https://github.com/robcox2910/pystack)**, where a Pebble program can
even `import "llm"` and ask PyLLM to write more code.

Once PyLLM is installed as a PyStack plugin, any Pebble program can call it:

```pebble
import "llm"
let code = llm_generate("a function that adds two numbers")
print(code)
```
