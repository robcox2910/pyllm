# PyLLM

An educational large language model built from scratch in Python.

Part of the "from scratch in Python" series. Built incrementally with TDD,
every concept explained with analogies a 12-year-old can follow. The only
dependency is numpy (for fast array math) — the autograd engine, neural-net
layers, attention, the Transformer, the tokenizer, the optimizer, the training
loop, and sampling are all ours.

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
beedrill
pidgey
```

Turn the boldness dial with `--temperature` (0 = safe/greedy, higher = wilder)
and add a `--top-k` guardrail. Start from a prompt with `--prompt "pi"`.

## Train your own model

Pick a rung on the teaching ladder (`bigram`, `mlp`, or `gpt`) and point it at
any text file (defaults to the bundled Pokémon corpus):

```console
$ uv run pyllm train --model gpt --steps 2000 --out mine.npz
step 0: loss 3.31
...
final loss 0.42; saved checkpoint to mine.npz

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
let total = total * z * (35 % 87)
print(compute())
let a = (y - result) % (37 + result)
...
parse rate: 33.3% of 3 programs
```

The bundled checkpoint is *tiny* (a few-minute char-level GPT), so it writes
Pebble-*flavoured* code and only some of it parses — and that's the whole point:
the parser is an honest, objective judge, so you can watch the score go up as you
train longer or bigger. Almost no LLM project can grade its own output like this.

Grow a fresh synthetic corpus with `uv run pyllm gen-corpus`. Live scoring needs
the optional `pebble-lang` dependency (`uv sync --all-extras`); generation and
training work without it. See [`docs/concepts/grow-your-own-data.md`](docs/concepts/grow-your-own-data.md).

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

Kid-friendly explanations live in [`docs/concepts/`](docs/concepts/):
tokens, embeddings, attention, the breadcrumb trail (autograd), how a model
learns, sampling, and how a Transformer is built.

## Related projects

Part of the "from scratch in Python" series — PyLLM is **the brain**.
