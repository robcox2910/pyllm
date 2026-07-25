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

Kid-friendly explanations live in [`docs/concepts/`](docs/concepts/README.md):
the breadcrumb trail (autograd), tokens, embeddings, attention, how a Transformer
is built, how a model learns, sampling, and — the flagship lesson — growing your
own data.

## Related projects — "from scratch in Python"

PyLLM is **the brain** of a family of educational projects, each building a real
system from scratch to teach how it works. They're tied together by
**[PyStack](https://github.com/robcox2910/pystack)**, where a Pebble program can
even `import "llm"` and ask PyLLM to write more code.

| Project | What it teaches | Repository |
|---------|-----------------|------------|
| PyOS | Operating systems | [robcox2910/py-os](https://github.com/robcox2910/py-os) |
| Pebble | Compilers and programming languages | [robcox2910/pebble-lang](https://github.com/robcox2910/pebble-lang) |
| PyDB | Relational databases | [robcox2910/pydb](https://github.com/robcox2910/pydb) |
| PyCrypt | Cryptography | [robcox2910/pycrypt](https://github.com/robcox2910/pycrypt) |
| PyWeb | HTTP web servers | [robcox2910/pyweb](https://github.com/robcox2910/pyweb) |
| PyNet | Computer networking | [robcox2910/pynet](https://github.com/robcox2910/pynet) |
| PyGit | Version control | [robcox2910/pygit](https://github.com/robcox2910/pygit) |
| PySearch | Search engines | [robcox2910/pysearch](https://github.com/robcox2910/pysearch) |
| PyMQ | Message queues | [robcox2910/pymq](https://github.com/robcox2910/pymq) |
| **PyLLM** | **Language models (the brain)** | **[robcox2910/pyllm](https://github.com/robcox2910/pyllm)** |
| PyStack | Full-stack integration | [robcox2910/pystack](https://github.com/robcox2910/pystack) |

### Inside PyStack: Pebble asks PyLLM to write code

Once PyLLM is installed as a PyStack plugin, any Pebble program can call it:

```pebble
import "llm"
let code = llm_generate("a function that adds two numbers")
print(code)
```
