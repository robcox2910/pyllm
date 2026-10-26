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
bellsprout
weepin
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
let result = 62
    }
    if count < 47 {
        let c = 51 + (83 + 32)
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
let total = n * (63 - 99)

let result = c
fount = 55 + (n - b) % (6 - b)

parse rate: 100.0% of 2 programs
```

From Python: `generate_pebble(constrained=True)`. See
[`docs/concepts/the-referee.md`](docs/concepts/the-referee.md).

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

## Developing alongside a sibling

PyLLM uses a released version of [Pebble](https://github.com/robcox2910/pebble-lang),
pinned to a tag in `pyproject.toml` (`[tool.uv.sources]`). To try an unreleased
Pebble change, install your local checkout over the pinned one:

```bash
uv pip install -e ../pebble-lang
uv run --no-sync pytest
```

Use `uv run --no-sync` while you do this: a plain `uv run` (like `uv sync`)
puts the pinned release back, which is how you undo the override.

## Releasing

1. Merge the change to `main`.
2. Tag it: `git tag -a vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`.
   A change to the public API bumps the minor version (`v0.1.0` → `v0.2.0`);
   anything else bumps the patch (`v0.1.0` → `v0.1.1`).
3. Projects that use PyLLM (like [PyStack](https://github.com/robcox2910/pystack))
   upgrade by editing the tag in their `[tool.uv.sources]` and running
   `uv lock --upgrade-package pyllm`, in a pull request so CI checks the pair.

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

<!-- series:start -->
## Related Projects

This project is one of a series that builds real systems "from scratch in Python" to teach how they work. See the [series home](https://robcox2910.github.io/) for the full story.

| Project | What it teaches | Repository | Docs |
| --- | --- | --- | --- |
| PyOS | Operating systems | [py-os](https://github.com/robcox2910/py-os) | [Docs](https://robcox2910.github.io/py-os/) |
| Pebble | Compilers and programming languages | [pebble-lang](https://github.com/robcox2910/pebble-lang) | [Docs](https://robcox2910.github.io/pebble-lang/) |
| PyDB | Relational databases | [pydb](https://github.com/robcox2910/pydb) | [Docs](https://robcox2910.github.io/pydb/) |
| PyKV | Key-value stores | [pykv](https://github.com/robcox2910/pykv) | [Docs](https://robcox2910.github.io/pykv/) |
| PyDocDB | Document databases | [pydoc-db](https://github.com/robcox2910/pydoc-db) | [Docs](https://robcox2910.github.io/pydoc-db/) |
| PyGraphDB | Graph databases | [pygraphdb](https://github.com/robcox2910/pygraphdb) | [Docs](https://robcox2910.github.io/pygraphdb/) |
| PyTSDB | Time-series databases | [pytsdb](https://github.com/robcox2910/pytsdb) | [Docs](https://robcox2910.github.io/pytsdb/) |
| PyVecDB | Vector databases | [pyvecdb](https://github.com/robcox2910/pyvecdb) | [Docs](https://robcox2910.github.io/pyvecdb/) |
| PyColDB | Column-family databases | [pycoldb](https://github.com/robcox2910/pycoldb) | [Docs](https://robcox2910.github.io/pycoldb/) |
| PyCrypt | Cryptography | [pycrypt](https://github.com/robcox2910/pycrypt) | [Docs](https://robcox2910.github.io/pycrypt/) |
| PyNet | Computer networking | [pynet](https://github.com/robcox2910/pynet) | [Docs](https://robcox2910.github.io/pynet/) |
| PyWeb | HTTP web servers | [pyweb](https://github.com/robcox2910/pyweb) | [Docs](https://robcox2910.github.io/pyweb/) |
| PyGit | Version control | [pygit](https://github.com/robcox2910/pygit) | [Docs](https://robcox2910.github.io/pygit/) |
| PySearch | Search engines | [pysearch](https://github.com/robcox2910/pysearch) | [Docs](https://robcox2910.github.io/pysearch/) |
| PyMQ | Message queues | [pymq](https://github.com/robcox2910/pymq) | [Docs](https://robcox2910.github.io/pymq/) |
| **PyLLM** | **Language models (the brain)** | [pyllm](https://github.com/robcox2910/pyllm) | [Docs](https://robcox2910.github.io/pyllm/) |
| PyScheduler | Job scheduling | [pyscheduler](https://github.com/robcox2910/pyscheduler) | [Docs](https://robcox2910.github.io/pyscheduler/) |
| PyStack | Full-stack integration | [pystack](https://github.com/robcox2910/pystack) | [Docs](https://robcox2910.github.io/pystack/) |
<!-- series:end -->

<!-- docs:start -->
## Documentation

Full docs (with kid-friendly explanations of every idea): https://robcox2910.github.io/pyllm/
<!-- docs:end -->
