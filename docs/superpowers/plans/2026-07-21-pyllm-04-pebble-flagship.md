# PyLLM Plan 4 — Pebble Flagship (grow your own corpus) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach the centrepiece lesson — *when there's no data, you grow it and you measure it*. Build a self-contained Pebble **program generator** (random valid ASTs → source), a **corpus harvester** (extract real Pebble from `pebble-lang`'s docs), and a **validity scorer** (run output back through Pebble's parser and report the % that parses). Bundle a synthetic Pebble corpus + a tiny trained checkpoint, and add `pyllm pebble` and `pyllm gen-corpus` CLI commands.

**Architecture:** The generator is **ours and stands alone** — its own tiny AST (frozen dataclasses) + a `render()` emitter — so generation and training never import `pebble-lang` (spec §6 dependency boundary). `pebble-lang` is used only as an **oracle**, imported lazily: its `Parser` grades validity and its `Formatter` canonicalizes. The shipped package degrades gracefully — `pyllm pebble` loads the bundled checkpoint and generates with no `pebble` import; live scoring imports `pebble` only when asked and only when installed. `pebble-lang` is an **optional, build-time** dependency (uv path source).

**Tech Stack:** Python 3.14, numpy, pytest, ruff, pyright, managed with `uv`. Builds on Plan 3 (`main`). `pebble-lang` (imports as `pebble`) as an optional dev/build dependency.

## Global Constraints

- **RULE #1 — Child-friendly docs about EVERYTHING.** Every public class/function gets an analogy-first docstring. This plan is not done until its concept doc (`docs/concepts/grow-your-own-data.md`) exists. This rule outranks all others.
- **Python 3.14**, managed with `uv`. Run everything via `uv run`.
- **Dependencies:** `numpy` only for the shipped runtime. `pebble-lang` is optional/build-time; never imported at module top-level in shipped code — always lazily, inside the function that needs it.
- **TDD:** write the failing test first, watch it fail, then implement.
- **ruff** + **pyright** must stay clean after every task.
- **No `TYPE_CHECKING`, no `from __future__ import annotations`.** PEP 695 `type X = ...` aliases are allowed (Python 3.14).
- **Determinism:** the generator takes an `rng` (`numpy.random.Generator`); tests pass a seeded `np.random.default_rng(0)`.
- **Graceful degradation:** tests that need `pebble` must `pytest.importorskip("pebble")` so the suite passes even where `pebble-lang` is not installed.
- Frequent commits: one per task minimum.

## Plan 3 interfaces this plan consumes

- `from pyllm.models import GPT`, `from pyllm.training import train`, `from pyllm.tokenizer import CharTokenizer`, `from pyllm.checkpoint import save, load`, `from pyllm.generate import generate`.
- `pyllm.data.CORPUS_DIR` (a `Path`), `pyllm.data.load_corpus(name)`.
- CLI lives in `src/pyllm/cli.py` (`main(argv=None)` with `argparse` subparsers; default command generates).

## pebble-lang APIs this plan consumes (verified)

- `from pebble.lexer import Lexer` — `Lexer(source).tokenize() -> list[Token]`.
- `from pebble.parser import Parser` — `Parser(tokens).parse() -> Program`.
- `from pebble.errors import PebbleError` — base class of `LexerError` + `ParseError`; carries `.message`, `.line`, `.column`.
- `from pebble.formatter import Formatter` — `Formatter(source).format() -> str` (lexes+parses internally; source→formatted-source).
- Pebble syntax (narrow subset we generate): `let x = <expr>`, `x = <expr>` (reassignment), `print(<expr>)`, `return [<expr>]`, `if <cond> { … } [else { … }]`, `while <cond> { … }`, `for i in range(<expr>) { … }`, `fn name(a, b) { … }`. Brace blocks, newline-delimited statements, no semicolons. Booleans `true`/`false`. Operators are plain strings: arithmetic `+ - * / %`, comparison `== != < <= > >=`, logical `and or`, unary `-`/`not`.
- **There is no bare-expression statement** in Pebble — a call is only valid as an *expression* (inside `let`/`print`/`return`/an argument), never alone on a line. The generator never emits a bare call statement.

## File structure

```
src/pyllm/
  pebble/
    __init__.py       re-exports; PEBBLE_AVAILABLE flag
    ast.py            our own tiny AST (frozen dataclasses) for a Pebble subset
    render.py         render(program) -> Pebble source string
    generator.py      random_program(rng, ...) -> Program
    harvest.py        harvest_text(md) / harvest_dir(path) -> list[str] of ```pebble blocks
    score.py          is_valid(source) / parse_rate(sources)  (lazy pebble import)
    corpus.py         build_corpus(rng, ...) -> str; canonicalize(source) (lazy)
  data/
    pebble_corpus.txt bundled synthetic+harvested corpus (committed, Task 9)
    pebble.npz        tiny trained Pebble checkpoint (committed, Task 10)
scripts/
  build_pebble_corpus.py   build-time: writes src/pyllm/data/pebble_corpus.txt
  train_pebble.py          build-time: writes src/pyllm/data/pebble.npz
tests/
  pebble/test_render.py test_generator.py test_harvest.py test_score.py test_corpus.py
  test_cli_pebble.py
  test_docs.py             MODIFY: guard docs/concepts/grow-your-own-data.md
docs/concepts/grow-your-own-data.md
pyproject.toml             MODIFY: optional [pebble] dep + uv source; ship pebble data files
src/pyllm/data/__init__.py MODIFY: add PEBBLE_CHECKPOINT path
src/pyllm/cli.py           MODIFY: add `pebble` + `gen-corpus` subcommands
```

**Our AST (subset) — the vocabulary of nodes the generator/emitter share:**
- Expressions: `Num(value:int)`, `Bool(value:bool)`, `Str(value:str)`, `Var(name:str)`, `Bin(op:str, left, right)`, `Unary(op:str, operand)`, `Call(name:str, args:list)`.
- Statements: `Let(name:str, value)`, `Assign(name:str, value)`, `Print(value)`, `Return(value)` (value may be `None`), `If(cond, body:list, orelse)` (orelse may be `None`), `While(cond, body:list)`, `For(var:str, count, body:list)`, `Func(name:str, params:list[str], body:list)`.
- Root: `Program(statements:list)`.

---

### Task 1: Optional pebble-lang dependency wiring

**Files:**
- Modify: `pyproject.toml`
- Test: `tests/pebble/test_pebble_available.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an optional `pebble` extra that installs the sibling `pebble-lang` project via a uv path source, so `import pebble` works in the dev/build environment. No shipped runtime dependency changes.

- [ ] **Step 1: Write the failing test**

Create `tests/pebble/test_pebble_available.py`:

```python
def test_pebble_is_importable_in_dev_env():
    # In the dev/build environment (with the optional `pebble` extra installed),
    # the pebble-lang oracle must be importable. Skipped elsewhere.
    import importlib.util

    if importlib.util.find_spec("pebble") is None:
        import pytest
        pytest.skip("pebble-lang not installed (optional build-time dependency)")
    from pebble.parser import Parser  # noqa: F401
    from pebble.lexer import Lexer  # noqa: F401
    from pebble.formatter import Formatter  # noqa: F401
```

- [ ] **Step 2: Run test to verify it skips (pebble not yet installed)**

Run: `uv run pytest tests/pebble/test_pebble_available.py -v`
Expected: SKIP ("pebble-lang not installed").

- [ ] **Step 3: Wire the optional dependency**

In `pyproject.toml`, add a `pebble` entry to `[project.optional-dependencies]`:

```toml
pebble = ["pebble-lang"]
```

Add a uv source pointing at the sibling project (new top-level table, e.g. after `[project.optional-dependencies]`):

```toml
[tool.uv.sources]
pebble-lang = { path = "../pebble-lang", editable = true }
```

Then install it into the dev environment:

```bash
uv sync --all-extras
```

Expected: resolves and installs `pebble-lang` (editable) from `../pebble-lang`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pebble/test_pebble_available.py -v`
Expected: PASS (pebble now importable).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "build: add optional pebble-lang dependency (uv path source)"
```

---

### Task 2: Pebble subset AST + renderer

**Files:**
- Create: `src/pyllm/pebble/__init__.py`
- Create: `src/pyllm/pebble/ast.py`
- Create: `src/pyllm/pebble/render.py`
- Test: `tests/pebble/test_render.py`

**Interfaces:**
- Consumes: nothing (pure Python, no pebble import).
- Produces:
  - `ast.py`: the frozen-dataclass node set listed above, plus PEP 695 aliases `type Expr = ...` and `type Stmt = ...`.
  - `render.py`: `render(program) -> str` turning a `Program` into Pebble source (trailing newline). Every binary/unary expression is wrapped in parens so precedence can never make it invalid; the Pebble formatter strips redundant parens later.

- [ ] **Step 1: Write the failing test**

Create `tests/pebble/test_render.py`:

```python
from pyllm.pebble.ast import Assign
from pyllm.pebble.ast import Bin
from pyllm.pebble.ast import Bool
from pyllm.pebble.ast import Call
from pyllm.pebble.ast import For
from pyllm.pebble.ast import Func
from pyllm.pebble.ast import If
from pyllm.pebble.ast import Let
from pyllm.pebble.ast import Num
from pyllm.pebble.ast import Print
from pyllm.pebble.ast import Program
from pyllm.pebble.ast import Return
from pyllm.pebble.ast import Unary
from pyllm.pebble.ast import Var
from pyllm.pebble.ast import While
from pyllm.pebble.render import render


def test_render_let_and_binop():
    prog = Program([Let("x", Bin("+", Num(1), Num(2)))])
    assert render(prog) == "let x = (1 + 2)\n"


def test_render_reassignment_and_print():
    prog = Program([Assign("x", Num(5)), Print(Var("x"))])
    assert render(prog) == "x = 5\nprint(x)\n"


def test_render_bool_and_unary():
    prog = Program([Let("b", Unary("not", Bool(True)))])
    assert render(prog) == "let b = (not true)\n"


def test_render_if_else_block_is_brace_delimited():
    prog = Program([
        If(Bin(">", Var("x"), Num(0)),
           [Print(Var("x"))],
           [Print(Num(0))]),
    ])
    assert render(prog) == (
        "if (x > 0) {\n"
        "    print(x)\n"
        "} else {\n"
        "    print(0)\n"
        "}\n"
    )


def test_render_while_and_for_and_call():
    prog = Program([
        While(Bool(True), [Assign("x", Call("step", [Var("x")]))]),
        For("i", Num(3), [Print(Var("i"))]),
    ])
    assert render(prog) == (
        "while true {\n"
        "    x = step(x)\n"
        "}\n"
        "for i in range(3) {\n"
        "    print(i)\n"
        "}\n"
    )


def test_render_function_def_and_return():
    prog = Program([Func("add", ["a", "b"], [Return(Bin("+", Var("a"), Var("b")))])])
    assert render(prog) == (
        "fn add(a, b) {\n"
        "    return (a + b)\n"
        "}\n"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pebble/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.pebble'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/pebble/__init__.py`:

```python
"""The Pebble flagship: generate Pebble code, harvest real code, grade validity."""

import importlib.util

# True when the pebble-lang oracle is importable (optional build-time dep).
PEBBLE_AVAILABLE = importlib.util.find_spec("pebble") is not None
```

Create `src/pyllm/pebble/ast.py`:

```python
"""A tiny AST for a slice of the Pebble language -- the shapes we can generate.

An AST ("abstract syntax tree") is a program drawn as a family tree: a `+` node
with two children, an `if` node with a condition and a body, and so on. We build
these trees at random (always following the rules) and then `render` turns a tree
back into text you could paste into Pebble. This is our own little copy of
Pebble's grammar -- just the parts we teach the model to write.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Num:
    """A whole number, like 42."""
    value: int


@dataclass(frozen=True)
class Bool:
    """A yes/no value: true or false."""
    value: bool


@dataclass(frozen=True)
class Str:
    """A piece of text in quotes."""
    value: str


@dataclass(frozen=True)
class Var:
    """A name that stands for a value, like `count`."""
    name: str


@dataclass(frozen=True)
class Bin:
    """Two values joined by an operator, like `a + b` or `x > 0`."""
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Unary:
    """One value with an operator in front, like `-x` or `not done`."""
    op: str
    operand: "Expr"


@dataclass(frozen=True)
class Call:
    """Calling a function by name with some arguments, like `step(x)`."""
    name: str
    args: list


type Expr = Num | Bool | Str | Var | Bin | Unary | Call


@dataclass(frozen=True)
class Let:
    """Make a new named box and put a value in it: `let x = 1`."""
    name: str
    value: Expr


@dataclass(frozen=True)
class Assign:
    """Put a new value in an existing box: `x = 2`."""
    name: str
    value: Expr


@dataclass(frozen=True)
class Print:
    """Show a value on the screen: `print(x)`."""
    value: Expr


@dataclass(frozen=True)
class Return:
    """Hand a value back out of a function: `return x` (or bare `return`)."""
    value: Expr | None


@dataclass(frozen=True)
class If:
    """Do one thing if a test is true, optionally another thing otherwise."""
    cond: Expr
    body: list
    orelse: list | None


@dataclass(frozen=True)
class While:
    """Keep doing something while a test stays true."""
    cond: Expr
    body: list


@dataclass(frozen=True)
class For:
    """Repeat once for each number in a range: `for i in range(3) { ... }`."""
    var: str
    count: Expr
    body: list


@dataclass(frozen=True)
class Func:
    """Define a reusable function: `fn add(a, b) { ... }`."""
    name: str
    params: list
    body: list


type Stmt = Let | Assign | Print | Return | If | While | For | Func


@dataclass(frozen=True)
class Program:
    """A whole program: a list of statements, top to bottom."""
    statements: list
```

Create `src/pyllm/pebble/render.py`:

```python
"""Turn a Pebble AST back into Pebble source text (the reverse of parsing).

Walking the family tree and writing out the text at each node. We wrap every
maths/logic expression in parentheses so the meaning can never get scrambled by
operator precedence -- Pebble's own formatter tidies away the extra brackets
afterwards.
"""

from pyllm.pebble.ast import Assign
from pyllm.pebble.ast import Bin
from pyllm.pebble.ast import Bool
from pyllm.pebble.ast import Call
from pyllm.pebble.ast import For
from pyllm.pebble.ast import Func
from pyllm.pebble.ast import If
from pyllm.pebble.ast import Let
from pyllm.pebble.ast import Num
from pyllm.pebble.ast import Print
from pyllm.pebble.ast import Program
from pyllm.pebble.ast import Return
from pyllm.pebble.ast import Str
from pyllm.pebble.ast import Unary
from pyllm.pebble.ast import Var
from pyllm.pebble.ast import While

_INDENT = "    "


def render(program):
    """Render a whole Program to Pebble source (ends with a newline)."""
    lines = []
    for stmt in program.statements:
        lines.extend(_render_stmt(stmt, 0))
    return "\n".join(lines) + "\n"


def _render_block(body, depth):
    lines = []
    for stmt in body:
        lines.extend(_render_stmt(stmt, depth))
    return lines


def _render_stmt(stmt, depth):
    pad = _INDENT * depth
    match stmt:
        case Let(name, value):
            return [f"{pad}let {name} = {_render_expr(value)}"]
        case Assign(name, value):
            return [f"{pad}{name} = {_render_expr(value)}"]
        case Print(value):
            return [f"{pad}print({_render_expr(value)})"]
        case Return(None):
            return [f"{pad}return"]
        case Return(value):
            return [f"{pad}return {_render_expr(value)}"]
        case If(cond, body, orelse):
            lines = [f"{pad}if {_render_expr(cond)} {{"]
            lines.extend(_render_block(body, depth + 1))
            if orelse:
                lines.append(f"{pad}}} else {{")
                lines.extend(_render_block(orelse, depth + 1))
            lines.append(f"{pad}}}")
            return lines
        case While(cond, body):
            lines = [f"{pad}while {_render_expr(cond)} {{"]
            lines.extend(_render_block(body, depth + 1))
            lines.append(f"{pad}}}")
            return lines
        case For(var, count, body):
            lines = [f"{pad}for {var} in range({_render_expr(count)}) {{"]
            lines.extend(_render_block(body, depth + 1))
            lines.append(f"{pad}}}")
            return lines
        case Func(name, params, body):
            lines = [f"{pad}fn {name}({', '.join(params)}) {{"]
            lines.extend(_render_block(body, depth + 1))
            lines.append(f"{pad}}}")
            return lines
    raise TypeError(f"cannot render statement: {stmt!r}")


def _render_expr(expr):
    match expr:
        case Num(value):
            return str(value)
        case Bool(value):
            return "true" if value else "false"
        case Str(value):
            return f'"{value}"'
        case Var(name):
            return name
        case Bin(op, left, right):
            return f"({_render_expr(left)} {op} {_render_expr(right)})"
        case Unary("not", operand):
            return f"(not {_render_expr(operand)})"
        case Unary(op, operand):
            return f"({op}{_render_expr(operand)})"
        case Call(name, args):
            rendered = ", ".join(_render_expr(a) for a in args)
            return f"{name}({rendered})"
    raise TypeError(f"cannot render expression: {expr!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pebble/test_render.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Pebble subset AST + renderer"
```

---

### Task 3: Random program generator

**Files:**
- Create: `src/pyllm/pebble/generator.py`
- Test: `tests/pebble/test_generator.py`

**Interfaces:**
- Consumes: `ast` nodes, numpy `Generator`.
- Produces: `random_program(rng, num_statements=8, max_depth=2) -> Program`. Builds a random program from the subset grammar: it starts by declaring a couple of variables (so references have something to point at), optionally defines a function, then emits statements (let / reassignment / print / if / while / for). Control-flow bodies recurse with a shrinking `max_depth` and always contain at least one statement (Pebble blocks must be non-empty in our generator). Deterministic for a given `rng`.

- [ ] **Step 1: Write the failing test**

Create `tests/pebble/test_generator.py`:

```python
import numpy as np

from pyllm.pebble.ast import Program
from pyllm.pebble.generator import random_program
from pyllm.pebble.render import render


def test_generator_returns_a_program():
    prog = random_program(np.random.default_rng(0))
    assert isinstance(prog, Program)
    assert len(prog.statements) > 0


def test_generator_is_deterministic_with_seed():
    a = render(random_program(np.random.default_rng(7)))
    b = render(random_program(np.random.default_rng(7)))
    assert a == b


def test_generator_varies_with_seed():
    a = render(random_program(np.random.default_rng(1)))
    b = render(random_program(np.random.default_rng(2)))
    assert a != b


def test_generated_source_is_nonempty_text():
    src = render(random_program(np.random.default_rng(3), num_statements=6))
    assert isinstance(src, str)
    assert "let " in src  # always declares at least one variable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pebble/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.pebble.generator'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/pebble/generator.py`:

```python
"""Grow random-but-valid Pebble programs, following the grammar rules.

Like a robot that writes little programs by rolling dice at every choice: "should
this be a loop or a print? add or multiply? which variable?" Because it only ever
picks moves the grammar allows, every program it writes is syntactically valid --
which is exactly how we manufacture unlimited, always-correct training data when
there's no real Pebble code to be found.
"""

from pyllm.pebble.ast import Assign
from pyllm.pebble.ast import Bin
from pyllm.pebble.ast import Bool
from pyllm.pebble.ast import Call
from pyllm.pebble.ast import For
from pyllm.pebble.ast import Func
from pyllm.pebble.ast import If
from pyllm.pebble.ast import Let
from pyllm.pebble.ast import Num
from pyllm.pebble.ast import Print
from pyllm.pebble.ast import Program
from pyllm.pebble.ast import Return
from pyllm.pebble.ast import Unary
from pyllm.pebble.ast import Var

_VARS = ["x", "y", "z", "a", "b", "c", "n", "total", "count", "result", "value"]
_FUNCS = ["compute", "helper", "calc", "combine", "step", "make"]
_BINOPS = ["+", "-", "*", "%", "and", "or"]
_CMPOPS = ["==", "!=", "<", "<=", ">", ">="]


def _choice(rng, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _rand_value_expr(rng, names, depth):
    """A value-producing expression (number, variable, or arithmetic)."""
    if depth <= 0 or rng.random() < 0.4:
        if names and rng.random() < 0.5:
            return Var(_choice(rng, names))
        return Num(int(rng.integers(0, 100)))
    left = _rand_value_expr(rng, names, depth - 1)
    right = _rand_value_expr(rng, names, depth - 1)
    return Bin(_choice(rng, ["+", "-", "*", "%"]), left, right)


def _rand_condition(rng, names, depth):
    """A truthy expression for `if`/`while` -- a comparison or a boolean combo."""
    if depth > 0 and rng.random() < 0.3:
        op = _choice(rng, ["and", "or"])
        return Bin(op, _rand_condition(rng, names, depth - 1),
                   _rand_condition(rng, names, depth - 1))
    if rng.random() < 0.15:
        return Unary("not", _rand_condition(rng, names, 0))
    left = _rand_value_expr(rng, names, depth)
    right = _rand_value_expr(rng, names, depth)
    return Bin(_choice(rng, _CMPOPS), left, right)


def _rand_call(rng, names):
    n_args = int(rng.integers(0, 3))
    args = [_rand_value_expr(rng, names, 1) for _ in range(n_args)]
    return Call(_choice(rng, _FUNCS), args)


def _rand_simple_stmt(rng, names):
    """A leaf statement that never nests: let / reassignment / print."""
    roll = rng.random()
    if roll < 0.4 or not names:
        name = _choice(rng, _VARS)
        value = _rand_call(rng, names) if rng.random() < 0.2 \
            else _rand_value_expr(rng, names, 2)
        names.append(name)
        return Let(name, value)
    if roll < 0.7:
        return Assign(_choice(rng, names), _rand_value_expr(rng, names, 2))
    return Print(_choice(rng, [Var(_choice(rng, names)), _rand_call(rng, names)]))


def _rand_block(rng, names, depth, size):
    """A non-empty block of statements (bodies must not be empty)."""
    body = [_rand_stmt(rng, names, depth) for _ in range(max(1, size))]
    return body


def _rand_stmt(rng, names, depth):
    """Any statement; control flow only appears while we still have depth budget."""
    if depth <= 0 or rng.random() < 0.55:
        return _rand_simple_stmt(rng, names)
    roll = rng.random()
    if roll < 0.4:
        orelse = _rand_block(rng, names, depth - 1, 1) if rng.random() < 0.5 else None
        return If(_rand_condition(rng, names, 1),
                  _rand_block(rng, names, depth - 1, 2), orelse)
    if roll < 0.7:
        return While(_rand_condition(rng, names, 1),
                     _rand_block(rng, names, depth - 1, 2))
    var = _choice(rng, ["i", "j", "k"])
    return For(var, Num(int(rng.integers(1, 6))),
               _rand_block(rng, names + [var], depth - 1, 2))


def random_program(rng, num_statements=8, max_depth=2):
    """Build one random valid Pebble program of roughly `num_statements` steps."""
    names = []
    statements = []
    # Seed a couple of variables so later statements have something to reference.
    for _ in range(2):
        statements.append(_rand_simple_stmt(rng, names))
    # Optionally define a small function up top.
    if rng.random() < 0.4:
        params = [_choice(rng, ["a", "b", "n"]) for _ in range(int(rng.integers(1, 3)))]
        body = _rand_block(rng, list(params), 1, 2)
        body.append(Return(_rand_value_expr(rng, list(params), 1)))
        statements.append(Func(_choice(rng, _FUNCS), params, body))
    while len(statements) < num_statements:
        statements.append(_rand_stmt(rng, names, max_depth))
    return Program(statements)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pebble/test_generator.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add random Pebble program generator"
```

---

### Task 4: Validity scorer (the oracle)

**Files:**
- Create: `src/pyllm/pebble/score.py`
- Test: `tests/pebble/test_score.py`

**Interfaces:**
- Consumes: `pebble.lexer.Lexer`, `pebble.parser.Parser`, `pebble.errors.PebbleError` — imported **lazily** inside the functions.
- Produces:
  - `is_valid(source) -> bool` — True iff `source` lexes+parses under Pebble. Raises `RuntimeError` with an install hint if `pebble` is not importable.
  - `parse_rate(sources) -> float` — fraction of an iterable of source strings that are valid (0.0 for an empty iterable).
- **The decisive spec test lives here:** a batch of generator output must be 100% valid.

- [ ] **Step 1: Write the failing test**

Create `tests/pebble/test_score.py`:

```python
import numpy as np
import pytest

pytest.importorskip("pebble")  # oracle-backed tests need pebble-lang installed

from pyllm.pebble.generator import random_program  # noqa: E402
from pyllm.pebble.render import render  # noqa: E402
from pyllm.pebble.score import is_valid  # noqa: E402
from pyllm.pebble.score import parse_rate  # noqa: E402


def test_is_valid_accepts_known_good():
    assert is_valid("let x = 1 + 2\nprint(x)\n")


def test_is_valid_rejects_known_bad():
    assert not is_valid("let = = = 1 +\n")


def test_parse_rate_of_empty_is_zero():
    assert parse_rate([]) == 0.0


def test_parse_rate_counts_fraction_valid():
    rate = parse_rate(["let x = 1\n", "@@@ not pebble @@@"])
    assert rate == 0.5


def test_every_generated_program_parses():
    rng = np.random.default_rng(0)
    sources = [render(random_program(rng)) for _ in range(200)]
    assert parse_rate(sources) == 1.0, "generator must emit 100% valid Pebble"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pebble/test_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.pebble.score'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/pebble/score.py`:

```python
"""Grade Pebble text by asking the real Pebble parser: "does this parse?"

Because we own the Pebble toolchain, we have a perfect marker: feed a program to
Pebble's own lexer and parser and see if it complains. Almost no AI project can
objectively score its own output; we can. `is_valid` marks one program;
`parse_rate` marks a pile of them and reports the percentage that pass.
"""


def _require_pebble():
    """Import the Pebble oracle lazily, with a friendly error if it's missing."""
    try:
        from pebble.errors import PebbleError
        from pebble.lexer import Lexer
        from pebble.parser import Parser
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Live Pebble scoring needs pebble-lang. Install the optional extra: "
            "`uv sync --all-extras` (uses ../pebble-lang)."
        ) from exc
    return Lexer, Parser, PebbleError


def is_valid(source):
    """True if `source` is syntactically valid Pebble (it lexes and parses)."""
    Lexer, Parser, PebbleError = _require_pebble()
    try:
        Parser(Lexer(source).tokenize()).parse()
    except PebbleError:
        return False
    return True


def parse_rate(sources):
    """The fraction of the given programs that parse (0.0 if there are none)."""
    sources = list(sources)
    if not sources:
        return 0.0
    return sum(is_valid(s) for s in sources) / len(sources)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pebble/test_score.py -v`
Expected: PASS (5 passed). If `test_every_generated_program_parses` fails, the generator/renderer emitted invalid Pebble — fix Task 2/3 (do not weaken the assertion); print a failing sample with `render(random_program(np.random.default_rng(0)))` and parse it directly to see the `PebbleError`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Pebble validity scorer (parser oracle)"
```

---

### Task 5: Corpus harvester

**Files:**
- Create: `src/pyllm/pebble/harvest.py`
- Test: `tests/pebble/test_harvest.py`

**Interfaces:**
- Consumes: `pathlib`, `re` (no pebble import).
- Produces:
  - `harvest_text(markdown) -> list[str]` — every fenced ` ```pebble ` code block in a markdown string (block contents, trailing newline stripped).
  - `harvest_dir(path) -> list[str]` — all such blocks across every `*.md` file under `path` (recursively), in sorted path order.

- [ ] **Step 1: Write the failing test**

Create `tests/pebble/test_harvest.py`:

```python
from pyllm.pebble.harvest import harvest_dir
from pyllm.pebble.harvest import harvest_text

_MD = """# Title

Some prose.

```pebble
let x = 1
print(x)
```

More prose.

```python
print("not pebble")
```

```pebble
fn f() { return 1 }
```
"""


def test_harvest_text_extracts_only_pebble_blocks():
    blocks = harvest_text(_MD)
    assert len(blocks) == 2
    assert blocks[0] == "let x = 1\nprint(x)"
    assert blocks[1] == "fn f() { return 1 }"


def test_harvest_text_no_blocks_returns_empty():
    assert harvest_text("just prose, no code") == []


def test_harvest_dir_reads_markdown_files(tmp_path):
    (tmp_path / "a.md").write_text("```pebble\nlet a = 1\n```\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("```pebble\nlet b = 2\n```\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("```pebble\nlet c = 3\n```\n", encoding="utf-8")
    blocks = harvest_dir(tmp_path)
    assert blocks == ["let a = 1", "let b = 2"]  # only .md, sorted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pebble/test_harvest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.pebble.harvest'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/pebble/harvest.py`:

```python
"""Dig real, hand-written Pebble out of the documentation.

There's no big pile of Pebble code on the internet, but there are hundreds of
little examples tucked inside Pebble's own docs. This scoops out every fenced
```pebble code block so we can mix genuine, idiomatic snippets into our
home-grown corpus. (Some doc snippets are deliberately broken to show errors, so
callers filter these through the validity scorer before keeping them.)
"""

import re
from pathlib import Path

_FENCE = re.compile(r"```pebble\n(.*?)```", re.DOTALL)


def harvest_text(markdown):
    """Return every fenced ```pebble block found in a markdown string."""
    return [match.group(1).rstrip("\n") for match in _FENCE.finditer(markdown)]


def harvest_dir(path):
    """Return every ```pebble block across all *.md files under `path` (sorted)."""
    blocks = []
    for md in sorted(Path(path).rglob("*.md")):
        blocks.extend(harvest_text(md.read_text(encoding="utf-8")))
    return blocks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pebble/test_harvest.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Pebble corpus harvester"
```

---

### Task 6: Corpus builder + canonicalizer + package API

**Files:**
- Create: `src/pyllm/pebble/corpus.py`
- Modify: `src/pyllm/pebble/__init__.py`
- Test: `tests/pebble/test_corpus.py`

**Interfaces:**
- Consumes: `generator.random_program`, `render`, `harvest_dir`, `score.is_valid` (lazy), `pebble.formatter.Formatter` (lazy).
- Produces:
  - `canonicalize(source) -> str` — run source through Pebble's `Formatter` (lazy import; `RuntimeError` if pebble missing).
  - `build_corpus(rng, num_generated=400, harvest_paths=(), canonical=False) -> str` — generate `num_generated` random programs; harvest `*.md` blocks from each path in `harvest_paths` and keep only those that pass `is_valid` (harvesting is skipped entirely if pebble is unavailable); optionally `canonicalize` every program; join all programs with a blank line between them (trailing newline).
  - `src/pyllm/pebble/__init__.py` re-exports `random_program`, `render`, `build_corpus`, `is_valid`, `parse_rate`, `harvest_dir`, `PEBBLE_AVAILABLE`.

- [ ] **Step 1: Write the failing test**

Create `tests/pebble/test_corpus.py`:

```python
import numpy as np

from pyllm.pebble.corpus import build_corpus


def test_build_corpus_generates_programs_without_pebble():
    # Generation must NOT require pebble; harvest_paths empty => no pebble needed.
    text = build_corpus(np.random.default_rng(0), num_generated=5)
    assert isinstance(text, str)
    assert text.count("let ") >= 5  # every program declares variables
    programs = [p for p in text.split("\n\n") if p.strip()]
    assert len(programs) == 5


def test_build_corpus_is_deterministic():
    a = build_corpus(np.random.default_rng(1), num_generated=4)
    b = build_corpus(np.random.default_rng(1), num_generated=4)
    assert a == b


def test_public_api_exports():
    import pyllm.pebble as pebble

    for name in ["random_program", "render", "build_corpus", "is_valid",
                 "parse_rate", "harvest_dir", "PEBBLE_AVAILABLE"]:
        assert hasattr(pebble, name), f"pyllm.pebble is missing {name}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pebble/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyllm.pebble.corpus'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/pyllm/pebble/corpus.py`:

```python
"""Assemble a training corpus: mostly home-grown programs, seasoned with real ones.

This is the "grow your own data" recipe in code: generate a big batch of
guaranteed-valid programs, optionally fold in genuine snippets harvested from the
docs (keeping only the ones that actually parse), and glue them into one text
file the model can learn from.
"""

from pyllm.pebble import PEBBLE_AVAILABLE
from pyllm.pebble.generator import random_program
from pyllm.pebble.harvest import harvest_dir
from pyllm.pebble.render import render


def canonicalize(source):
    """Tidy Pebble source into its one true formatting, via Pebble's formatter."""
    try:
        from pebble.formatter import Formatter
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Canonicalizing needs pebble-lang. Install it: `uv sync --all-extras`."
        ) from exc
    return Formatter(source).format()


def build_corpus(rng, num_generated=400, harvest_paths=(), canonical=False):
    """Build a Pebble training corpus as one big string (programs blank-line-separated).

    - `num_generated`: how many random valid programs to grow.
    - `harvest_paths`: directories of markdown to mine for real Pebble snippets
      (kept only if they parse; skipped entirely when pebble-lang is absent).
    - `canonical`: if True, run every program through Pebble's formatter.
    """
    programs = [render(random_program(rng)) for _ in range(num_generated)]
    if harvest_paths and PEBBLE_AVAILABLE:
        from pyllm.pebble.score import is_valid
        for path in harvest_paths:
            programs.extend(block for block in harvest_dir(path) if is_valid(block))
    if canonical:
        programs = [canonicalize(p) for p in programs]
    return "\n\n".join(p.strip() for p in programs) + "\n"
```

Update `src/pyllm/pebble/__init__.py` (append the re-exports below the `PEBBLE_AVAILABLE` definition):

```python
from pyllm.pebble.corpus import build_corpus  # noqa: E402
from pyllm.pebble.generator import random_program  # noqa: E402
from pyllm.pebble.harvest import harvest_dir  # noqa: E402
from pyllm.pebble.render import render  # noqa: E402
from pyllm.pebble.score import is_valid  # noqa: E402
from pyllm.pebble.score import parse_rate  # noqa: E402

__all__ = [
    "PEBBLE_AVAILABLE",
    "random_program",
    "render",
    "build_corpus",
    "is_valid",
    "parse_rate",
    "harvest_dir",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pebble/test_corpus.py -v`
Expected: PASS (3 passed). Run the whole pebble suite too: `uv run pytest tests/pebble/ -v`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add Pebble corpus builder + canonicalizer + package API"
```

---

### Task 7: Bundle the Pebble corpus

**Files:**
- Create: `scripts/build_pebble_corpus.py`
- Modify: `src/pyllm/data/__init__.py`
- Create (committed artifact): `src/pyllm/data/pebble_corpus.txt`
- Test: `tests/pebble/test_bundled_corpus.py`

**Interfaces:**
- Consumes: `build_corpus`, `parse_rate`, `pyllm.data.CORPUS_DIR`.
- Produces:
  - `scripts/build_pebble_corpus.py` — build-time script: generate ~600 programs + harvest `../pebble-lang/docs`, assert 100% parse via `parse_rate`, write `src/pyllm/data/pebble_corpus.txt`.
  - `pyllm.data.load_corpus("pebble")` returns the bundled Pebble corpus (already supported by the Plan 3 loader — filename is `pebble_corpus.txt`).

- [ ] **Step 1: Write the failing test**

Create `tests/pebble/test_bundled_corpus.py`:

```python
from pyllm.data import load_corpus


def test_bundled_pebble_corpus_loads():
    text = load_corpus("pebble")
    assert isinstance(text, str)
    assert len(text) > 2000
    assert "let " in text and "fn " in text


def test_bundled_pebble_corpus_parses_when_pebble_available():
    import pytest
    pytest.importorskip("pebble")
    from pyllm.pebble import parse_rate

    programs = [p for p in load_corpus("pebble").split("\n\n") if p.strip()]
    assert parse_rate(programs) > 0.98  # harvested docs may contain a rare edge case
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pebble/test_bundled_corpus.py -v`
Expected: FAIL — `pebble_corpus.txt` does not exist.

- [ ] **Step 3: Write the builder and build the artifact**

Create `scripts/build_pebble_corpus.py`:

```python
"""Build-time: generate + harvest the bundled Pebble corpus.

Run once with `uv run python scripts/build_pebble_corpus.py`; the resulting
`src/pyllm/data/pebble_corpus.txt` is committed.
"""

from pathlib import Path

import numpy as np

from pyllm.data import CORPUS_DIR
from pyllm.pebble import build_corpus
from pyllm.pebble import parse_rate

_PEBBLE_DOCS = Path(__file__).resolve().parents[2] / "pebble-lang" / "docs"


def main():
    rng = np.random.default_rng(2024)
    harvest = [_PEBBLE_DOCS] if _PEBBLE_DOCS.exists() else []
    text = build_corpus(rng, num_generated=600, harvest_paths=harvest,
                        canonical=True)
    programs = [p for p in text.split("\n\n") if p.strip()]
    rate = parse_rate(programs)
    print(f"{len(programs)} programs, parse rate {rate:.3f}")
    out = CORPUS_DIR / "pebble_corpus.txt"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
```

Add a `PEBBLE_CHECKPOINT` path to `src/pyllm/data/__init__.py` (after `POKEMON_CHECKPOINT`):

```python
PEBBLE_CHECKPOINT = CORPUS_DIR / "pebble.npz"
```

Then build the corpus:

```bash
uv run python scripts/build_pebble_corpus.py
```

Expected: prints a parse rate of `1.000` (or ≥0.98 once harvested docs are mixed in) and writes the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pebble/test_bundled_corpus.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit (including the corpus artifact)**

```bash
git add -A
git commit -m "feat: build and bundle synthetic+harvested Pebble corpus"
```

---

### Task 8: Train + bundle the tiny Pebble checkpoint

**Files:**
- Create: `scripts/train_pebble.py`
- Create (committed artifact): `src/pyllm/data/pebble.npz`
- Test: `tests/pebble/test_bundled_checkpoint.py`

**Interfaces:**
- Consumes: `load_corpus`, `CharTokenizer`, `GPT`, `train`, `save`, `PEBBLE_CHECKPOINT`, `generate`.
- Produces:
  - `scripts/train_pebble.py` — build-time: train a tiny GPT on `pebble_corpus.txt`, write `src/pyllm/data/pebble.npz`; defines `build_and_train(rng) -> (model, tokenizer)`.
  - the committed `pebble.npz`.

- [ ] **Step 1: Write the failing test**

Create `tests/pebble/test_bundled_checkpoint.py`:

```python
import numpy as np

from pyllm.checkpoint import load
from pyllm.data import PEBBLE_CHECKPOINT
from pyllm.generate import generate


def test_bundled_pebble_checkpoint_exists_and_generates():
    assert PEBBLE_CHECKPOINT.exists(), "run: uv run python scripts/train_pebble.py"
    model, tok = load(PEBBLE_CHECKPOINT)
    out = generate(model, tok, prompt="let ", max_new_tokens=60,
                   rng=np.random.default_rng(0))
    assert isinstance(out, str) and out.startswith("let ")
    assert set(out).issubset(set(tok.stoi))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/pebble/test_bundled_checkpoint.py -v`
Expected: FAIL — `pebble.npz` does not exist.

- [ ] **Step 3: Write the training script and build the checkpoint**

Create `scripts/train_pebble.py`:

```python
"""Build-time: train a tiny GPT on the Pebble corpus and save the checkpoint.

Run once with `uv run python scripts/train_pebble.py`; the resulting
`src/pyllm/data/pebble.npz` is committed so `pyllm pebble` runs instantly.
"""

import numpy as np

from pyllm.checkpoint import save
from pyllm.data import PEBBLE_CHECKPOINT
from pyllm.data import load_corpus
from pyllm.generate import generate
from pyllm.models import GPT
from pyllm.tokenizer import CharTokenizer
from pyllm.training import train


def build_and_train(rng):
    """Train a small Pebble GPT and return (model, tokenizer)."""
    text = load_corpus("pebble")
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text))
    model = GPT(vocab_size=tokenizer.vocab_size, block_size=32, embed_dim=64,
                num_heads=4, num_layers=3, rng=rng)
    train(model, data, steps=3000, batch_size=32, lr=3e-3, rng=rng,
          log_every=250)
    return model, tokenizer


def main():
    rng = np.random.default_rng(7)
    model, tokenizer = build_and_train(rng)
    save(PEBBLE_CHECKPOINT, model, tokenizer)
    print(f"saved checkpoint to {PEBBLE_CHECKPOINT}")
    sample = generate(model, tokenizer, prompt="let ", max_new_tokens=200,
                      temperature=0.7, rng=np.random.default_rng(0))
    print("--- sample program ---")
    print(sample)


if __name__ == "__main__":
    main()
```

Then build it (tune `block_size`/`embed_dim`/`steps` so it finishes in a few minutes on CPU and the sample looks Pebble-shaped — braces, `let`, `fn`, `print`):

```bash
uv run python scripts/train_pebble.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/pebble/test_bundled_checkpoint.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit (including the .npz artifact)**

```bash
git add -A
git commit -m "feat: train and bundle tiny Pebble GPT checkpoint"
```

---

### Task 9: CLI — `pyllm pebble` + `pyllm gen-corpus`

**Files:**
- Modify: `src/pyllm/cli.py`
- Modify: `pyproject.toml` (ship the new data files)
- Test: `tests/test_cli_pebble.py`

**Interfaces:**
- Consumes: `load` (checkpoint), `PEBBLE_CHECKPOINT`, `generate`, `build_corpus`, `parse_rate`, `PEBBLE_AVAILABLE`.
- Produces two new subcommands on `main`:
  - `run_pebble(args, out=print)` — load the bundled Pebble checkpoint (or `args.checkpoint`), generate `args.max_new_tokens` from `args.prompt` (default `"let "`), print it; if `args.score` and `pebble` is available, split the output on blank lines and print the parse rate.
  - `run_gen_corpus(args, out=print)` — `build_corpus` with `args.num_generated` (+ optionally harvest `args.harvest`), write to `args.out`, print program count.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_pebble.py`:

```python
from pyllm.cli import main


def test_pebble_command_generates(capsys):
    code = main(["pebble", "--max-new-tokens", "40", "--seed", "0"])
    out = capsys.readouterr().out
    assert code == 0
    assert len(out.strip()) > 0


def test_pebble_command_score_reports_rate_when_available(capsys):
    import pytest
    pytest.importorskip("pebble")
    code = main(["pebble", "--max-new-tokens", "80", "--seed", "1", "--score"])
    out = capsys.readouterr().out
    assert code == 0
    assert "parse rate" in out.lower()


def test_gen_corpus_writes_a_file(tmp_path):
    dest = tmp_path / "corpus.txt"
    code = main(["gen-corpus", "--num-generated", "10", "--out", str(dest),
                 "--seed", "0"])
    assert code == 0
    assert dest.exists()
    assert dest.read_text(encoding="utf-8").count("let ") >= 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_pebble.py -v`
Expected: FAIL with `SystemExit`/argparse error (`invalid choice: 'pebble'`).

- [ ] **Step 3: Write minimal implementation**

Add to `src/pyllm/cli.py` — new imports at the top (merge with existing import block):

```python
from pyllm.data import PEBBLE_CHECKPOINT
from pyllm.pebble import PEBBLE_AVAILABLE
from pyllm.pebble import build_corpus
from pyllm.pebble import parse_rate
```

Add two handler functions (near the other `run_*` handlers):

```python
def run_pebble(args, out=print):
    """Flagship demo: write Pebble code, and (optionally) grade what parses."""
    rng = np.random.default_rng(args.seed)
    checkpoint = args.checkpoint if args.checkpoint else PEBBLE_CHECKPOINT
    model, tokenizer = load(checkpoint)
    text = generate(model, tokenizer, prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_k=args.top_k, rng=rng)
    out(text)
    if args.score:
        if not PEBBLE_AVAILABLE:
            out("(install pebble-lang to score validity: uv sync --all-extras)")
        else:
            programs = [p for p in text.split("\n\n") if p.strip()]
            out(f"parse rate: {parse_rate(programs):.1%} of {len(programs)} programs")
    return 0


def run_gen_corpus(args, out=print):
    """Build-time helper: grow a synthetic Pebble corpus to a file."""
    from pathlib import Path

    rng = np.random.default_rng(args.seed)
    harvest = [Path(args.harvest)] if args.harvest else []
    text = build_corpus(rng, num_generated=args.num_generated,
                        harvest_paths=harvest, canonical=args.canonical)
    Path(args.out).write_text(text, encoding="utf-8")
    programs = [p for p in text.split("\n\n") if p.strip()]
    out(f"wrote {len(programs)} programs to {args.out}")
    return 0
```

Register the subparsers inside `main`, before `args = parser.parse_args(argv)`:

```python
    p_pebble = sub.add_parser("pebble", help="write Pebble code (the flagship demo)")
    p_pebble.add_argument("--prompt", default="let ")
    p_pebble.add_argument("--max-new-tokens", type=int, default=200)
    p_pebble.add_argument("--temperature", type=float, default=0.7)
    p_pebble.add_argument("--top-k", type=int, default=None)
    p_pebble.add_argument("--checkpoint", default=None)
    p_pebble.add_argument("--seed", type=int, default=None)
    p_pebble.add_argument("--score", action="store_true",
                          help="report what %% of the output parses")
    p_pebble.set_defaults(func=run_pebble)

    p_corpus = sub.add_parser("gen-corpus", help="grow a synthetic Pebble corpus")
    p_corpus.add_argument("--num-generated", type=int, default=400)
    p_corpus.add_argument("--harvest", default=None,
                          help="directory of *.md docs to mine for real Pebble")
    p_corpus.add_argument("--canonical", action="store_true",
                          help="run each program through the Pebble formatter")
    p_corpus.add_argument("--out", default="pebble_corpus.txt")
    p_corpus.add_argument("--seed", type=int, default=None)
    p_corpus.set_defaults(func=run_gen_corpus)
```

In `pyproject.toml`, extend the wheel force-include with the Pebble data files:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/pyllm/data/pokemon_corpus.txt" = "pyllm/data/pokemon_corpus.txt"
"src/pyllm/data/pokemon.npz" = "pyllm/data/pokemon.npz"
"src/pyllm/data/pebble_corpus.txt" = "pyllm/data/pebble_corpus.txt"
"src/pyllm/data/pebble.npz" = "pyllm/data/pebble.npz"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_pebble.py -v`
Expected: PASS (3 passed). Try it for real:

```bash
uv run pyllm pebble --max-new-tokens 200 --seed 0 --score
uv run pyllm gen-corpus --num-generated 20 --out /tmp/pebble.txt
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: add pyllm pebble + gen-corpus CLI commands"
```

---

### Task 10: Concept doc (RULE #1) + doc guard

**Files:**
- Create: `docs/concepts/grow-your-own-data.md`
- Modify: `tests/test_docs.py`

**Interfaces:**
- Consumes: nothing (prose + a test guard).
- Produces: the kid-friendly concept doc for Plan 4 and a test asserting it exists and covers its key ideas. **This plan is NOT done until this task passes** (RULE #1).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_docs.py`:

```python
def test_plan4_concept_doc_exists_and_covers_key_ideas():
    from pathlib import Path

    doc = Path("docs/concepts/grow-your-own-data.md")
    assert doc.exists(), "RULE #1: missing concept doc grow-your-own-data.md"
    text = doc.read_text().lower()
    for idea in ["harvest", "generate", "grade", "parser", "analogy"]:
        assert idea in text, f"grow-your-own-data.md should explain '{idea}'"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docs.py::test_plan4_concept_doc_exists_and_covers_key_ideas -v`
Expected: FAIL (doc doesn't exist).

- [ ] **Step 3: Write the concept doc**

Create `docs/concepts/grow-your-own-data.md`:

```markdown
# When there's no data, grow your own

Every AI learns from examples. But what if the thing you want to teach has almost
no examples in the world? Pebble is a brand-new language — there's no giant pile
of Pebble code to learn from. So we do something clever: we **grow our own data**
and we **measure it**. Here's the analogy: instead of foraging for wild berries
that might be poisonous, we plant our own garden where we control every seed — and
we still taste-test everything before we serve it.

Three steps: **harvest**, **generate**, **grade**.

## 1. Harvest — gather the real examples that do exist

Pebble's documentation is full of small, hand-written code snippets. We scoop
every one of them out (the `harvest` step) — genuine, idiomatic Pebble written by
a human. There aren't many, but they're gold: they show how real Pebble *feels*.

## 2. Generate — grow unlimited fresh examples

This is the key trick. We wrote a little robot (`generator.py`) that builds Pebble
programs by rolling dice at every choice — "loop or print? add or multiply? which
variable?" — but it only ever makes moves the grammar allows. So **every program
it writes is valid by construction**. Where there was almost no data, we now have
as much as we want, and it's always correct.

## 3. Grade — measure quality with a perfect ruler

Because we *own* the Pebble toolchain, we have something almost no AI project has:
a perfect answer key. We feed a program to Pebble's own **parser** — the same one
the real language uses — and ask "does this parse?" Count the percentage that pass
and you have an honest, objective quality score (the `parse_rate`). No guessing,
no vibes: the parser is the judge.

## A tiny worked example

Ask the trained model to continue `let `, and it dreams up something like:

```pebble
let total = (3 + 4)
fn step(n) {
    return (n * 2)
}
print(step(total))
```

Run that back through Pebble's parser: it parses. ✓ Now do it for a thousand
programs and report the percentage — that's the model's grade.

## Why does this matter?

"Grow your own data and measure it" is one of the most useful ideas in all of AI.
When you can *generate* examples and *check* them automatically, you can teach a
model almost anything — even a language that didn't exist last week.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_docs.py -v`
Expected: PASS (all doc guards green).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: add grow-your-own-data concept doc"
```

---

## Final verification (run before declaring Plan 4 done)

- [ ] `uv run pytest -q` — the entire suite is green (pebble-backed tests run; they skip only where pebble-lang is absent).
- [ ] `uv run ruff check .` — clean (scope to Plan 4 files if pre-existing files are noisy).
- [ ] `uv run ruff format --check <plan-4 files>` — clean.
- [ ] `uv run pyright` — clean.
- [ ] `uv run pyllm pebble --seed 0 --score` — prints Pebble-shaped code and a parse rate.
- [ ] `uv run pyllm gen-corpus --num-generated 20 --out /tmp/c.txt` — writes a corpus.
- [ ] Update `README.md` with a Pebble flagship example session.
- [ ] Confirm the roadmap's Plan 4 row is satisfied (generator, harvester, scorer, bundled corpus + checkpoint, `pyllm pebble` + `gen-corpus`). Grammar-constrained decoding remains a deferred stretch goal (roadmap) — NOT in this plan.
```
