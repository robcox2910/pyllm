# PyLLM Plan 6 — Grammar-Constrained Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a from-scratch Pebble "referee" so constrained generation only ever returns programs that parse (100% parse rate), plus the corpus fix and kid-friendly docs.

**Architecture:** `pyllm/pebble/referee.py` is a character-level recursive-descent parser over the canonical Pebble subset that answers `viable(text)` / `complete(text)`. `sample_next` gains an `allowed` callback (rejection sampling); `generate` gains a `referee` and trims output to the last complete point. CLI/API expose a `constrained` switch.

**Tech Stack:** Python 3.14, numpy, pytest, ruff, pyright, uv. pebble-lang (optional) is the independent judge in tests.

**Spec:** `docs/superpowers/specs/2026-10-16-pyllm-06-grammar-constrained-decoding-design.md`

## Global Constraints

- **RULE #1:** every public function/class gets a docstring a 12-year-old could follow (analogy first); the plan is not done until `docs/concepts/the-referee.md` exists.
- **Python 3.14**, run everything via `uv run`.
- **Dependencies:** numpy only at runtime. pebble-lang stays optional (`pytest.importorskip("pebble")` in tests that need it).
- **TDD:** failing test first for every behaviour.
- **ruff** (`uv run ruff check . && uv run ruff format --check .`) and **pyright** (`uv run pyright`) clean; coverage ≥ 90% (enforced by pytest addopts).
- **No `TYPE_CHECKING`, no `from __future__ import annotations`.**
- One commit per task minimum; end commit messages with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Unconstrained generation must stay byte-for-byte identical for a given seed (don't change RNG consumption when `allowed is None`).

## Review Focus

1. A `--prompt` that already breaks the rules (e.g. `"}"`) with `--constrained` → friendly message and exit code 2, not a traceback. (Task 3)
2. `--top-k 1` with `--constrained` where the single shortlisted token is illegal → falls back to the full vocabulary, never crashes. (Task 2)
3. `--temperature 0` with `--constrained` → picks the best *legal* token every step and still returns complete programs. (Task 2)
4. Budget runs out mid-program → output trimmed to the last complete program, never half a program. (Task 2)
5. A name that merely starts like a keyword (`n`, `no`, `whi`, `letter`) → treated as a name, not a foul, and a program ending in it counts as complete. (Task 1)

---

### Task 1: The referee

**Files:**
- Create: `src/pyllm/pebble/referee.py`
- Modify: `src/pyllm/pebble/__init__.py`
- Test: `tests/pebble/test_referee.py`, `tests/pebble/test_pebble_corpus.py` (exports)

**Interfaces:**
- Produces: `pyllm.pebble.referee.viable(text: str) -> bool`, `pyllm.pebble.referee.complete(text: str) -> bool`; both re-exported from `pyllm.pebble`. The module itself is passed as a referee object later (anything with `viable`/`complete`).

- [ ] **Step 1: Write the failing tests** — `tests/pebble/test_referee.py`:

```python
import numpy as np
import pytest

from pyllm.pebble import random_program, render
from pyllm.pebble.referee import complete, viable

FINISHED = [
    "let a = 1",
    "let a = 1\n",
    "let a = n",
    "let letter = no",
    "# just a comment",
    "fn add(a, b) {\n    return a + b\n}",
    'if not a == 1 and b {\n    print(-a)\n} else if a > 2 {\n    a = 0\n'
    '} else {\n    x = f(1, "hi")\n}',
    "for i in range(3) {\n    while i < 2 {\n        i = i + 1\n    }\n}",
    "let a = 1\n\nlet b = 2",
]

UNFINISHED = [
    "",
    "le",
    "whi",
    "let a = ",
    "let a = 1 an",
    "let a = not",
    'print("hi',
    "fn f(a, ",
    "if a {\n    ",
    "if a {\n    b = 1\n  ",
    "let a = 1\n\n",
]

FOULS = {
    "return outside a function": "return 1",
    "chained comparison": "let a = 1 < 2 < 3",
    "empty block": "if x {\n}",
    "keyword as a name": "let let = 1",
    "wrong indent": "if x {\n  let a = 1\n}",
    "indented top level": "let a = 1\n    let b = 2",
    "string across a newline": 'let a = "x\n"',
    "string interpolation": 'let a = "{x}"',
    "closing brace at top level": "}",
    "function inside a block": "if a {\n    fn f() {\n        let a = 1\n    }\n}",
    "two blank lines": "let a = 1\n\n\nlet b = 2",
    "blank line mid-program": "let a = (1\n\nlet b = 1",
    "trailing spaces": "print(x)  ",
    "spaced minus": "let a = - 1",
    "bare expression": "x == 1",
}


@pytest.mark.parametrize("text", FINISHED)
def test_finished_programs_are_complete_and_viable(text):
    assert complete(text)
    assert viable(text)


@pytest.mark.parametrize("text", UNFINISHED)
def test_unfinished_text_is_viable_but_not_complete(text):
    assert viable(text)
    assert not complete(text)


@pytest.mark.parametrize("text", list(FOULS.values()), ids=list(FOULS))
def test_rule_breaking_text_gets_the_whistle(text):
    assert not viable(text)
    assert not complete(text)


def test_every_generated_program_is_complete():
    for seed in range(300):
        source = render(random_program(np.random.default_rng(seed)))
        assert complete(source), source


def test_every_prefix_of_a_generated_program_is_viable():
    for seed in range(20):
        source = render(random_program(np.random.default_rng(seed)))
        for end in range(len(source) + 1):
            assert viable(source[:end]), repr(source[:end])


def test_formatter_style_programs_are_complete():
    pytest.importorskip("pebble")
    from pyllm.pebble.corpus import canonicalize

    for seed in range(100):
        source = canonicalize(render(random_program(np.random.default_rng(seed))))
        assert complete(source), source


def test_everything_the_referee_accepts_really_parses():
    pytest.importorskip("pebble")
    from pyllm.pebble import is_valid

    for text in FINISHED:
        for program in text.split("\n\n"):
            assert is_valid(program), program
```

Add `"viable"` and `"complete"` to the name list in `tests/pebble/test_pebble_corpus.py::test_public_api_exports`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pebble/test_referee.py tests/pebble/test_pebble_corpus.py -q --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyllm.pebble.referee'`.

- [ ] **Step 3: Implement** — create `src/pyllm/pebble/referee.py` (this exact code was prototyped and verified: all 600 generator programs, raw and formatter-canonical, are `complete`; every prefix is `viable`; 0 false accepts against pebble-lang on the bundled corpus):

```python
"""The referee: a rule-checker that watches Pebble being written, one letter at a time.

Imagine a football referee who knows every rule of the game. They don't tell the
players what to do -- they only blow the whistle when a move is *illegal*. This
module is that referee for Pebble code. Give it some half-written text and it
answers two questions:

- `viable(text)`: "could this still grow into a valid program?" (no whistle yet)
- `complete(text)`: "is this a finished, valid program right now?" (full time)

It only knows the canonical slice of Pebble our corpus is written in (4-space
indents, single spaces around operators), which is a strict subset of real
Pebble -- so anything the referee lets through, Pebble's own parser accepts too.

Programs are separated by a blank line, just like in the training corpus. Every
earlier program must be complete; only the last one may still be in progress.
"""

from functools import lru_cache

_INDENT = "    "
_KEYWORDS = frozenset(
    "let fn if else while for in return print and or not true false range".split()
)
_NAME_START = "abcdefghijklmnopqrstuvwxyz_"
_NAME_CHARS = _NAME_START + "0123456789"
_DIGITS = "0123456789"
_STRING_BANNED = '"\\\n{}'
_COMPARE_OPS = (" == ", " != ", " < ", " <= ", " > ", " >= ")


class _NeedMore(Exception):
    """The text stopped mid-move: not wrong yet, just unfinished."""


class _Foul(Exception):
    """The text broke a rule: no amount of extra typing can fix it."""


class _Parser:
    """A recursive-descent parser that reads characters left to right.

    Each `_parse_*` method reads one grammar shape. Running out of text raises
    `_NeedMore`; reading something the rules forbid raises `_Foul`.
    """

    def __init__(self, text):
        self.text = text
        self.pos = 0

    # -- reading helpers -------------------------------------------------------

    def _at_end(self):
        return self.pos >= len(self.text)

    def _peek(self):
        if self._at_end():
            raise _NeedMore
        return self.text[self.pos]

    def _expect(self, literal):
        """Read exactly `literal`, letter by letter."""
        for char in literal:
            if self._peek() != char:
                raise _Foul
            self.pos += 1

    def _choose(self, options):
        """Read whichever option comes next; None if none of them do.

        If the text ends part-way through an option ("a" of " and "), we can't
        tell yet, so we ask for more.
        """
        rest = self.text[self.pos :]
        if not rest:
            return None  # the text simply ends here; the caller decides if that's ok
        for option in options:
            if rest.startswith(option):
                self.pos += len(option)
                return option
        if any(option.startswith(rest) for option in options):
            raise _NeedMore
        return None

    def _word(self):
        """Read a run of name letters. Returns (word, touched_the_end)."""
        start = self.pos
        while not self._at_end() and self.text[self.pos] in _NAME_CHARS:
            self.pos += 1
        return self.text[start : self.pos], self._at_end()

    def _name(self):
        """Read a new name (for let/for/fn/params): not a keyword, not empty."""
        if self._peek() not in _NAME_START:
            raise _Foul
        word, at_end = self._word()
        if at_end:
            raise _NeedMore  # "le" might grow into "level"
        if word in _KEYWORDS:
            raise _Foul
        return word

    # -- programs and blocks ---------------------------------------------------

    def parse_program(self):
        """A program: statements on their own lines, nothing indented."""
        self._parse_statement(depth=0, in_fn=False)
        while not self._at_end():
            self._expect("\n")
            if self._at_end():
                return  # one trailing newline is fine
            self._parse_statement(depth=0, in_fn=False)

    def _parse_block(self, depth, in_fn):
        """`{` already read: one or more indented statements, then `}`."""
        inner = _INDENT * (depth + 1)
        self._expect("\n" + inner)
        self._parse_statement(depth + 1, in_fn)
        while True:
            self._expect("\n")
            self._peek()  # the block can't simply stop: it needs a `}`
            step = self._choose([inner, _INDENT * depth + "}"])
            if step is None:
                raise _Foul
            if step != inner:
                return
            self._parse_statement(depth + 1, in_fn)

    # -- statements ------------------------------------------------------------

    def _parse_statement(self, depth, in_fn):
        char = self._peek()
        if char == "#":
            self._parse_comment()
            return
        if char not in _NAME_START:
            raise _Foul
        word, at_end = self._word()
        if at_end:
            raise _NeedMore  # a half-typed keyword or name
        match word:
            case "let":
                self._expect(" ")
                self._name()
                self._expect(" = ")
                self._parse_expr()
            case "print":
                self._expect("(")
                self._parse_expr()
                self._expect(")")
            case "return":
                if not in_fn:
                    raise _Foul  # you can only hand back a value from a function
                if self._choose([" "]):
                    self._parse_expr()
            case "if":
                self._expect(" ")
                self._parse_expr()
                self._expect(" {")
                self._parse_block(depth, in_fn)
                while branch := self._choose([" else {", " else if "]):
                    if branch == " else {":
                        self._parse_block(depth, in_fn)
                        break
                    self._parse_expr()  # `else if`: another test, another block
                    self._expect(" {")
                    self._parse_block(depth, in_fn)
            case "while":
                self._expect(" ")
                self._parse_expr()
                self._expect(" {")
                self._parse_block(depth, in_fn)
            case "for":
                self._expect(" ")
                self._name()
                self._expect(" in range(")
                self._parse_expr()
                self._expect(") {")
                self._parse_block(depth, in_fn)
            case "fn":
                if depth > 0:
                    raise _Foul  # functions live at the top of the program
                self._expect(" ")
                self._name()
                self._expect("(")
                if self._choose([")"]) is None:
                    self._name()
                    while self._choose([", "]):
                        self._name()
                    self._expect(")")
                self._expect(" {")
                self._parse_block(depth, in_fn=True)
            case _ if word in _KEYWORDS:
                raise _Foul
            case _:
                self._expect(" = ")
                self._parse_expr()

    def _parse_comment(self):
        """`#` and then anything up to the end of the line."""
        while not self._at_end() and self.text[self.pos] != "\n":
            self.pos += 1

    # -- expressions, loosest-binding first ------------------------------------

    def _parse_expr(self):
        self._parse_and()
        while self._choose([" or "]):
            self._parse_and()

    def _parse_and(self):
        self._parse_not()
        while self._choose([" and "]):
            self._parse_not()

    def _parse_not(self):
        # Only a full "not " counts: "no" might just be the start of a name.
        if self.text.startswith("not ", self.pos):
            self.pos += len("not ")
            self._parse_not()
        else:
            self._parse_compare()

    def _parse_compare(self):
        self._parse_sum()
        if self._choose(_COMPARE_OPS):
            self._parse_sum()  # just one comparison: no `a < b < c`

    def _parse_sum(self):
        self._parse_term()
        while self._choose([" + ", " - "]):
            self._parse_term()

    def _parse_term(self):
        self._parse_unary()
        while self._choose([" * ", " / ", " % "]):
            self._parse_unary()

    def _parse_unary(self):
        self._choose(["-"])  # an optional minus sign in front
        self._parse_primary()

    def _parse_primary(self):
        char = self._peek()
        if char in _DIGITS:
            while not self._at_end() and self.text[self.pos] in _DIGITS:
                self.pos += 1
        elif char == '"':
            self.pos += 1
            while self._peek() != '"':
                if self._peek() in _STRING_BANNED:
                    raise _Foul
                self.pos += 1
            self.pos += 1
        elif char == "(":
            self.pos += 1
            self._parse_expr()
            self._expect(")")
        elif char in _NAME_START:
            word, at_end = self._word()
            if word in ("true", "false"):
                return
            if word in _KEYWORDS:
                if at_end:
                    raise _NeedMore  # "not" might grow into "nothing"
                raise _Foul
            if self._choose(["("]):
                if self._choose([")"]) is None:
                    self._parse_expr()
                    while self._choose([", "]):
                        self._parse_expr()
                    self._expect(")")
        else:
            raise _Foul


def _judge(program):
    """'complete', 'unfinished' or 'foul' for a single program's text."""
    try:
        _Parser(program).parse_program()
    except _NeedMore:
        return "unfinished"
    except _Foul:
        return "foul"
    return "complete"


@lru_cache(maxsize=4096)
def _program_complete(program):
    return _judge(program) == "complete"


def viable(text):
    """True if `text` could still grow into valid Pebble (no whistle yet)."""
    *finished, current = text.split("\n\n")
    if not all(_program_complete(p) for p in finished):
        return False
    return _judge(current) != "foul"


def complete(text):
    """True if `text` is finished, valid Pebble right now (full time)."""
    return all(_program_complete(p) for p in text.split("\n\n"))
```

In `src/pyllm/pebble/__init__.py` add after the `harvest_dir` import:

```python
from pyllm.pebble.referee import complete, viable  # noqa: E402
```

and append `"viable", "complete"` to `__all__`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/pebble -q --no-cov`
Expected: all PASS.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format . && uv run pyright
git add src/pyllm/pebble tests/pebble
git commit -m "feat: add Pebble referee (viable/complete prefix checker)"
```

---

### Task 2: Constrained sampling

**Files:**
- Modify: `src/pyllm/generate.py`
- Test: `tests/test_generate.py`

**Interfaces:**
- Consumes: any referee object with `viable(text) -> bool` and `complete(text) -> bool` (Task 1's module fits).
- Produces: `sample_next(logits_row, temperature=1.0, top_k=None, rng=None, allowed=None) -> int` where `allowed: Callable[[int], bool] | None`; `generate(model, tokenizer, prompt="", max_new_tokens=100, temperature=1.0, top_k=None, rng=None, referee=None) -> str`. With a referee: raises `ValueError` for a non-viable prompt; returns text trimmed to the last complete point (or the prompt if none).

- [ ] **Step 1: Write the failing tests** — append to `tests/test_generate.py`:

```python
import pytest


def test_forbidden_tokens_are_never_chosen():
    logits = np.array([0.0, 9.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    picks = {
        sample_next(logits, rng=rng, allowed=lambda token: token != 1)
        for _ in range(200)
    }
    assert 1 not in picks


def test_greedy_picks_the_best_allowed_token():
    logits = np.array([0.1, 5.0, -2.0, 3.0])
    pick = sample_next(logits, temperature=0.0, allowed=lambda token: token != 1)
    assert pick == 3


def test_top_k_falls_back_when_the_shortlist_is_all_forbidden():
    logits = np.array([0.1, 5.0, -2.0, 3.0])
    rng = np.random.default_rng(0)
    pick = sample_next(
        logits, top_k=1, rng=rng, allowed=lambda token: token != 1
    )
    assert pick != 1


def test_nothing_allowed_is_an_error():
    with pytest.raises(RuntimeError):
        sample_next(np.array([1.0, 2.0]), allowed=lambda token: False)


class OnlyAs:
    """A toy referee: text may only contain 'a', and is finished at even lengths."""

    def viable(self, text):
        return set(text) <= {"a"}

    def complete(self, text):
        return self.viable(text) and len(text) % 2 == 0


def _tiny_gpt(text="abc"):
    tokenizer = CharTokenizer(text)
    model = GPT(
        tokenizer.vocab_size,
        block_size=8,
        embed_dim=16,
        num_heads=2,
        num_layers=1,
        rng=np.random.default_rng(0),
    )
    return model, tokenizer


def test_referee_keeps_generation_legal_and_trims_to_a_finished_point():
    model, tokenizer = _tiny_gpt()
    text = generate(
        model,
        tokenizer,
        max_new_tokens=7,
        rng=np.random.default_rng(0),
        referee=OnlyAs(),
    )
    assert text == "aaaaaa"  # 7 legal letters, trimmed back to the last even length


def test_greedy_generation_with_a_referee_stays_legal():
    model, tokenizer = _tiny_gpt()
    text = generate(
        model, tokenizer, max_new_tokens=5, temperature=0.0, referee=OnlyAs()
    )
    assert text == "aaaa"


def test_rule_breaking_prompt_is_rejected():
    model, tokenizer = _tiny_gpt()
    with pytest.raises(ValueError, match="breaks the rules"):
        generate(model, tokenizer, prompt="b", referee=OnlyAs())


def test_no_finished_point_returns_the_prompt():
    class NeverDone(OnlyAs):
        def complete(self, text):
            return False

    model, tokenizer = _tiny_gpt()
    text = generate(
        model, tokenizer, prompt="a", max_new_tokens=4, referee=NeverDone()
    )
    assert text == "a"
```

(`CharTokenizer` and `GPT` are already imported at the top of the file.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_generate.py -q --no-cov`
Expected: FAIL — `TypeError: sample_next() got an unexpected keyword argument 'allowed'`.

- [ ] **Step 3: Implement** — replace `sample_next` in `src/pyllm/generate.py` and add a helper above it:

```python
def _probabilities(logits, temperature, top_k):
    """Turn raw scores into dice weights (temperature + optional top-k shortlist)."""
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        keep = np.argsort(logits)[-top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def _anything_goes(token):
    return True


def sample_next(logits_row, temperature=1.0, top_k=None, rng=None, allowed=None):
    """Pick the next token from the model's scores -- boldly or cautiously.

    The model hands back a score for every possible next character. We turn those
    into a dice with weighted sides and roll it.
    - `temperature` is the boldness dial: 0 always takes the single best guess
      (safe but repetitive); 1 rolls fairly by the model's confidence; higher is
      wilder and more surprising.
    - `top_k` says "only consider the k most likely characters" so the roll never
      lands on something absurd. `None` or any value <= 0 means "no shortlist --
      every character stays in the running".
    - `allowed` is an optional referee: a function that says whether a token is a
      legal move. If the dice land on an illegal token we cross it out and roll
      again, so only legal tokens can ever come out. (If the top-k shortlist runs
      out of legal tokens, we widen the search to every token.)
    """
    if rng is None:
        rng = np.random.default_rng()
    if allowed is None:
        allowed = _anything_goes
    logits = np.asarray(logits_row, dtype=np.float64)
    if temperature != 0.0:
        shortlists = [top_k, None] if top_k is not None and top_k > 0 else [None]
        for shortlist in shortlists:
            probs = _probabilities(logits, temperature, shortlist)
            while probs.sum() > 0:
                pick = int(rng.choice(len(probs), p=probs))
                if allowed(pick):
                    return pick
                probs[pick] = 0.0  # cross out the illegal move...
                if probs.sum() > 0:
                    probs = probs / probs.sum()  # ...and re-balance the dice
    # Greedy (or every roll was crossed out): the best-scoring legal token.
    for token in np.argsort(-logits, kind="stable"):
        if allowed(int(token)):
            return int(token)
    raise RuntimeError("the referee forbade every possible next token")
```

Replace `generate` with:

```python
def generate(
    model,
    tokenizer,
    prompt="",
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    rng=None,
    referee=None,
):
    """Dream up new text one character at a time, feeding each guess back in.

    Start with the prompt, ask the model "what comes next?", roll for a character
    (`sample_next`), stick it on the end, and repeat. Each new character becomes
    part of the context for the next guess -- that feedback loop is why a model
    trained on Pokémon names can invent brand-new ones that *feel* like Pokémon.
    We only ever look at the last `block_size` characters (the model's memory).

    Pass a `referee` (anything with `viable(text)` and `complete(text)`, such as
    `pyllm.pebble.referee`) and every character must be a legal move. We remember
    the last moment the text was a finished program and, when we run out of
    characters, cut back to it -- so you never get half a program.
    """
    if rng is None:
        rng = np.random.default_rng()
    if referee is not None and not referee.viable(prompt):
        raise ValueError(
            f"the prompt {prompt!r} already breaks the rules, "
            "so it can never grow into valid code"
        )
    model.eval()
    context = list(tokenizer.encode(prompt))
    text = prompt
    finished_at = len(text) if referee is not None and referee.complete(text) else None
    for _ in range(max_new_tokens):
        window = context[-model.block_size :]
        if model.mode == "single":
            # MLP needs exactly block_size tokens; left-pad short windows.
            if len(window) < model.block_size:
                window = [0] * (model.block_size - len(window)) + window
        elif not window:
            # Sequence models with an empty prompt need a seed token to read.
            window = [0]
        x = np.array([window], dtype=np.int64)
        logits = model(x).data
        last_row = logits[0, -1] if logits.ndim == 3 else logits[0]
        allowed = None
        if referee is not None:

            def legal(token, so_far=text):
                return referee.viable(so_far + tokenizer.decode([token]))

            allowed = legal
        token = sample_next(
            last_row, temperature=temperature, top_k=top_k, rng=rng, allowed=allowed
        )
        context.append(token)
        text += tokenizer.decode([token])
        if referee is not None and referee.complete(text):
            finished_at = len(text)
    if referee is None:
        return text
    return text[:finished_at] if finished_at is not None else prompt
```

(`text` equals `tokenizer.decode(context)` because the tokenizer is lossless, so unconstrained output is unchanged.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: all PASS (including the existing seeded-output tests — unconstrained RNG use is unchanged).

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format . && uv run pyright
git add src/pyllm/generate.py tests/test_generate.py
git commit -m "feat: referee-constrained sampling (rejection sampling + trim to finished)"
```

---

### Task 3: `--constrained` CLI flag and `constrained=` API switch

**Files:**
- Modify: `src/pyllm/cli.py`, `src/pyllm/api.py`
- Test: `tests/test_cli_pebble.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `generate(..., referee=...)` (Task 2), `pyllm.pebble.referee` module (Task 1), `pyllm.pebble.parse_rate`.
- Produces: `pyllm pebble --constrained`; `generate_pebble(prompt="let ", max_new_tokens=120, temperature=0.7, seed=None, constrained: bool = False) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli_pebble.py`:

```python
def test_constrained_pebble_always_parses(capsys):
    import pytest

    pytest.importorskip("pebble")
    code = main(
        ["pebble", "--constrained", "--max-new-tokens", "150", "--seed", "0", "--score"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "parse rate: 100.0%" in out


def test_constrained_pebble_rejects_a_rule_breaking_prompt(capsys):
    code = main(["pebble", "--constrained", "--prompt", "}", "--seed", "0"])
    out = capsys.readouterr().out
    assert code == 2
    assert "referee" in out.lower()
```

Append to `tests/test_api.py`:

```python
def test_constrained_pebble_is_always_a_finished_program():
    from pyllm.api import generate_pebble
    from pyllm.pebble import complete

    for seed in range(3):
        text = generate_pebble(max_new_tokens=150, seed=seed, constrained=True)
        assert complete(text), text


def test_constrained_pebble_parses_under_the_real_parser():
    import pytest

    pytest.importorskip("pebble")
    from pyllm.api import generate_pebble
    from pyllm.pebble import parse_rate

    programs = []
    for seed in range(4):
        text = generate_pebble(max_new_tokens=150, seed=seed, constrained=True)
        programs += [p for p in text.split("\n\n") if p.strip()]
    assert programs
    assert parse_rate(programs) == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli_pebble.py tests/test_api.py -q --no-cov`
Expected: FAIL — `unrecognized arguments: --constrained` / `unexpected keyword argument 'constrained'`.

- [ ] **Step 3: Implement**

In `src/pyllm/cli.py` add the import `from pyllm.pebble import referee as pebble_referee` (next to the other `pyllm.pebble` import), then in `run_pebble` replace the `text = generate(...)` call with:

```python
    try:
        text = generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            rng=rng,
            referee=pebble_referee if args.constrained else None,
        )
    except ValueError as exc:
        out(f"(the referee blew the whistle: {exc})")
        return 2
```

and register the flag on `p_pebble` (after `--score`):

```python
    p_pebble.add_argument(
        "--constrained",
        action="store_true",
        help="let the grammar referee veto illegal characters (always valid code)",
    )
```

In `src/pyllm/api.py` add `from pyllm.pebble import referee as pebble_referee` and change `generate_pebble` to:

```python
def generate_pebble(
    prompt: str = "let ",
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    seed: int | None = None,
    constrained: bool = False,
) -> str:
    """Write Pebble-flavoured code using the bundled Pebble brain.

    With `constrained=True` a grammar referee vetoes every illegal character, so
    what comes back is always real, parseable Pebble.
    """
    model, tokenizer = _load_cached(PEBBLE_CHECKPOINT)
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        rng=np.random.default_rng(seed),
        referee=pebble_referee if constrained else None,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: all PASS. If a chosen seed happens to produce no finished program within 150 tokens, raise `max_new_tokens` to 200 in that test rather than changing seeds silently.

- [ ] **Step 5: Lint, type-check, commit**

```bash
uv run ruff check . && uv run ruff format . && uv run pyright
git add src/pyllm/cli.py src/pyllm/api.py tests/test_cli_pebble.py tests/test_api.py
git commit -m "feat: add pyllm pebble --constrained and generate_pebble(constrained=)"
```

---

### Task 4: Corpus fix — re-check validity after formatting

Root cause (verified): pebble-lang's formatter turns `struct Point { x, y }` into a multi-line form its own parser rejects, so harvested snippets validated *before* `canonicalize` can come out invalid. 15 of 994 bundled programs are affected.

**Files:**
- Modify: `src/pyllm/pebble/corpus.py`, `src/pyllm/data/pebble_corpus.txt`
- Test: `tests/pebble/test_pebble_corpus.py`

**Interfaces:**
- Consumes: `pyllm.pebble.score.is_valid`.
- Produces: `build_corpus` output in which every program parses (when pebble-lang is available).

- [ ] **Step 1: Write the failing tests** — append to `tests/pebble/test_pebble_corpus.py`:

```python
def test_canonical_corpus_drops_snippets_the_formatter_breaks(tmp_path):
    import numpy as np
    import pytest

    pytest.importorskip("pebble")
    from pyllm.pebble import build_corpus, is_valid

    (tmp_path / "doc.md").write_text("```pebble\nstruct Point { x, y }\n```\n")
    text = build_corpus(
        np.random.default_rng(0),
        num_generated=3,
        harvest_paths=[tmp_path],
        canonical=True,
    )
    programs = [p for p in text.split("\n\n") if p.strip()]
    assert "struct" not in text
    assert len(programs) == 3
    assert all(is_valid(p) for p in programs)


def test_every_bundled_corpus_program_parses():
    import pytest

    pytest.importorskip("pebble")
    from pyllm.data import load_corpus
    from pyllm.pebble import is_valid

    programs = [p for p in load_corpus("pebble").split("\n\n") if p.strip()]
    broken = [p for p in programs if not is_valid(p)]
    assert not broken, broken[0]
```

(Check `load_corpus("pebble")` exists in `src/pyllm/data/__init__.py`; if the bundled name differs, read the file via `PEBBLE_CORPUS`/path constant defined there.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/pebble/test_pebble_corpus.py -q --no-cov`
Expected: FAIL — `struct` found in text; 15 broken bundled programs.

- [ ] **Step 3: Implement** — in `src/pyllm/pebble/corpus.py` replace the last two lines of `build_corpus` (from `if canonical:` to the `return`) with:

```python
    if canonical:
        programs = [canonicalize(p) for p in programs]
    programs = [_one_program(p) for p in programs]
    if PEBBLE_AVAILABLE and (harvest_paths or canonical):
        from pyllm.pebble.score import is_valid

        # Tidying can break a snippet that parsed before (Pebble's formatter
        # mangles one-line structs), so judge the *final* text once more.
        programs = [p for p in programs if is_valid(p)]
    return "\n\n".join(programs) + "\n"
```

Then strip the broken programs from the bundled corpus (no retrain; checkpoint unchanged):

```bash
uv run python -c "
from pathlib import Path
from pyllm.pebble import is_valid
path = Path('src/pyllm/data/pebble_corpus.txt')
programs = [p for p in path.read_text(encoding='utf-8').split('\n\n') if p.strip()]
kept = [p.strip('\n') for p in programs if is_valid(p)]
path.write_text('\n\n'.join(kept) + '\n', encoding='utf-8')
print(len(programs), '->', len(kept))
"
```

Expected output: `994 -> 979`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q`
Expected: all PASS (bundled-checkpoint tests unaffected: the checkpoint stores its own tokenizer vocabulary).

- [ ] **Step 5: Commit**

```bash
git add src/pyllm/pebble/corpus.py src/pyllm/data/pebble_corpus.txt tests/pebble/test_pebble_corpus.py
git commit -m "fix: re-validate corpus programs after formatting; drop 15 broken ones"
```

---

### Task 5: Docs — the referee concept, README, roadmap, spec

**Files:**
- Create: `docs/concepts/the-referee.md`
- Modify: `docs/concepts/README.md`, `README.md`, `docs/superpowers/plans/2026-06-22-pyllm-roadmap.md`, `docs/superpowers/specs/2026-06-22-pyllm-design.md`
- Test: `tests/test_docs.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_docs.py`:

```python
def test_plan6_referee_concept_doc_exists_and_covers_key_ideas():
    doc = Path("docs/concepts/the-referee.md")
    assert doc.exists(), "RULE #1: missing concept doc docs/concepts/the-referee.md"
    text = doc.read_text().lower()
    for idea in ["referee", "whistle", "grammar", "analogy", "valid", "roll again"]:
        assert idea in text, f"the-referee.md should explain '{idea}'"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_docs.py -q --no-cov`
Expected: FAIL — missing concept doc.

- [ ] **Step 3: Write the docs**

Create `docs/concepts/the-referee.md`:

````markdown
# The referee: making the model play by the rules

Our Pebble brain is tiny. Left alone, it writes code that *looks* like Pebble but
often isn't quite right — a missing `}`, an `=` in a strange place — and Pebble's
parser rejects it. What if, instead of hoping the model gets it right, we simply
**never let it make an illegal move**?

## The analogy: a referee with a whistle

Picture a football match. The referee doesn't play, and doesn't tell anyone
where to kick. They just know every rule of the game, and the moment someone
tries an illegal move — *whistle!* — that move doesn't count, and the player has
to choose something else.

PyLLM's referee (`pyllm/pebble/referee.py`) knows the **grammar** of Pebble: the
rules for how code is allowed to be put together. Every time the model picks the
next character, the referee checks: "could the program still turn out valid if
we add this?"

- If yes, the character stays.
- If no — *whistle!* — we cross that character off the model's dice and
  **roll again**. Only legal characters can ever survive.

The referee answers two questions about any half-written text:

- **viable** — "could this still grow into a valid program?" (no whistle yet)
- **complete** — "is this a finished, valid program right now?" (full time)

## A tiny worked example

The model has written:

```
let total = 5 +
```

It badly wants the next character to be a newline. The referee checks
`let total = 5 +⏎` — a `+` with nothing after it can never be fixed by anything
that follows a line break. *Whistle!* The newline is crossed out, the dice are
re-balanced over the characters that are left, and the model rolls again — say,
a space, then `2`. Now `let total = 5 + 2` is **complete**, so the referee
remembers "the program was finished here".

When the model runs out of characters mid-way through the next program, we cut
the text back to the last moment it was complete. You never get half a program.

## How the referee reads code

It's a small *recursive-descent parser*: one little function per grammar shape
(`_parse_statement`, `_parse_expr`, `_parse_sum`...) that reads the text left to
right. Two things can go wrong as it reads:

- it **runs out of text** — that's fine, the program just isn't finished
  ("need more");
- it reads something the rules **forbid** — that's a foul, no amount of extra
  typing can fix it.

The referee only knows the tidy, canonical slice of Pebble our training corpus
uses (four-space indents, one space around operators). That's a strict subset of
real Pebble, so anything the referee lets through, Pebble's own parser accepts
too — and the tests check exactly that.

## Try it

```console
$ uv run pyllm pebble --seed 7 --score                # the model on its own
$ uv run pyllm pebble --seed 7 --constrained --score  # with the referee
parse rate: 100.0% of 2 programs
```

## Why does this matter?

The model didn't get any smarter — it's the same tiny brain. We just made it
*impossible* to break the rules. Real AI systems use exactly this trick
(often called **grammar-constrained** or **structured** decoding) when they must
produce valid JSON, valid code, or answers that fit a form: rather than hoping a
huge model behaves, they let a rule-checker veto illegal tokens. Knowing the
rules perfectly is often easier than learning them perfectly.
````

In `docs/concepts/README.md` append item 9:

```markdown
9. [The referee](the-referee.md) — grammar-constrained decoding: vetoing illegal
   characters so generated Pebble always parses.
```

In `README.md`, inside "The flagship: teach PyLLM to write Pebble" section, after the paragraph ending "...as you train longer or bigger. Almost no LLM project can grade its own output like this.", add (fill the first `parse rate` line with the real output of `uv run pyllm pebble --seed 7 --temperature 0.5 --score` and the second with the real output of the `--constrained` run — copy them verbatim):

````markdown
### Call in the referee: always-valid Pebble

Add `--constrained` and a from-scratch grammar **referee** vetoes every character
that would break Pebble's rules, so the model rolls again until it picks a legal
one. Same tiny brain — but now every program it returns parses:

```console
$ uv run pyllm pebble --seed 7 --temperature 0.5 --constrained --score
...
parse rate: 100.0% of N programs
```

From Python: `generate_pebble(constrained=True)`. See
[`docs/concepts/the-referee.md`](docs/concepts/the-referee.md).
````

In the roadmap (`docs/superpowers/plans/2026-06-22-pyllm-roadmap.md`): change the status banner to say all **six** plans are done; add a table row:

```markdown
| 6 | **Grammar-constrained decoding** — `2026-10-16-pyllm-06-grammar-constrained-decoding.md` | ✅ Done | `pyllm/pebble/referee.py` (viable/complete prefix checker), referee-aware `sample_next`/`generate`, `pyllm pebble --constrained`, `generate_pebble(constrained=True)` — 100% parse rate. Corpus re-validated after formatting. |
```

add `- Plan 6: docs/concepts/the-referee.md — the referee with a whistle.` to "Concept docs each plan must deliver", and replace the "## Grammar-constrained decoding" section body with: `Delivered as Plan 6. Follow-up (pebble-lang repo): expose a constrained option through Pebble's import "llm" module.`

In the design spec (`docs/superpowers/specs/2026-06-22-pyllm-design.md`): in §6 change "**Stretch goal (clearly out of the initial scope): grammar-constrained decoding**" to "**Grammar-constrained decoding (delivered in Plan 6)**"; in §11 delete the bullet "Grammar-constrained decoding (deferred stretch goal, see §6)."

- [ ] **Step 4: Run all checks**

Run: `uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run pyright`
Expected: all PASS, coverage ≥ 90%.

- [ ] **Step 5: Commit**

```bash
git add docs README.md tests/test_docs.py
git commit -m "docs: add the-referee concept doc; mark Plan 6 complete"
```
