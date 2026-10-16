# PyLLM Plan 6 — Grammar-Constrained Decoding ("the referee")

**Status:** approved design, 2026-10-16. Picks up the deferred stretch goal from
spec §6 of `2026-06-22-pyllm-design.md`.

## 1. Goal

Today the bundled Pebble brain writes Pebble-*flavoured* code: only some of its
output parses. Plan 6 adds a **referee** that watches generation one character at
a time and blows the whistle on any character the Pebble grammar would not allow.
The sampler then crosses that character out and rolls again. With the referee on,
**100% of returned programs parse under Pebble's real parser** — and the whole
thing is built from scratch, readable by a 12-year-old (RULE #1 still applies).

Success criteria:

- `pyllm pebble --constrained --score` reports `parse rate: 100.0%` for any seed.
- `generate_pebble(constrained=True)` likewise only returns valid programs.
- No new runtime dependencies (numpy only). pebble-lang stays optional and is
  only used by tests/scoring as the independent judge.
- A kid-friendly `docs/concepts/the-referee.md`.

## 2. The referee — `src/pyllm/pebble/referee.py`

Two public functions:

- `viable(text) -> bool` — "could this text still grow into a valid program?"
- `complete(text) -> bool` — "is this text a finished, valid program right now?"

Both judge only the **current program**: the text after the last blank line
(`"\n\n"`), matching how the corpus and the scorer separate programs. Everything
before that was already judged complete when its blank line was allowed.

**Implementation:** a small hand-written lexer plus a recursive-descent parser
over the *prefix*. Running out of input anywhere means "need more" (viable, not
complete); any mismatch means "foul" (not viable). A partial final word (e.g.
`whi`) is viable if it is a prefix of a keyword or a legal name in that position.
The referee re-parses the current program on each call; programs are short
(~10 lines), so this is fast enough and far simpler than an incremental machine.

**Grammar (a strict subset of Pebble, in its canonical formatting):**

```
program   := stmt ("\n" stmt)* "\n"?
stmt      := "let " NAME " = " expr
           | NAME " = " expr
           | "print(" expr ")"
           | "return" (" " expr)?             -- only inside a fn body
           | "if " expr " {" block ("} else {" block)? "}"
           | "while " expr " {" block "}"
           | "for " NAME " in range(" expr ") {" block "}"
           | "fn " NAME "(" (NAME (", " NAME)*)? ") {" block "}"
           | "# " any-text-to-end-of-line
block     := ("\n" INDENT stmt)+ "\n" OUTER-INDENT   -- non-empty
expr      := or
or        := and (" or " and)*
and       := not (" and " not)*
not       := "not " not | cmp
cmp       := sum ((" == " | " != " | " < " | " <= " | " > " | " >= ") sum)?
sum       := term ((" + " | " - ") term)*
term      := unary ((" * " | " / " | " % ") unary)*
unary     := "-" unary | primary
primary   := NUMBER | STRING | "true" | "false" | NAME
           | NAME "(" (expr (", " expr)*)? ")" | "(" expr ")"
NAME      := [a-z_][a-z0-9_]*  that is not a keyword
NUMBER    := [0-9]+
STRING    := '"' [^"\\\n{}]* '"'      -- no escapes, no {interpolation}
```

- INDENT is exactly 4 spaces per nesting level; OUTER-INDENT is the enclosing
  level's indent before the closing `}`.
- Keywords (never names): `let fn if else while for in return print and or not
  true false range`.
- Comparisons do not chain (one per `cmp`) — stricter than Pebble, which is fine.
- Strict canonical spacing keeps the referee small and matches the formatter's
  house style the corpus is written in, so the model is not fighting it.

Because every string the referee accepts is valid Pebble (a subset), the real
parser accepts it too. Tests verify this against pebble-lang.

## 3. Constrained sampling — `src/pyllm/generate.py`

- `sample_next(..., allowed=None)`: `allowed` is a callback `token_id -> bool`.
  After temperature/top-k shaping, sample; if the token is not allowed, zero its
  probability, renormalise and roll again (**rejection sampling** — the same
  distribution as masking everything up front, but usually only 1–2 checks).
  Temperature 0 picks the most likely *allowed* token. If top-k left no allowed
  token, fall back to the full vocabulary (still refereed). If nothing at all is
  allowed, raise `RuntimeError` (a referee bug, never expected).
- `generate(..., referee=None)`: when given a referee (`pyllm.pebble.referee`
  module or any object with `viable`/`complete`), each candidate is checked with
  `referee.viable(text_so_far + char)`. The loop remembers the longest prefix that
  was `complete`; when `max_new_tokens` runs out, output is trimmed back to it, so
  a half-written program is never returned. A non-viable prompt raises a friendly
  `ValueError`. If no complete point exists, return the prompt unchanged.

## 4. Surfaces

- CLI: `pyllm pebble --constrained` (works with `--score`).
- API: `generate_pebble(..., constrained: bool = False)`.
- `pyllm.pebble` exports `viable` and `complete`.
- Out of scope: wiring `constrained` through PyStack's Pebble `import "llm"`
  module (lives in the pebble-lang repo) — noted as a follow-up.

## 5. Corpus fix

`build_corpus` currently validates harvested snippets *before* canonicalising and
squeezing blank lines, so 15 of 994 bundled programs no longer parse. Fix: drop
invalid programs after the final transformation. Strip the 15 invalid programs
from the bundled `pebble_corpus.txt` (no retrain; checkpoint unchanged).

## 6. Testing (TDD)

- Referee accepts every program from `random_program` over 300 seeds (rendered and
  canonicalised) as `complete`, and every prefix as `viable`.
- Referee rejects a table of fouls: wrong indent, `return` outside `fn`, empty
  block, keyword as name, chained comparison, mid-program blank line, unclosed
  string across a newline, `}` at top level.
- `sample_next`: disallowed tokens never chosen; greedy picks best allowed; top-k
  fallback; all-disallowed raises.
- `generate` with a referee: trimming to last complete point; bad prompt raises.
- Bundled checkpoint, constrained, several seeds: 100% parse under the real
  parser (skipped without pebble-lang) and `complete` on every program.
- CLI flag and API flag tests; corpus regression test (every bundled program
  parses, skipped without pebble-lang).
- ruff, pyright and coverage ≥ 90% stay green.

## 7. Docs

- `docs/concepts/the-referee.md`: analogy (a referee who blows the whistle on
  illegal moves so you pick again), tiny worked example, "why does this matter?"
  (real-world structured output / JSON mode uses the same idea).
- Concepts index, README section with before/after parse rate, roadmap Plan 6
  row, and spec §6/§11 updated from "deferred" to "done".
