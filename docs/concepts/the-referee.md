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
...
parse rate: 0.0% of 1 programs

$ uv run pyllm pebble --seed 7 --constrained --score  # with the referee
let value = step()
print(14 - 82)
for i in range(4) {
    x = b
    while 57 >= 48 and 83 != a {
        print(compute())
        let z = helper()
    }
}

parse rate: 100.0% of 1 programs
```

## Why does this matter?

The model didn't get any smarter — it's the same tiny brain. We just made it
*impossible* to break the rules. Real AI systems use exactly this trick
(often called **grammar-constrained** or **structured** decoding) when they must
produce valid JSON, valid code, or answers that fit a form: rather than hoping a
huge model behaves, they let a rule-checker veto illegal tokens. Knowing the
rules perfectly is often easier than learning them perfectly.
