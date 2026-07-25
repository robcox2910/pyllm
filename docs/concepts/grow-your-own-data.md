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
