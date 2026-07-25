# How a model learns (the guessing game)

Here's the analogy to hold onto: imagine a game where you try to guess the next
letter of a Pokémon name, over and over. Every time you guess, a friend tells you
how wrong you were. If you keep score and adjust after every round, you slowly
get better. That is exactly how our model learns.

## The four steps, on repeat

1. **Guess.** The model reads some letters and gives a score to every possible
   next letter.
2. **Measure the surprise (the *loss*).** We look at how much probability the
   model gave the *correct* next letter. Confident and right → tiny loss.
   Confident and wrong → huge loss. This number is the **loss**: lower is better.
3. **Trace the blame (the *gradient*).** We walk backward through every step the
   model took and work out, for each internal dial, "which way should I turn you,
   and how much, to make the loss smaller?" That direction-and-amount is the
   **gradient** (see the breadcrumb-trail analogy in `autograd.md`).
4. **Nudge the dials.** The optimizer turns every dial a tiny step in the
   blame-reducing direction.

Repeat thousands of times and the loss falls: the model has learned.

## A tiny worked example

Train a bigram on `"abcabcabc..."`. At first it guesses randomly (loss ≈ log of
the vocab size). After a few hundred steps it has noticed "after `a` comes `b`,
after `b` comes `c`" and the loss drops close to zero — it has *memorized the
pattern*.

## Why does this matter?

Every AI that "learns from data" — from Pokémon-name dreamers to giant chatbots —
is doing this same loop: guess, measure surprise, trace blame, nudge. The only
differences are how big the model is and how much text it reads.
