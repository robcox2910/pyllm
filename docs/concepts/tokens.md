# What Is a Token? (Chopping Text into Lego Bricks)

> **Analogy:** A model can't read letters the way you do. First we chop text into
> small pieces called **tokens** -- like snapping a sentence into Lego bricks --
> and give every brick a number. The model only ever sees the numbers.

## Two ways to chop

**Character tokens** (`CharTokenizer`): every single letter is its own brick.
`"cat"` becomes `c, a, t` -> `[2, 0, 19]`. Simple and lossless for any character
it has already seen (an unseen character is a KeyError), but the model has to
spell out everything one letter at a time.

**BPE tokens** (`BPETokenizer`): we let the computer *learn* common chunks. It
keeps gluing together the two neighbours that show up together most often, so
pieces like `"th"`, `"ing"`, or `"ab"` become single bricks. Now the model reads
in bigger, smarter gulps.

## Encode and decode

- **Encode** = text -> numbers (chop into bricks, write down their numbers).
- **Decode** = numbers -> text (look the bricks back up and glue them together).

They are perfect opposites: `decode(encode("hello")) == "hello"`.

## Why it matters

Tokens are the model's alphabet. Every other part of PyLLM -- embeddings,
attention, the whole GPT -- works on these token numbers, never on raw letters.
