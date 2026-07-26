# How a Transformer is built (stacking the pieces)

A **Transformer** is the engine inside modern language models — including ours.
It sounds fancy, but the analogy is simple: it's built from a few plain bricks we've
already met, stacked up like Lego.

## The bricks

1. **Token embedding** — turn each letter into a little list of numbers (its
   "meaning"). See `embeddings.md`.
2. **Position embedding** — also add a number for *where* the letter sits,
   because "sat the cat" and "the cat sat" use the same letters in a different
   order.
3. **Attention** — every position looks back at earlier positions and pulls in
   the clues that matter. See `attention.md` ("re-reading a sentence").
4. **Feed-forward** — each position then does a little private thinking on what
   it gathered.
5. **LayerNorm + residuals** — tidy the numbers before each step, and always
   keep the original notes and only *add* edits (a residual connection), so deep
   stacks stay stable.

## The repeatable block

Steps 3–5 together make one **Transformer block**: *communicate* (attention),
then *think* (feed-forward), each wrapped so nothing gets lost. Stack a few
blocks and finish with one last layer that turns each position into scores for
the next letter. That's a **GPT**.

## A tiny worked example

Our bundled Pokémon model is a real GPT with a couple of blocks. Reading
`pi`, the token+position embeddings say *what* and *where*, attention notices
"names that start `pi` often continue `pika...`", the feed-forward sharpens the
hunch, and the final layer scores `k` highest. Roll the dice (see `sampling.md`)
and out comes another letter.

## A knob for later: dropout

Real transformers add one more trick while *training*: **dropout**. Think of a
sports team practising with a few random players sitting out each drill, so the
whole team gets good and nobody becomes a single point of failure. During
training we randomly ignore a fraction of the signals inside each block; during
real use (generation) everyone plays and dropout does nothing. Our
`TransformerBlock` and `GPT` accept a `dropout` setting (default `0.0`, i.e.
off), so you can switch it on to help a bigger model avoid over-memorising.

## Why does this matter?

This exact structure — embeddings, stacked attention+feed-forward blocks — is
what powers the biggest AI models in the world. Ours is tiny, but it is the
*same machine*, and you can read every line of it.
