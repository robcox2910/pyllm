# PyLLM concepts, explained for a curious 12-year-old

These pages explain every big idea in PyLLM using real-world analogies — no jargon
first. Read them roughly in this order; each builds on the last.

1. [The breadcrumb trail (autograd)](autograd.md) — how a network remembers what
   it did so it can learn from mistakes.
2. [What is a token?](tokens.md) — chopping text into Lego bricks.
3. [What is an embedding?](embeddings.md) — giving each token a place on a map of
   meaning.
4. [Attention is re-reading a sentence](attention.md) — how a token decides which
   earlier words matter.
5. [How a Transformer is built](transformer.md) — stacking the pieces into a GPT.
6. [How a model learns](how-a-model-learns.md) — loss, gradients, the training loop.
7. [Sampling](sampling.md) — how the model chooses what to say next (temperature,
   top-k).
8. [When there's no data, grow your own](grow-your-own-data.md) — the Pebble
   corpus: harvest, generate, and grade.
