# Attention (Re-reading a Sentence to Understand It)

> **Analogy:** When you read "The trophy didn't fit in the case because *it* was
> too big," you instantly **look back** to decide what "it" means. Attention is
> the model doing exactly that: each word looks back at earlier words and decides
> which ones matter right now.

## Query, key, value

Every token produces three things:
- a **query** -- "what am I looking for?"
- a **key** -- "here's what I am, in case you're looking for me",
- a **value** -- "here's the information I'll hand over if you pick me".

A token compares its query against every earlier token's key. Good matches get a
high score; we turn the scores into shares with **softmax** and use them to take
a weighted blend of the **values**. That blend is what the token learned by
looking back.

## Looking back only (the causal mask)

A token may only look at itself and tokens *before* it -- you can't use words you
haven't read yet. We enforce this with a **causal mask** that blocks the future
(it adds a huge negative number to future scores so softmax gives them ~0 share).

## Many heads

One head tracks one kind of relationship. **Multi-head attention** runs several
in parallel -- one might follow "who did what", another "when" -- then blends
their reports. Stack attention + a little per-token thinking (feed-forward) and
you have a **Transformer block**, the repeatable brick of a GPT.
