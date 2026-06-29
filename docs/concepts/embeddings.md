# What Is an Embedding? (A Map of Meaning)

> **Analogy:** An embedding gives every token its own spot on a giant "map of
> meaning". Tokens that mean similar things end up close together, the way
> similar towns sit near each other on a real map.

## From a number to a vector

A token id like `42` is just a label -- it tells us nothing about meaning. An
`Embedding` is a big lookup table with one row of numbers per token. We swap each
id for its row (its **vector**). That little list of numbers is the token's
position on the map.

```
id 42  -->  [ 0.13, -0.88, 0.42, ... ]
```

## The map is *learned*

At the start every row is random -- the map is nonsense. As the model trains,
tokens that get used in similar ways drift toward each other. Nobody tells the
model "cat and dog are similar"; it discovers it, and the map arranges itself.

## Why it matters

Only the rows the model actually uses get nudged when learning (look at
`embedding`'s breadcrumb rule). Embeddings turn cold id numbers into rich
vectors that attention and the Transformer can actually reason about.
