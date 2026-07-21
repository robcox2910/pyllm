# Sampling: how the model chooses what to say next

The model never hands you a single answer. It hands you a *score for every
possible next letter*. Turning those scores into an actual choice is called
**sampling**, and how boldly we choose changes the whole personality of the
output.

## The dice analogy

Think of the scores as a weighted dice: letters the model likes have bigger
sides. **Sampling** is rolling that dice.

- **Temperature** is the boldness dial.
  - `temperature = 0`: never roll — always take the single best guess. Safe, but
    it repeats itself and gets stuck.
  - `temperature = 1`: roll fairly, trusting the model's confidence.
  - `temperature > 1`: flatten the dice so even unlikely letters get a chance —
    wilder, more surprising, sometimes nonsense.
- **top-k** is a guardrail: "only allow the k most likely letters onto the dice."
  It stops the wild rolls from picking something absurd while still letting the
  model be creative among sensible options.

## A tiny worked example

Scores favour `u` after `q`. At `temperature = 0.2` you'll almost always get
`qu...`. Crank it to `temperature = 1.5` and you might see `qx` or `qe` — rarer,
riskier names. `top-k = 3` keeps the surprises to the three best letters only.

## Why does this matter?

Sampling is the knob between *boring-but-correct* and *creative-but-chaotic*.
Every chatbot you've used has a temperature setting under the hood; picking it
well is the difference between a dull answer and a delightful one.
