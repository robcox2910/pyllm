# The Breadcrumb Trail (Autograd)

> **Analogy:** Imagine walking through a forest and dropping a breadcrumb at
> every step. If you take a wrong turn, you can follow the breadcrumbs *backward*
> to find exactly where you went wrong — and fix it. That is exactly how our
> neural network learns. This trick is called **autograd** (automatic gradients).

## The problem

A neural network is just a big pile of numbers (we call them **weights**). When
it makes a guess, it's usually a bit wrong at first. To get better, it needs to
know: *"If I nudge this one number up a tiny bit, does my answer get better or
worse?"* That "does it get better or worse, and by how much" is called a
**gradient**.

A real network has *millions* of numbers. Working out every gradient by hand
would take forever. So we make the computer do it automatically.

## How the breadcrumbs work

Every time we do a piece of maths (add, multiply, etc.), our `Tensor` quietly
writes down:

1. **who its parents were** (which tensors went into it), and
2. **how to pass the blame backward** to those parents.

That "write down how to pass the blame backward" note is the breadcrumb.

When we finally measure how wrong we were (the **loss**), we call
`loss.backward()`. The computer follows every breadcrumb in reverse — from the
final answer all the way back to each weight — adding up exactly how much each
number was responsible for the mistake. That responsibility is stored in
`.grad`.

## A tiny worked example

```python
a = Tensor([2.0])
b = Tensor([3.0])
c = a * b        # c is 6, and remembers "a and b made me"
c.backward()     # follow the breadcrumbs back
# a.grad is 3  -> "if a goes up by 1, c goes up by 3" (because b = 3)
# b.grad is 2  -> "if b goes up by 1, c goes up by 2" (because a = 2)
```

## How do we *know* it's right?

We cheat-check it the slow, obvious way: nudge a number up a tiny bit, see how
much the answer changed, and compare. That's **numerical gradient checking**
(`numerical_grad`). If the fast breadcrumb answer matches the slow nudge answer,
we trust the breadcrumbs.

## Why this matters

This one idea — leaving a trail so you can walk back and learn from mistakes — is
the engine inside *every* modern AI, including the big ones. Everything else in
PyLLM (attention, transformers, the whole GPT) is built on top of these
breadcrumbs.
