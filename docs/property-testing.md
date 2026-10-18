# Property Testing (The Tireless Examiner)

## The Tireless Examiner

Imagine a maths teacher checking a pupil's homework. The usual way: set
one question ("what's the slope of `a * b` when `a` is 2?"), check the
answer, and tick it. That's an **example test**. It's useful, but it only
tells you about the one question the teacher thought of.

Now imagine a tireless examiner. They set thousands of questions -- tiny
numbers, negative numbers, odd-shaped grids nobody would think to try --
and for each one they check a single rule:

> However you worked out the slope, it must match the slope I measure by
> nudging the number a tiny bit and seeing what happens.

That rule is a **property**. It doesn't say what any one answer is. It
says something that must be true for *every* question.

When the examiner finds a question the pupil gets wrong, they don't just
hand back a huge, messy sum. They keep making the question simpler, bit
by bit, checking it still goes wrong each time, until it's as small as it
can be. That simplifying is called **shrinking**, and it's the best part:
the smallest failing question usually points straight at the mistake.

In Python, the tireless examiner is a library called
[Hypothesis](https://hypothesis.readthedocs.io/). PyLLM's property tests
live in `tests/test_properties.py`, next to the ordinary example tests.

## A Property from PyLLM

PyLLM learns by [following breadcrumbs back down a hill](concepts/autograd.md):
`backward()` works out, for every number in a sum, how much the answer
would change if that number grew a little. That's its *gradient*, or
slope. If the slopes are wrong, the model learns the wrong lesson, so the
examiner checks them:

```python
# ~~ Numbers for the autograd engine ~~

# Small arrays (at most 3 dimensions, at most 3 long each) keep the
# finite-difference check fast: it runs the sum twice per number.
MAX_DIMS = 3
MAX_SIDE = 3
SMALL = 2.0
NOT_TOO_SMALL = 0.5


def small_numbers() -> st.SearchStrategy[float]:
    """Make an everyday number between -2 and 2."""
    return st.floats(min_value=-SMALL, max_value=SMALL)


def numbers_away_from_zero() -> st.SearchStrategy[float]:
    """Make a number between 0.5 and 2, or between -2 and -0.5 (safe to divide by)."""
    # Dividing by a number near zero gives a slope so steep that the
    # nudge-and-measure check can't measure it accurately.
    return st.one_of(
        st.floats(min_value=-SMALL, max_value=-NOT_TOO_SMALL),
        st.floats(min_value=NOT_TOO_SMALL, max_value=SMALL),
    )


def two_shapes_that_broadcast() -> st.SearchStrategy[tuple[tuple[int, ...], ...]]:
    """Make two array shapes numpy can combine, like (2, 3) and (3,) or (2, 1) and (1, 4)."""
    return hnp.mutually_broadcastable_shapes(
        num_shapes=2, max_dims=MAX_DIMS, max_side=MAX_SIDE
    ).map(operator.attrgetter("input_shapes"))


def tensors(shape: tuple[int, ...], numbers: st.SearchStrategy[float]) -> st.SearchStrategy[Tensor]:
    """Make a Tensor of the given shape, filled from the given numbers."""
    return hnp.arrays(np.float64, shape, elements=numbers).map(Tensor)


@st.composite
def two_input_problems(draw: st.DrawFn) -> tuple[Tensor, Tensor, Tensor]:
    """Make two tensors whose shapes broadcast, plus a weight for every number of the answer."""
    a_shape, b_shape = draw(two_shapes_that_broadcast())
    a = draw(tensors(a_shape, numbers_away_from_zero()))
    b = draw(tensors(b_shape, numbers_away_from_zero()))
    weights = draw(tensors(np.broadcast_shapes(a_shape, b_shape), small_numbers()))
    return a, b, weights


def two_input_ops() -> st.SearchStrategy[Callable[[Tensor, Tensor], Tensor]]:
    """Make one of +, -, * and /."""
    return st.sampled_from([plus, minus, times, divided_by])


# The nudge-and-measure slope is only an estimate, good to about 1e-8 for
# numbers this size, so we allow a small difference.
RELATIVE_TOLERANCE = 1e-5
ABSOLUTE_TOLERANCE = 1e-6


def assert_gradients_match(make_output: Callable[[], Tensor], *inputs: Tensor) -> None:
    """Run backward() on `make_output()`, then check every input's gradient by nudging it."""
    make_output().sum().backward()
    for tensor in inputs:
        measured = numerical_grad(make_output, tensor)
        np.testing.assert_allclose(
            tensor.grad, measured, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE
        )


@given(problem=two_input_problems(), op=two_input_ops())
def test_two_input_gradients_match_nudging(
    problem: tuple[Tensor, Tensor, Tensor], op: Callable[[Tensor, Tensor], Tensor]
) -> None:
    """For +, -, * and / on any shapes that broadcast, backward() matches the measured slope."""
    a, b, weights = problem
    assert_gradients_match(lambda: op(a, b) * weights, a, b)
```

Line by line:

- **`small_numbers()` and `numbers_away_from_zero()`** are **strategies**:
  recipes for making random inputs. Each is a named function, so the test
  reads like a sentence. The second keeps away from zero, and its comment
  says why: we only restrict inputs for a reason we can write down.
- **`two_shapes_that_broadcast()`** asks Hypothesis for two shapes numpy
  is happy to combine, such as `(2, 1)` and `(1, 3)`. This is where
  gradients like to go wrong: the slope has to be squashed back to each
  input's own shape.
- **`two_input_problems()`** is a `@st.composite` strategy: a recipe that
  uses other recipes. It makes `a`, `b`, and a grid of `weights`, one for
  every number in the answer.
- **`numerical_grad`** is the examiner's own way of finding the slope:
  nudge one number up, nudge it down, and see how much the sum moved.
  It's slow and a tiny bit fuzzy, which is why there's a tolerance, but
  it's built a completely different way from `backward()`. A check that
  copied `backward()`'s working could never catch its mistakes.
- **`@given(...)`** runs the test many times (100 in a normal run), each
  with a fresh problem and a fresh operator.
- **`np.testing.assert_allclose`** (inside `assert_gradients_match`) is
  the rule itself: the slope `backward()` worked out must match the
  measured one, give or take the tolerance.

Why the `weights`? Without them, the answer is just added up, so every
number in it counts exactly once. An operator that forgot to pass the
slope along (`out.grad`) would still look right. Multiplying by random
weights makes every number count differently, so that mistake shows up.

Notice what the test *doesn't* say. It never says "the slope of `a * b`
with respect to `a` is `b`". It doesn't need to know any answers. It
only needs a rule that's always true.

And it isn't fooled by laziness: **a do-nothing `backward()` fails it**,
because every gradient would stay at zero while the measured slope
doesn't.

PyLLM has nine more properties:

| Rule | Test |
|---|---|
| For tanh, exp, squaring, sums, averages and reshapes, `backward()` matches the measured slope | `test_one_input_gradients_match_nudging` |
| For `a @ b` on any vectors or tables that fit, batches included, `backward()` matches the measured slope | `test_matmul_gradients_match_nudging` |
| Encoding a message and decoding it gives back exactly the same message | `test_char_tokenizer_round_trip` |
| BPE may glue letters into chunks, but decoding still gives back exactly the message | `test_bpe_round_trip` |
| Every number BPE hands out is between 0 and `vocab_size - 1` | `test_token_numbers_fit_the_vocabulary` |
| Saving a model and loading it gives the same recipe, tokenizer (letters or BPE chunks) and weights, bit for bit | `test_checkpoint_round_trip` |
| Cut Pebble off anywhere: if the referee calls it finished, Pebble's own parser agrees | `test_whatever_the_referee_passes_pebble_passes` |
| Tidy Pebble is finished, and the referee never whistles part-way through typing it | `test_every_prefix_of_a_finished_program_is_viable` |
| Whatever a model types, the referee answers, and never whistles at finished text | `test_the_referee_always_answers` |

The tokenizer tests only write messages with letters the tokenizer has
seen. That's a known limit, not a forgotten case: a `CharTokenizer` that
meets a brand-new letter raises a `KeyError`, as its docstring says, and
`BPETokenizer` starts from the same single letters.

## Try It Yourself

Let's break the tokenizer on purpose and watch the examiner find it.

Open `src/pyllm/tokenizer/char.py` and change one line in `encode`, as if
you'd decided to tidy up stray spaces:

```python
return [self.stoi[char] for char in text.strip()]  # was: ...for char in text]
```

Run the property tests:

```bash
uv run pytest tests/test_properties.py --no-cov
```

Hypothesis reports this (trimmed to the interesting part):

```
text_and_msg = (' ', ' ')

    @given(text_and_message())
    def test_char_tokenizer_round_trip(text_and_msg: tuple[str, str]) -> None:
        """Encoding a message and decoding the numbers gives back exactly the same message."""
        text, message = text_and_msg
        tokenizer = CharTokenizer(text)
>       assert tokenizer.decode(tokenizer.encode(message)) == message
E       AssertionError: assert '' == ' '
E         Strings contain only whitespace, escaping them using repr()
E         - ' '
E         ?  -
E         + ''
E       Failing test case: test_char_tokenizer_round_trip(
E           text_and_msg=(' ', ' '),
E       )
```

Look at how small that is. Hypothesis could try long strings full of
emoji, accents and Chinese, but it shrank the failure all the way down to a
tokenizer that knows **one letter, a space**, and a message that is
**one space**. Encode it, decode it, and the space has vanished.

It's `test_char_tokenizer_round_trip` that catches it. The other nine
properties stay green, because none of them encode with a
`CharTokenizer`. The example tests all still pass. Only the property
notices: every example message, like `"hello"`, happens to have no spaces
at either end.

Now put the line back to `for char in text]` and run the tests again.
They pass.

!!! tip "Older Hypothesis versions"
    Older versions of Hypothesis print `Falsifying example:` instead of
    `Failing test case:`. It means the same thing.

## A Bug We Found

When we first wrote `test_the_referee_always_answers`, it failed on
PyLLM's real code. The [referee](concepts/the-referee.md) crashed:

```
  | hypothesis.errors.FlakyFailure: Inconsistent results from replaying a test case!
  |   last: INTERESTING from RecursionError at src/pyllm/pebble/referee.py:292
  |   this: INTERESTING from RecursionError at src/pyllm/pebble/referee.py:288 (3 sub-exceptions)
```

This time there's no tidy `Failing test case:`, and that's a clue in
itself. The referee reads code with functions that call each other once
for every open bracket. Each one takes a little of Python's stack, and
Python stops with a `RecursionError` when the stack runs out. Give it
about 120 brackets inside each other, `let a = ((((...1...))))`, and it
crashed. But *exactly* how many depends on how much of the stack was
already in use, so each time Hypothesis replayed a smaller example, the
crash landed on a different line. A **flaky** test is one that asks the
same question and gets a different answer. Hypothesis noticed, and told
us. Running it again until it went green would have hidden the bug; the
right move was to fix the crash.

The example tests never nested more than a couple of brackets. Real code
doesn't. But a small model can get stuck typing `((((((`, and then
`generate()`, which asks the referee about every letter, would crash.

The old code had no counting at all. The fix counts how deep we are,
and gives the whistle past a fixed limit (`_MAX_NESTING = 90`) instead of
waiting for Python to run out. Every statement, operator's right-hand
side, minus sign, bracket, call, `not` and block now steps in with
`_open()` and back out with `_close()`:

```python
def _open(self) -> None:
    """Step one level deeper (see `_MAX_NESTING`); too deep is a foul."""
    self.nesting += 1
    if self.nesting > _MAX_NESTING:
        raise _FoulError
```

A fixed limit gives the same answer every time, wherever it's called
from. Pebble has a limit too: 100 levels, counting the same kinds of
step. The referee counts at least everything Pebble counts and stops at
90, so anything it accepts is never too deep for Pebble. The crash also became plain example tests
in `tests/pebble/test_referee.py`:

```python
def test_deeply_nested_brackets_get_the_whistle_not_a_crash():
    """Deeply nested brackets get the whistle instead of crashing the referee.

    Found by test_the_referee_always_answers in tests/test_properties.py.
    """
    text = "let a = " + "(" * 1000 + "1" + ")" * 1000
    assert not viable(text)
    assert not complete(text)
```

## Run It Yourself

Run just the property tests (100 examples each, the same every time):

```bash
uv run pytest tests/test_properties.py --no-cov
```

The `--no-cov` turns off the coverage check, which expects the *whole*
test suite to run and would fail on one file.

Run them the long way (2,000 random examples each, different every run)
to hunt for rarer bugs. It takes a few minutes:

```bash
uv run pytest tests/test_properties.py --no-cov --hypothesis-profile=thorough
```

The normal run uses a fixed seed, so it tries the same inputs every time
and a passing build means the same thing on every computer. The thorough
run is random, so each time you run it the examiner sets brand-new
questions. Both settings live in `conftest.py` at the top of the project.

## Your Turn

Can you write your own property? Try [sampling](concepts/sampling.md):
"whatever scores you give `sample_next` in `pyllm/generate.py`, with
`temperature=0.0` it picks the highest one, just like `np.argmax`". What's
the strategy? What's the `assert`? Then break `sample_next` on purpose
and see what Hypothesis shrinks it to.
