"""Property-based tests -- rules that must hold for *every* input.

An ordinary test checks one example someone thought of: "the gradient of
``a * b`` at a=2, b=3 is 3". A property test states a rule instead, such as
"the gradient backward() works out matches the slope you measure by nudging
the input, for any numbers and any shapes", and lets Hypothesis invent
hundreds of inputs to try to break it. If one breaks the rule, Hypothesis
*shrinks* it down to the smallest input that still fails, and shows you that.

Each test below says its rule in plain English in its docstring. The
strategies at the top are recipes for making random inputs; each one is
named for what it makes.

See ``docs/property-testing.md`` for the walk-through.
"""

import operator
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pyllm.autograd import Tensor
from pyllm.autograd.gradcheck import numerical_grad
from pyllm.checkpoint import load, save
from pyllm.models import LanguageModel, ModelConfig, build_model
from pyllm.pebble import PEBBLE_AVAILABLE, complete, is_valid, viable
from pyllm.tokenizer import BPETokenizer, CharTokenizer

# --- Strategies: recipes for random inputs -----------------------------------

# ~~ Text for the tokenizers ~~

MAX_TRAINING_LETTERS = 40
MAX_MERGES = 10


def training_texts() -> st.SearchStrategy[str]:
    """Make some text to learn an alphabet from: any letters at all, at least one."""
    return st.text(min_size=1, max_size=MAX_TRAINING_LETTERS)


def repetitive_texts() -> st.SearchStrategy[str]:
    """Make text from just a few letters, so pairs repeat and BPE has chunks to learn."""
    return st.text(alphabet="ab c", min_size=1, max_size=MAX_TRAINING_LETTERS)


@st.composite
def text_and_message(
    draw: st.DrawFn, texts: st.SearchStrategy[str] | None = None
) -> tuple[str, str]:
    """Make a training text (from `texts`, or any letters), plus a message using only its letters."""
    text = draw(training_texts() if texts is None else texts)
    # Known limit (deferred): a tokenizer only knows the letters it was built
    # from. CharTokenizer's docstring says encoding an unseen character raises
    # a KeyError, and BPETokenizer.encode starts from the same single letters
    # (it looks each one up in `stoi`), so messages here use only letters the
    # tokenizer has seen.
    letters = draw(st.lists(st.sampled_from(sorted(set(text)))))
    message = "".join(letters)
    return text, message


def merge_counts() -> st.SearchStrategy[int]:
    """Make how many chunks BPE may learn: none at all, up to a handful."""
    return st.integers(min_value=0, max_value=MAX_MERGES)


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


def two_shapes_that_matmul() -> st.SearchStrategy[tuple[tuple[int, ...], ...]]:
    """Make two shapes that `@` can multiply, like (2, 3) and (3, 4), (3,) and (3, 4), or (3,) and (3,)."""
    # The `?` lets either side be a plain 1-D vector, as numpy's matmul allows;
    # a batch may still sit in front of any side that is a table.
    return hnp.mutually_broadcastable_shapes(
        signature="(m?,n),(n,p?)->(m?,p?)", max_dims=MAX_DIMS, max_side=MAX_SIDE
    ).map(operator.attrgetter("input_shapes"))


def tables() -> st.SearchStrategy[tuple[int, ...]]:
    """Make the shape of a table with 2 or 3 dimensions, like (2, 3) or (1, 3, 2)."""
    return hnp.array_shapes(min_dims=2, max_dims=MAX_DIMS, max_side=MAX_SIDE)


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


@st.composite
def matmul_problems(draw: st.DrawFn) -> tuple[Tensor, Tensor, Tensor]:
    """Make two vectors or tables `@` can multiply, plus a weight for every number of the answer."""
    a_shape, b_shape = draw(two_shapes_that_matmul())
    a = draw(tensors(a_shape, small_numbers()))
    b = draw(tensors(b_shape, small_numbers()))
    weights = draw(tensors((a @ b).shape, small_numbers()))
    return a, b, weights


def plus(a: Tensor, b: Tensor) -> Tensor:
    """Add two tensors."""
    return a + b


def minus(a: Tensor, b: Tensor) -> Tensor:
    """Take one tensor away from another."""
    return a - b


def times(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two tensors."""
    return a * b


def divided_by(a: Tensor, b: Tensor) -> Tensor:
    """Divide one tensor by another."""
    return a / b


def two_input_ops() -> st.SearchStrategy[Callable[[Tensor, Tensor], Tensor]]:
    """Make one of +, -, * and /."""
    return st.sampled_from([plus, minus, times, divided_by])


def total(t: Tensor) -> Tensor:
    """Add up every number."""
    return t.sum()


def row_totals(t: Tensor) -> Tensor:
    """Add up along the last axis."""
    return t.sum(axis=-1)


def column_averages(t: Tensor) -> Tensor:
    """Average down the first axis, keeping it as size 1."""
    return t.mean(axis=0, keepdims=True)


def squared(t: Tensor) -> Tensor:
    """Multiply every number by itself."""
    return t**2


def flipped(t: Tensor) -> Tensor:
    """Swap rows and columns."""
    return t.transpose()


def flattened(t: Tensor) -> Tensor:
    """Lay every number out in one long row."""
    return t.reshape((-1,))


def one_input_ops() -> st.SearchStrategy[Callable[[Tensor], Tensor]]:
    """Make one of the one-input operations: tanh, exp, squaring, sums, averages, reshapes."""
    # relu has a sharp corner at 0 and log needs positive numbers, so the
    # nudge-and-measure check can't judge them at every number we make; the
    # example tests in tests/autograd/test_gradcheck.py cover those two.
    return st.sampled_from(
        [Tensor.tanh, Tensor.exp, squared, total, row_totals, column_averages, flipped, flattened]
    )


@st.composite
def one_input_problems(
    draw: st.DrawFn,
) -> tuple[Callable[[Tensor], Tensor], Tensor, Tensor]:
    """Make a one-input operation, a table to feed it, and a weight for every number of its answer."""
    op = draw(one_input_ops())
    x = draw(tensors(draw(tables()), small_numbers()))
    weights = draw(tensors(op(x).shape, small_numbers()))
    return op, x, weights


# ~~ Models for checkpoints ~~

MAX_ALPHABET = 8
MAX_SIZE = 4
MAX_WEIGHT_VALUES = 5


def model_configs(vocab_size: int) -> st.SearchStrategy[ModelConfig]:
    """Make a small recipe card for a Bigram, MLP or GPT that knows `vocab_size` letters."""
    sizes = st.integers(min_value=1, max_value=MAX_SIZE)
    return st.one_of(
        st.fixed_dictionaries(
            {"kind": st.just("bigram"), "vocab_size": st.just(vocab_size), "block_size": sizes}
        ),
        st.fixed_dictionaries(
            {
                "kind": st.just("mlp"),
                "vocab_size": st.just(vocab_size),
                "block_size": sizes,
                "embed_dim": sizes,
                "hidden_dim": sizes,
            }
        ),
        st.fixed_dictionaries(
            {
                "kind": st.just("gpt"),
                "vocab_size": st.just(vocab_size),
                "block_size": sizes,
                # embed_dim must split evenly into heads (MultiHeadAttention says
                # so), so heads are 1 or 2 and the size is always even.
                "embed_dim": st.sampled_from([2, 4]),
                "num_heads": st.sampled_from([1, 2]),
                "num_layers": st.integers(min_value=1, max_value=2),
                "dropout": st.sampled_from([0.0, 0.25, 0.5]),
            }
        ),
    )


def any_float64() -> st.SearchStrategy[float]:
    """Make any 64-bit float at all: NaN, infinity, -0.0, tiny, huge."""
    return st.floats(width=64)


@st.composite
def tokenizers_and_recipes(draw: st.DrawFn) -> tuple[CharTokenizer | BPETokenizer, ModelConfig]:
    """Make a character or BPE tokenizer, and a small model recipe card that fits its vocabulary."""
    tokenizer: CharTokenizer | BPETokenizer
    if draw(st.booleans()):
        tokenizer = BPETokenizer()
        tokenizer.train(draw(repetitive_texts()), draw(merge_counts()))
    else:
        tokenizer = CharTokenizer(draw(st.text(min_size=1, max_size=MAX_ALPHABET)))
    return tokenizer, draw(model_configs(tokenizer.vocab_size))


def weight_values() -> st.SearchStrategy[list[float]]:
    """Make a few floats of any kind to fill a model's weights with."""
    return st.lists(any_float64(), min_size=1, max_size=MAX_WEIGHT_VALUES)


def filled_model(recipe: ModelConfig, values: list[float]) -> LanguageModel:
    """Build a model from its recipe card, then fill every weight by repeating `values`."""
    model = build_model(recipe)
    for param in model.parameters():
        param.data = np.resize(np.array(values, dtype=np.float64), param.data.shape)
    return model


# ~~ Pebble code for the referee ~~

# Names that are never Pebble keywords, so a program never trips over one.
# Some start like a keyword ("nothing", "letter", "iffy") to keep the referee honest.
NAMES = ("a", "total", "n2", "_x", "nothing", "letter", "iffy")
OPERATORS = ("+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "and", "or")
STRING_BANNED = '"\\\n{}'
MAX_EXPRESSION_LEAVES = 4
MAX_BLOCK_DEPTH = 2
MAX_STATEMENTS = 2
MAX_PROGRAMS = 2
# Every character the canonical slice of Pebble uses, plus a few it doesn't.
PEBBLE_LETTERS = 'abcdefghijklmnopqrstuvwxyz_0123456789 \n(){}"#=+-*/%<>!,é\t'
MAX_NESTING = 1000


def pebble_names() -> st.SearchStrategy[str]:
    """Make a variable name, like `total`."""
    return st.sampled_from(NAMES)


def pebble_strings() -> st.SearchStrategy[str]:
    """Make a quoted string, like `"hi"`, using any letters the referee allows inside one."""
    inside = st.text(alphabet=st.characters(exclude_characters=STRING_BANNED), max_size=5)
    return st.builds('"{}"'.format, inside)


def pebble_atoms() -> st.SearchStrategy[str]:
    """Make the smallest kind of value: a number, a name, a string, true, false or null."""
    return st.one_of(
        st.integers(min_value=0, max_value=999).map(str),
        pebble_names(),
        pebble_strings(),
        st.sampled_from(["true", "false", "null"]),
    )


def in_brackets(expression: str) -> str:
    """Wrap an expression in brackets."""
    return f"({expression})"


def binary(left: str, op: str, right: str) -> str:
    """Join two values with an operator and one space each side."""
    return f"{left} {op} {right}"


def call(name: str, arguments: list[str]) -> str:
    """Call a function, like `f(1, b)`."""
    return f"{name}({', '.join(arguments)})"


def bigger_expressions(smaller: st.SearchStrategy[str]) -> st.SearchStrategy[str]:
    """Make an expression one step bigger than `smaller`: brackets, an operator, not, -, a call."""
    # Only atoms and bracketed expressions sit beside an operator, so
    # `a < b < c` (which the referee calls a foul) is never made.
    operand = st.one_of(pebble_atoms(), st.builds(in_brackets, smaller))
    return st.one_of(
        st.builds(in_brackets, smaller),
        st.builds(binary, operand, st.sampled_from(OPERATORS), operand),
        st.builds("not {}".format, operand),
        st.builds("-{}".format, operand),
        st.builds(call, pebble_names(), st.lists(smaller, max_size=2)),
    )


def pebble_expressions() -> st.SearchStrategy[str]:
    """Make a Pebble expression, like `(a + 1) * f(2)`."""
    return st.recursive(pebble_atoms(), bigger_expressions, max_leaves=MAX_EXPRESSION_LEAVES)


def braced(depth: int, body: list[str]) -> str:
    """Wrap statements in `{ ... }`: each indented one level past `depth`, then `}`."""
    inner = "    " * (depth + 1)
    return " {\n" + inner + ("\n" + inner).join(body) + "\n" + "    " * depth + "}"


def pebble_blocks(depth: int, in_fn: bool) -> st.SearchStrategy[str]:
    """Make a `{ ... }` block: indented statements, then `}` at `depth`'s indent."""
    lines = st.lists(pebble_statements(depth + 1, in_fn), min_size=1, max_size=MAX_STATEMENTS)
    return st.builds(braced, st.just(depth), lines)


def pebble_statements(depth: int, in_fn: bool) -> st.SearchStrategy[str]:
    """Make one statement at `depth` (blocks nest at most MAX_BLOCK_DEPTH deep)."""
    expr = pebble_expressions()
    simple = [
        st.builds("let {} = {}".format, pebble_names(), expr),
        st.builds("{} = {}".format, pebble_names(), expr),
        st.builds("print({})".format, expr),
        st.builds("#{}".format, st.text(alphabet=st.characters(exclude_characters="\n"))),
    ]
    if in_fn:  # `return` only makes sense inside a function
        simple += [st.just("return"), st.builds("return {}".format, expr)]
    if depth >= MAX_BLOCK_DEPTH:
        return st.one_of(simple)
    block = pebble_blocks(depth, in_fn)
    else_ifs = st.lists(st.builds(" else if {}{}".format, expr, block), max_size=1)
    elses = st.one_of(st.just(""), st.builds(" else{}".format, block))
    compound = [
        st.builds("if {}{}{}{}".format, expr, block, st.builds("".join, else_ifs), elses),
        st.builds("while {}{}".format, expr, block),
        st.builds("for {} in range({}){}".format, pebble_names(), expr, block),
    ]
    if depth == 0:  # functions live at the top of a program
        params = st.lists(pebble_names(), max_size=3).map(", ".join)
        fn_block = pebble_blocks(depth, in_fn=True)
        compound.append(st.builds("fn {}({}){}".format, pebble_names(), params, fn_block))
    return st.one_of(simple + compound)


def pebble_programs() -> st.SearchStrategy[str]:
    """Make one tidy Pebble program: statements on their own lines."""
    statements = st.lists(pebble_statements(0, in_fn=False), min_size=1, max_size=MAX_STATEMENTS)
    return statements.map("\n".join)


def pebble_texts() -> st.SearchStrategy[str]:
    """Make a few programs separated by blank lines, like the training corpus."""
    programs = st.lists(pebble_programs(), min_size=1, max_size=MAX_PROGRAMS)
    return programs.map("\n\n".join)


def with_typo(text: str, where: int, cut: int, extra: str) -> str:
    """Cut `cut` letters out of `text` at `where` and type `extra` in their place."""
    where %= len(text) + 1
    return text[:where] + extra + text[where + cut :]


def nearly_pebble() -> st.SearchStrategy[str]:
    """Make a Pebble text with one typo in it, which may or may not still be valid."""
    return st.builds(
        with_typo,
        pebble_texts(),
        st.integers(min_value=0),
        st.integers(min_value=0, max_value=3),
        st.text(alphabet=PEBBLE_LETTERS, max_size=2),
    )


def brackets_nested(depth: int) -> str:
    """Write `let a = ((...(1)...))` with `depth` pairs of brackets."""
    return "let a = " + "(" * depth + "1" + ")" * depth


def model_scribbles() -> st.SearchStrategy[str]:
    """Make the kind of text a model might type: Pebble-ish letters, or brackets nested deep."""
    return st.one_of(
        st.text(alphabet=PEBBLE_LETTERS),
        st.builds(brackets_nested, st.integers(min_value=0, max_value=MAX_NESTING)),
    )


# --- Tokenizers: text -> numbers -> the same text -----------------------------


@given(text_and_message())
def test_char_tokenizer_round_trip(text_and_msg: tuple[str, str]) -> None:
    """Encoding a message and decoding the numbers gives back exactly the same message."""
    text, message = text_and_msg
    tokenizer = CharTokenizer(text)
    assert tokenizer.decode(tokenizer.encode(message)) == message


@given(text_and_message(repetitive_texts()), merge_counts())
def test_bpe_round_trip(text_and_msg: tuple[str, str], merges: int) -> None:
    """BPE may glue letters into chunks, but decoding still gives back exactly the message."""
    text, message = text_and_msg
    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=merges)
    assert tokenizer.decode(tokenizer.encode(message)) == message


@given(text_and_message(repetitive_texts()), merge_counts())
def test_token_numbers_fit_the_vocabulary(text_and_msg: tuple[str, str], merges: int) -> None:
    """Every number BPE hands out is between 0 and vocab_size - 1, so the model can look it up."""
    text, message = text_and_msg
    tokenizer = BPETokenizer()
    tokenizer.train(text, num_merges=merges)
    for number in tokenizer.encode(message):
        assert 0 <= number < tokenizer.vocab_size


# --- Checkpoints: save, load, and get the same model back ---------------------


@given(tokenizer_and_recipe=tokenizers_and_recipes(), weights=weight_values())
def test_checkpoint_round_trip(
    tokenizer_and_recipe: tuple[CharTokenizer | BPETokenizer, ModelConfig], weights: list[float]
) -> None:
    """Saving a model and loading it back gives the same recipe, tokenizer and every weight, bit for bit."""
    tokenizer, recipe = tokenizer_and_recipe
    model = filled_model(recipe, weights)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.npz"
        save(path, model, tokenizer)
        reloaded, reloaded_tokenizer = load(path)
    assert repr(reloaded.config()) == repr(model.config())
    assert type(reloaded_tokenizer) is type(tokenizer)
    assert reloaded_tokenizer.stoi == tokenizer.stoi
    if isinstance(tokenizer, BPETokenizer):
        assert isinstance(reloaded_tokenizer, BPETokenizer)
        assert reloaded_tokenizer.merges == tokenizer.merges
    for before, after in zip(model.parameters(), reloaded.parameters(), strict=True):
        assert after.data.dtype == before.data.dtype
        assert after.data.shape == before.data.shape
        assert after.data.tobytes() == before.data.tobytes()


# --- Autograd: backward() agrees with nudging the input and measuring ---------

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


@given(problem=one_input_problems())
def test_one_input_gradients_match_nudging(
    problem: tuple[Callable[[Tensor], Tensor], Tensor, Tensor],
) -> None:
    """For tanh, exp, squaring, sums, averages and reshapes, backward() matches the measured slope."""
    op, x, weights = problem
    assert_gradients_match(lambda: op(x) * weights, x)


@given(problem=matmul_problems())
def test_matmul_gradients_match_nudging(problem: tuple[Tensor, Tensor, Tensor]) -> None:
    """For `a @ b` on any vectors or tables that fit, batches included, backward() matches the measured slope."""
    a, b, weights = problem
    assert_gradients_match(lambda: (a @ b) * weights, a, b)


# --- The Pebble referee: never wrong about what's finished --------------------

requires_pebble = pytest.mark.skipif(
    not PEBBLE_AVAILABLE, reason="needs pebble-lang: uv sync --all-extras"
)


@requires_pebble
@given(text=st.one_of(pebble_programs(), nearly_pebble()))
def test_whatever_the_referee_passes_pebble_passes(text: str) -> None:
    """Cut a text off anywhere: if the referee calls that much finished, Pebble's parser agrees."""
    # generate() cuts the model's text back to the last point the referee
    # called finished, so every cut-off point matters.
    for end in range(len(text) + 1):
        cut_off = text[:end]
        if complete(cut_off):
            for program in cut_off.split("\n\n"):
                assert is_valid(program), program


@given(text=pebble_programs())
def test_every_prefix_of_a_finished_program_is_viable(text: str) -> None:
    """Tidy Pebble is finished, and the referee never blows the whistle part-way through typing it."""
    assert complete(text)
    for end in range(len(text)):
        assert viable(text[:end]), repr(text[:end])


@given(text=model_scribbles())
def test_the_referee_always_answers(text: str) -> None:
    """Whatever a model types, the referee answers, and never whistles at finished text."""
    # Calling viable() and complete() is the real test: any crash fails it.
    assert viable(text) or not complete(text)
