"""Generation: dreaming up new text one character at a time."""

from collections.abc import Callable
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike

from pyllm._types import Array
from pyllm.models import LanguageModel
from pyllm.tokenizer import Tokenizer

_SEQUENCE_DIMS = 3
"""Sequence models return (batch, position, vocab) scores: three dimensions."""


class Referee(Protocol):
    """A rule-checker that can judge half-written text (see `pyllm.pebble.referee`)."""

    def viable(self, text: str) -> bool:
        """Say whether `text` could still grow into something valid."""
        ...

    def complete(self, text: str) -> bool:
        """Say whether `text` is finished and valid right now."""
        ...


def _probabilities(logits: Array, temperature: float, top_k: int | None) -> Array:
    """Turn raw scores into dice weights (temperature + optional top-k shortlist)."""
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        keep = np.argsort(logits)[-top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def _anything_goes(token: int) -> bool:  # noqa: ARG001
    """Allow every token (the default when there is no referee)."""
    return True


def sample_next(
    logits_row: ArrayLike,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
    allowed: Callable[[int], bool] | None = None,
) -> int:
    """Pick the next token from the model's scores -- boldly or cautiously.

    The model hands back a score for every possible next character. We turn those
    into a dice with weighted sides and roll it.
    - `temperature` is the boldness dial: 0 always takes the single best guess
      (safe but repetitive); 1 rolls fairly by the model's confidence; higher is
      wilder and more surprising.
    - `top_k` says "only consider the k most likely characters" so the roll never
      lands on something absurd. `None` or any value <= 0 means "no shortlist --
      every character stays in the running".
    - `allowed` is an optional referee: a function that says whether a token is a
      legal move. If the dice land on an illegal token we cross it out and roll
      again, so only legal tokens can ever come out. (If the top-k shortlist runs
      out of legal tokens, we widen the search to every token.)
    """
    if rng is None:
        rng = np.random.default_rng()
    if allowed is None:
        allowed = _anything_goes
    logits = np.asarray(logits_row, dtype=np.float64)
    if temperature != 0.0:
        shortlists = [top_k, None] if top_k is not None and top_k > 0 else [None]
        for shortlist in shortlists:
            probs = _probabilities(logits, temperature, shortlist)
            while probs.sum() > 0:
                pick = int(rng.choice(len(probs), p=probs))
                if allowed(pick):
                    return pick
                probs[pick] = 0.0  # cross out the illegal move...
                if probs.sum() > 0:
                    probs = probs / probs.sum()  # ...and re-balance the dice
    # Greedy (or every roll was crossed out): the best-scoring legal token.
    for token in np.argsort(-logits, kind="stable"):
        if allowed(int(token)):
            return int(token)
    msg = "the referee forbade every possible next token"
    raise RuntimeError(msg)


def _next_scores(model: LanguageModel, context: list[int]) -> Array:
    """Ask the model for a score for every possible next token, given the context so far."""
    window = context[-model.block_size :]
    if model.mode == "single":
        # MLP needs exactly block_size tokens; left-pad short windows.
        if len(window) < model.block_size:
            window = [0] * (model.block_size - len(window)) + window
    elif not window:
        # Sequence models with an empty prompt need a seed token to read.
        window = [0]
    x = np.array([window], dtype=np.int64)
    logits = model(x).data
    return logits[0, -1] if logits.ndim == _SEQUENCE_DIMS else logits[0]


def _legal_moves(referee: Referee, tokenizer: Tokenizer, so_far: str) -> Callable[[int], bool]:
    """Make the referee's yes/no check for each possible next token after `so_far`."""

    def legal(token: int) -> bool:
        return referee.viable(so_far + tokenizer.decode([token]))

    return legal


def _write_one_try(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    rng: np.random.Generator,
    referee: Referee,
) -> str | None:
    """Write past the prompt under the referee; the text up to the last new finish, or None.

    Only a finish *after* the prompt counts: the prompt on its own (say, a lone
    `# comment` line) is not an answer. If the budget runs out before anything
    new has finished -- usually mid-statement, or inside an unclosed string or
    block -- we play overtime: up to `max_new_tokens` more characters, picked
    greedily (the model's most confident legal guess), stopping at the first
    finished point.
    """
    context = list(tokenizer.encode(prompt))
    text = prompt
    finished_at: int | None = None
    for step in range(2 * max_new_tokens):
        in_overtime = step >= max_new_tokens
        if in_overtime and finished_at is not None:
            break
        token = sample_next(
            _next_scores(model, context),
            temperature=0.0 if in_overtime else temperature,
            top_k=top_k,
            rng=rng,
            allowed=_legal_moves(referee, tokenizer, text),
        )
        context.append(token)
        text += tokenizer.decode([token])
        if referee.complete(text):
            finished_at = len(text)
    return text[:finished_at] if finished_at is not None else None


_TRIES = 4
"""How many fresh starts a refereed generation gets before giving up on new code."""


def generate(
    model: LanguageModel,
    tokenizer: Tokenizer,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: np.random.Generator | None = None,
    referee: Referee | None = None,
) -> str:
    """Dream up new text one character at a time, feeding each guess back in.

    Start with the prompt, ask the model "what comes next?", roll for a character
    (`sample_next`), stick it on the end, and repeat. Each new character becomes
    part of the context for the next guess -- that feedback loop is why a model
    trained on Pokémon names can invent brand-new ones that *feel* like Pokémon.
    We only ever look at the last `block_size` characters (the model's memory).

    Pass a `referee` (anything with `viable(text)` and `complete(text)`, such as
    `pyllm.pebble.referee`) and every character must be a legal move. We remember
    the last moment the text was a finished program and, when we run out of
    characters, cut back to it -- so you never get half a program. A finished
    prompt doesn't count: if nothing new has finished when the budget runs out,
    we play overtime (see `_write_one_try`) and, failing that, start again from
    the prompt, up to `_TRIES` times. Only if every try fails do we hand back the
    prompt (when it is itself finished) or nothing.
    """
    if rng is None:
        rng = np.random.default_rng()
    if referee is not None and not referee.viable(prompt):
        msg = (
            f"the prompt {prompt!r} already breaks the rules, so it can never grow into valid code"
        )
        raise ValueError(msg)
    model.eval()
    if referee is not None:
        for _ in range(_TRIES):
            finished = _write_one_try(
                model, tokenizer, prompt, max_new_tokens, temperature, top_k, rng, referee
            )
            if finished is not None:
                return finished
        # Nothing new ever finished: never hand back half a program.
        return prompt if referee.complete(prompt) else ""
    context = list(tokenizer.encode(prompt))
    text = prompt
    for _ in range(max_new_tokens):
        token = sample_next(
            _next_scores(model, context), temperature=temperature, top_k=top_k, rng=rng
        )
        context.append(token)
        text += tokenizer.decode([token])
    return text
