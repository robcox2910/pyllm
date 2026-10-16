import numpy as np


def _probabilities(logits, temperature, top_k):
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


def _anything_goes(token):
    return True


def sample_next(logits_row, temperature=1.0, top_k=None, rng=None, allowed=None):
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
    raise RuntimeError("the referee forbade every possible next token")


def generate(
    model,
    tokenizer,
    prompt="",
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    rng=None,
    referee=None,
):
    """Dream up new text one character at a time, feeding each guess back in.

    Start with the prompt, ask the model "what comes next?", roll for a character
    (`sample_next`), stick it on the end, and repeat. Each new character becomes
    part of the context for the next guess -- that feedback loop is why a model
    trained on Pokémon names can invent brand-new ones that *feel* like Pokémon.
    We only ever look at the last `block_size` characters (the model's memory).

    Pass a `referee` (anything with `viable(text)` and `complete(text)`, such as
    `pyllm.pebble.referee`) and every character must be a legal move. We remember
    the last moment the text was a finished program and, when we run out of
    characters, cut back to it -- so you never get half a program.
    """
    if rng is None:
        rng = np.random.default_rng()
    if referee is not None and not referee.viable(prompt):
        raise ValueError(
            f"the prompt {prompt!r} already breaks the rules, "
            "so it can never grow into valid code"
        )
    model.eval()
    context = list(tokenizer.encode(prompt))
    text = prompt
    finished_at = len(text) if referee is not None and referee.complete(text) else None
    for _ in range(max_new_tokens):
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
        last_row = logits[0, -1] if logits.ndim == 3 else logits[0]
        allowed = None
        if referee is not None:

            def legal(token, so_far=text):
                return referee.viable(so_far + tokenizer.decode([token]))

            allowed = legal
        token = sample_next(
            last_row, temperature=temperature, top_k=top_k, rng=rng, allowed=allowed
        )
        context.append(token)
        text += tokenizer.decode([token])
        if referee is not None and referee.complete(text):
            finished_at = len(text)
    if referee is None:
        return text
    return text[:finished_at] if finished_at is not None else prompt
