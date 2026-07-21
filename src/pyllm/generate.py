import numpy as np


def sample_next(logits_row, temperature=1.0, top_k=None, rng=None):
    """Pick the next token from the model's scores -- boldly or cautiously.

    The model hands back a score for every possible next character. We turn those
    into a dice with weighted sides and roll it.
    - `temperature` is the boldness dial: 0 always takes the single best guess
      (safe but repetitive); 1 rolls fairly by the model's confidence; higher is
      wilder and more surprising.
    - `top_k` says "only consider the k most likely characters" so the roll never
      lands on something absurd.
    """
    if rng is None:
        rng = np.random.default_rng()
    logits = np.asarray(logits_row, dtype=np.float64)
    if temperature == 0.0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k is not None:
        keep = np.argsort(logits)[-top_k:]
        masked = np.full_like(logits, -np.inf)
        masked[keep] = logits[keep]
        logits = masked
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


def generate(
    model,
    tokenizer,
    prompt="",
    max_new_tokens=100,
    temperature=1.0,
    top_k=None,
    rng=None,
):
    """Dream up new text one character at a time, feeding each guess back in.

    Start with the prompt, ask the model "what comes next?", roll for a character
    (`sample_next`), stick it on the end, and repeat. Each new character becomes
    part of the context for the next guess -- that feedback loop is why a model
    trained on Pokémon names can invent brand-new ones that *feel* like Pokémon.
    We only ever look at the last `block_size` characters (the model's memory).
    """
    if rng is None:
        rng = np.random.default_rng()
    model.eval()
    context = list(tokenizer.encode(prompt))
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
        context.append(
            sample_next(last_row, temperature=temperature, top_k=top_k, rng=rng)
        )
    return tokenizer.decode(context)
