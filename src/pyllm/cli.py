import argparse
from pathlib import Path

import numpy as np

from pyllm.checkpoint import load, save
from pyllm.data import POKEMON_CHECKPOINT, load_corpus
from pyllm.generate import generate
from pyllm.models import GPT, MLP, Bigram
from pyllm.tokenizer import CharTokenizer
from pyllm.training import train


def _read_corpus(source):
    """A corpus arg is either a bundled name ('pokemon') or a path to a file."""
    path = Path(source)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return load_corpus(source)


def run_generate(args, out=print):
    """Default command: dream up new text from a trained checkpoint."""
    rng = np.random.default_rng(args.seed)
    checkpoint = args.checkpoint if args.checkpoint else POKEMON_CHECKPOINT
    model, tokenizer = load(checkpoint)
    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=rng,
    )
    out(text)
    return 0


def run_train(args, out=print):
    """Train a model on a corpus and save a checkpoint."""
    rng = np.random.default_rng(args.seed)
    text = _read_corpus(args.corpus)
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text))
    if args.model == "bigram":
        model = Bigram(tokenizer.vocab_size, block_size=8, rng=rng)
    elif args.model == "mlp":
        model = MLP(tokenizer.vocab_size, block_size=3, rng=rng)
    else:
        model = GPT(
            tokenizer.vocab_size,
            block_size=16,
            embed_dim=64,
            num_heads=4,
            num_layers=3,
            rng=rng,
        )
    losses = train(
        model,
        data,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        rng=rng,
        log_every=max(1, args.steps // 10),
        log=out,
    )
    save(args.out, model, tokenizer)
    out(f"final loss {losses[-1]:.4f}; saved checkpoint to {args.out}")
    return 0


def run_tokenize(args, out=print):
    """Show how a piece of text is chopped into token ids (a teaching demo)."""
    text = _read_corpus(args.corpus) if args.corpus else load_corpus("pokemon")
    tokenizer = CharTokenizer(text)
    for char in args.text:
        shown = "\\n" if char == "\n" else char
        out(f"{shown!r} -> {tokenizer.encode(char)[0]}")
    return 0


def main(argv=None):
    """Entry point for `pyllm`: no subcommand generates; `train`/`tokenize` too."""
    parser = argparse.ArgumentParser(
        prog="pyllm", description="A tiny LLM you can read."
    )
    parser.set_defaults(func=run_generate)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=None)

    sub = parser.add_subparsers()

    p_train = sub.add_parser("train", help="train your own model")
    p_train.add_argument("--model", choices=["bigram", "mlp", "gpt"], default="gpt")
    p_train.add_argument("--corpus", default="pokemon")
    p_train.add_argument("--steps", type=int, default=1000)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=3e-3)
    p_train.add_argument("--out", default="model.npz")
    p_train.add_argument("--seed", type=int, default=None)
    p_train.set_defaults(func=run_train)

    p_tok = sub.add_parser("tokenize", help="show how text becomes tokens")
    p_tok.add_argument("--text", required=True)
    p_tok.add_argument("--corpus", default=None)
    p_tok.set_defaults(func=run_tokenize)

    args = parser.parse_args(argv)
    return args.func(args)
