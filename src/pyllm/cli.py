"""The `pyllm` command line: generate, train, tokenize, write Pebble, grow a corpus."""

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from pyllm.checkpoint import load, save
from pyllm.data import PEBBLE_CHECKPOINT, POKEMON_CHECKPOINT, load_corpus
from pyllm.generate import generate
from pyllm.models import GPT, MLP, Bigram, LanguageModel
from pyllm.pebble import PEBBLE_AVAILABLE, build_corpus, parse_rate
from pyllm.pebble import referee as pebble_referee
from pyllm.tokenizer import CharTokenizer
from pyllm.training import train


def _read_corpus(source: str) -> str:
    """Read a corpus arg, which is either a bundled name ('pokemon') or a path to a file."""
    path = Path(source)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return load_corpus(source)


def run_generate(args: argparse.Namespace, out: Callable[[str], None] = print) -> int:
    """Default command: dream up new text from a trained checkpoint."""
    rng = np.random.default_rng(args.seed)
    checkpoint = args.checkpoint or POKEMON_CHECKPOINT
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


def run_train(args: argparse.Namespace, out: Callable[[str], None] = print) -> int:
    """Train a model on a corpus and save a checkpoint."""
    rng = np.random.default_rng(args.seed)
    text = _read_corpus(args.corpus)
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text))
    model: LanguageModel
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


def run_tokenize(args: argparse.Namespace, out: Callable[[str], None] = print) -> int:
    """Show how a piece of text is chopped into token ids (a teaching demo)."""
    text = _read_corpus(args.corpus) if args.corpus else load_corpus("pokemon")
    tokenizer = CharTokenizer(text)
    for char in args.text:
        shown = "\\n" if char == "\n" else char
        try:
            out(f"{shown!r} -> {tokenizer.encode(char)[0]}")
        except KeyError:
            out(f"{shown!r} -> (I have not seen that character yet)")
    return 0


def run_pebble(args: argparse.Namespace, out: Callable[[str], None] = print) -> int:
    """Flagship demo: write Pebble code, and (optionally) grade what parses."""
    rng = np.random.default_rng(args.seed)
    checkpoint = args.checkpoint or PEBBLE_CHECKPOINT
    model, tokenizer = load(checkpoint)
    try:
        text = generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            rng=rng,
            referee=pebble_referee if args.constrained else None,
        )
    except ValueError as exc:
        out(f"(the referee blew the whistle: {exc})")
        return 2
    out(text)
    if args.score:
        if not PEBBLE_AVAILABLE:
            out("(install fromscratch-pebble to score validity: uv sync --all-extras)")
        else:
            programs = [p for p in text.split("\n\n") if p.strip()]
            out(f"parse rate: {parse_rate(programs):.1%} of {len(programs)} programs")
    return 0


def run_gen_corpus(args: argparse.Namespace, out: Callable[[str], None] = print) -> int:
    """Build-time helper: grow a synthetic Pebble corpus to a file."""
    rng = np.random.default_rng(args.seed)
    harvest = [Path(args.harvest)] if args.harvest else []
    for folder in harvest:
        if not folder.is_dir():
            out(f"(no docs folder to harvest at {folder}; check the --harvest path)")
            return 2
    text = build_corpus(
        rng,
        num_generated=args.num_generated,
        harvest_paths=harvest,
        canonical=args.canonical,
    )
    Path(args.out).write_text(text, encoding="utf-8")
    programs = [p for p in text.split("\n\n") if p.strip()]
    out(f"wrote {len(programs)} programs to {args.out}")
    return 0


# Each command's own defaults for the options that can sit on either side of
# the subcommand. They are filled in after parsing, so a value you typed before
# the subcommand (``pyllm --seed 7 pebble``) is never overwritten.
_GENERATE_DEFAULTS: dict[str, object] = {
    "prompt": "",
    "max_new_tokens": 100,
    "temperature": 0.8,
    "top_k": None,
    "checkpoint": None,
    "seed": None,
}
_PEBBLE_DEFAULTS: dict[str, object] = {
    **_GENERATE_DEFAULTS,
    "prompt": "let ",
    "max_new_tokens": 200,
    "temperature": 0.7,
}
_SEED_ONLY_DEFAULTS: dict[str, object] = {"seed": None}
_UNSET = argparse.SUPPRESS


def _add_generation_options(parser: argparse.ArgumentParser) -> None:
    """Add the sampling options shared by plain generation and `pebble`."""
    parser.add_argument("--prompt", default=_UNSET)
    parser.add_argument("--max-new-tokens", type=int, default=_UNSET)
    parser.add_argument("--temperature", type=float, default=_UNSET)
    parser.add_argument("--top-k", type=int, default=_UNSET)
    parser.add_argument("--checkpoint", default=_UNSET)
    parser.add_argument("--seed", type=int, default=_UNSET)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read the command line, with each command's defaults for anything left out.

    Args:
        argv: The arguments (without the program name), or None for ``sys.argv``.

    Returns:
        The parsed options, including ``func``, the command to run.

    """
    parser = argparse.ArgumentParser(prog="pyllm", description="A tiny LLM you can read.")
    parser.set_defaults(func=run_generate, command_defaults=_GENERATE_DEFAULTS)
    _add_generation_options(parser)

    sub = parser.add_subparsers()

    p_train = sub.add_parser("train", help="train your own model")
    p_train.add_argument("--model", choices=["bigram", "mlp", "gpt"], default="gpt")
    p_train.add_argument("--corpus", default="pokemon")
    p_train.add_argument("--steps", type=int, default=1000)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=3e-3)
    p_train.add_argument("--out", default="model.npz")
    p_train.add_argument("--seed", type=int, default=_UNSET)
    p_train.set_defaults(func=run_train, command_defaults=_SEED_ONLY_DEFAULTS)

    p_tok = sub.add_parser("tokenize", help="show how text becomes tokens")
    p_tok.add_argument("--text", required=True)
    p_tok.add_argument("--corpus", default=None)
    p_tok.set_defaults(func=run_tokenize, command_defaults={})

    p_pebble = sub.add_parser("pebble", help="write Pebble code (the flagship demo)")
    _add_generation_options(p_pebble)
    p_pebble.add_argument(
        "--score", action="store_true", help="report what %% of the output parses"
    )
    p_pebble.add_argument(
        "--constrained",
        action="store_true",
        help="let the grammar referee veto illegal characters (always valid code)",
    )
    p_pebble.set_defaults(func=run_pebble, command_defaults=_PEBBLE_DEFAULTS)

    p_corpus = sub.add_parser("gen-corpus", help="grow a synthetic Pebble corpus")
    p_corpus.add_argument("--num-generated", type=int, default=400)
    p_corpus.add_argument(
        "--harvest", default=None, help="directory of *.md docs to mine for real Pebble"
    )
    p_corpus.add_argument(
        "--canonical",
        action="store_true",
        help="run each program through the Pebble formatter",
    )
    p_corpus.add_argument("--out", default="pebble_corpus.txt")
    p_corpus.add_argument("--seed", type=int, default=_UNSET)
    p_corpus.set_defaults(func=run_gen_corpus, command_defaults=_SEED_ONLY_DEFAULTS)

    args = parser.parse_args(argv)
    for name, value in args.command_defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `pyllm`: no subcommand generates; `train`/`tokenize` too."""
    args = parse_args(argv)
    return args.func(args)
