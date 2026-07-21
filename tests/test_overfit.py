import numpy as np

from pyllm.models import GPT
from pyllm.tokenizer import CharTokenizer
from pyllm.training import train


def test_gpt_can_overfit_a_tiny_corpus():
    # A GPT with enough steps must drive the loss near zero on a tiny, fixed text.
    text = "pikachu bulbasaur charmander squirtle"
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text))
    rng = np.random.default_rng(0)
    model = GPT(vocab_size=tok.vocab_size, block_size=8, embed_dim=32,
                num_heads=4, num_layers=2, rng=rng)
    losses = train(model, data, steps=400, batch_size=16, lr=1e-2, rng=rng)
    assert losses[-1] < 0.3, f"expected near-zero loss, got {losses[-1]}"
