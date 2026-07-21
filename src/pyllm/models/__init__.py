"""Language models on the teaching ladder, assembled from nn building blocks."""

from pyllm.models.bigram import Bigram
from pyllm.models.gpt import GPT
from pyllm.models.mlp import MLP

__all__ = ["Bigram", "MLP", "GPT", "build_model"]


def build_model(config, rng=None):
    """Rebuild a model from its recipe card (`config()` output).

    A saved checkpoint stores only the *numbers* (weights) plus this little
    recipe card. To bring the model back to life we first build the right empty
    shape from the card, then pour the saved numbers in (that second step lives
    in `checkpoint.load`). `kind` picks which model to build.
    """
    kind = config["kind"]
    if kind == "bigram":
        return Bigram(config["vocab_size"], block_size=config["block_size"],
                      rng=rng)
    if kind == "mlp":
        return MLP(config["vocab_size"], block_size=config["block_size"],
                   embed_dim=config["embed_dim"],
                   hidden_dim=config["hidden_dim"], rng=rng)
    if kind == "gpt":
        return GPT(config["vocab_size"], block_size=config["block_size"],
                   embed_dim=config["embed_dim"], num_heads=config["num_heads"],
                   num_layers=config["num_layers"], rng=rng)
    raise ValueError(f"unknown model kind: {kind!r}")
