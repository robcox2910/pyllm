"""The shared shape of every language model on the teaching ladder."""

from abc import abstractmethod
from typing import Any, Literal

from pyllm.nn import Module

type ModelConfig = dict[str, Any]
"""The recipe card of a model: its `kind` plus the sizes needed to rebuild it."""


class LanguageModel(Module):
    """A Module that can read a window of tokens and score the next token.

    Every model says how big a window it reads (`block_size`), whether it scores
    every position (`"sequence"`) or just the last one (`"single"`), and can write
    out the recipe card (`config`) needed to rebuild an identical empty model.
    """

    vocab_size: int
    block_size: int
    mode: Literal["sequence", "single"]

    @abstractmethod
    def config(self) -> ModelConfig:
        """Return the recipe card needed to rebuild an identical (untrained) model."""
        raise NotImplementedError
