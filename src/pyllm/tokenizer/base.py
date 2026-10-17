"""The shared shape of every tokenizer."""

from collections.abc import Iterable
from typing import Protocol


class Tokenizer(Protocol):
    """Anything that can turn text into token numbers and back again."""

    stoi: dict[str, int]
    """Maps each token (a string) to its number."""

    @property
    def vocab_size(self) -> int:
        """Return how many different tokens are known."""
        ...

    def encode(self, text: str) -> list[int]:
        """Turn text into a list of token numbers."""
        ...

    def decode(self, ids: Iterable[int]) -> str:
        """Turn token numbers back into text."""
        ...
