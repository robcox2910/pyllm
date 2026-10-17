"""Shared type aliases, so every module spells "a grid of numbers" the same way."""

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]
"""A grid of 64-bit decimal numbers (the only kind of array pyllm uses)."""

type IntArray = NDArray[np.int64]
"""A grid of whole numbers, such as token ids."""
