"""The 27-symbol alphabet (A-Z plus space) and helpers for moving text in and out of it.

Internally every symbol is an integer 0..26: 'A'=0, ..., 'Z'=25, ' '=26.
Working with integer arrays lets numpy do the heavy lifting in the sampler.
"""

import re

import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
N_SYMBOLS = len(ALPHABET)  # 27
_INDEX = {ch: i for i, ch in enumerate(ALPHABET)}


def normalize(text: str) -> str:
    """Uppercase, turn every non-letter into a space, and collapse runs of spaces.

    >>> normalize("To be, or not--to be!")
    'TO BE OR NOT TO BE'
    """
    text = text.upper()
    text = re.sub(r"[^A-Z]+", " ", text)  # any run of non-letters -> one space
    return text.strip()


def encode(text: str) -> np.ndarray:
    """Normalized string -> array of symbol indices."""
    return np.array([_INDEX[ch] for ch in text], dtype=np.int64)


def decode(indices) -> str:
    """Array of symbol indices -> string."""
    return "".join(ALPHABET[i] for i in indices)
