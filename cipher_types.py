"""Cipher variants beyond a fully random 27-symbol permutation.

Keys follow the convention in cipher.py: an int array k where symbol i maps to
k[i]. The space symbol is index 26.
"""

import numpy as np

from alphabet import N_SYMBOLS

SPACE = N_SYMBOLS - 1


def random_key(rng: np.random.Generator, fix_space: bool = False) -> np.ndarray:
    """Random permutation; with fix_space the space maps to itself."""
    if not fix_space:
        return rng.permutation(N_SYMBOLS)
    return np.concatenate([rng.permutation(N_SYMBOLS - 1), [SPACE]])


def atbash_key() -> np.ndarray:
    """A<->Z, B<->Y, ... The space maps to itself."""
    key = np.arange(N_SYMBOLS)
    key[:SPACE] = np.arange(SPACE)[::-1]
    return key


def caesar_key(shift: int) -> np.ndarray:
    """Rotate the 26 letters by `shift`; the space is unchanged."""
    key = np.arange(N_SYMBOLS)
    key[:SPACE] = (np.arange(SPACE) + shift) % SPACE
    return key


def remove_spaces(symbols: np.ndarray) -> np.ndarray:
    """Drop every space symbol from a message."""
    return symbols[symbols != SPACE]


def add_typos(symbols: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Replace a fraction `rate` of letters with random letters (space is left alone)."""
    out = symbols.copy()
    letters = np.flatnonzero(out != SPACE)
    n = int(round(rate * len(letters)))
    chosen = rng.choice(letters, size=n, replace=False)
    out[chosen] = rng.integers(0, SPACE, size=n)
    return out
