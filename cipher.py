"""Substitution ciphers over the 27-symbol alphabet.

A key is a permutation stored as an int array `k` of length 27, read as
"symbol i maps to symbol k[i]". The same representation serves both directions:

  * encryption key e:  plaintext symbol -> cipher symbol
  * decoding key f:    cipher symbol    -> plaintext symbol

Applying a key to a whole message is just fancy indexing: key[message].
"""

import difflib

import numpy as np

from alphabet import N_SYMBOLS


def random_key(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random permutation of the 27 symbols (space included)."""
    return rng.permutation(N_SYMBOLS)


def apply_key(key: np.ndarray, symbols: np.ndarray) -> np.ndarray:
    """Substitute every symbol in `symbols` through `key`."""
    return key[symbols]


def invert_key(key: np.ndarray) -> np.ndarray:
    """The inverse permutation: if key maps a -> b, the result maps b -> a."""
    inverse = np.empty_like(key)
    inverse[key] = np.arange(len(key))
    return inverse


def accuracy(decoded: np.ndarray, plaintext: np.ndarray) -> float:
    """Fraction of plaintext characters decoded correctly.

    When the lengths match (e.g. we encrypted the plaintext ourselves) this is a
    position-by-position comparison. A ciphertext supplied from elsewhere may have
    typos or dropped/extra spaces, so otherwise the two texts are aligned first
    and we count characters in matching stretches.
    """
    if len(decoded) == len(plaintext):
        return float(np.mean(decoded == plaintext))
    matcher = difflib.SequenceMatcher(None, decoded.tolist(), plaintext.tolist(), autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks()) / len(plaintext)
