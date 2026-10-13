"""Bigram and trigram language models over the 27-symbol alphabet.

Both models expose the same interface: `score(decoded)` returns the log
probability of a decoded symbol array, so the sampler does not care which
order it is using.

The trigram model is smoothed toward the bigram model: with N(ab) the count of
the pair (a, b) and N(abc) the count of the triple,

    P(c | a, b) = (N(abc) + k * P2(c | b)) / (N(ab) + k)

so rare contexts fall back to the bigram estimate instead of to zero.
"""

import numpy as np

from alphabet import N_SYMBOLS, encode, normalize


class BigramModel:
    order = 2

    def __init__(self, text: str):
        symbols = encode(normalize(text))
        counts = np.ones((N_SYMBOLS, N_SYMBOLS))  # add-one smoothing
        np.add.at(counts, (symbols[:-1], symbols[1:]), 1)
        self.log_M = np.log(counts / counts.sum(axis=1, keepdims=True))

    def score(self, decoded: np.ndarray) -> float:
        return float(self.log_M[decoded[:-1], decoded[1:]].sum())


class TrigramModel(BigramModel):
    order = 3

    def __init__(self, text: str, k: float = 2.0):
        super().__init__(text)
        symbols = encode(normalize(text))
        counts = np.zeros((N_SYMBOLS, N_SYMBOLS, N_SYMBOLS))
        np.add.at(counts, (symbols[:-2], symbols[1:-1], symbols[2:]), 1)
        base = np.exp(self.log_M)[None, :, :]                       # P2(c | b), broadcast over a
        context = counts.sum(axis=2, keepdims=True)                 # N(ab)
        self.log_T = np.log((counts + k * base) / (context + k))

    def score(self, decoded: np.ndarray) -> float:
        first = self.log_M[decoded[0], decoded[1]]
        rest = self.log_T[decoded[:-2], decoded[1:-1], decoded[2:]].sum()
        return float(first + rest)


def build_model(text: str, order: int):
    """Return a BigramModel (order 2) or TrigramModel (order 3) trained on `text`."""
    if order == 2:
        return BigramModel(text)
    if order == 3:
        return TrigramModel(text)
    raise ValueError(f"order must be 2 or 3, got {order}")
