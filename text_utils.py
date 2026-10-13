"""Shared text helpers for the Goal 2 and Goal 3 experiments."""

import unicodedata

import numpy as np

from alphabet import N_SYMBOLS, encode, normalize

_SPECIAL = {"ß": "ss", "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE", "ø": "o", "Ø": "O"}


def strip_accents(text: str) -> str:
    """Remove accents so non-English text maps into the 26-letter alphabet (é -> e, ß -> ss)."""
    for old, new in _SPECIAL.items():
        text = text.replace(old, new)
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def clean_text(raw: str) -> str:
    """Strip accents, then reduce to A-Z and single spaces."""
    return normalize(strip_accents(raw))


def letters_only(text: str) -> str:
    """Normalized text with every space removed."""
    return normalize(text).replace(" ", "")


def unigram_probs(text: str) -> np.ndarray:
    """Symbol frequencies of `text` over the 27-symbol alphabet (add-one smoothed)."""
    counts = np.ones(N_SYMBOLS)
    np.add.at(counts, encode(normalize(text)), 1)
    return counts / counts.sum()


def hold_out_tail(text: str, n_test: int):
    """Split off the last `n_test` characters so they are never seen in training."""
    return text[:-n_test], text[-n_test:]
