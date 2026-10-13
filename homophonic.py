"""Homophonic substitution ciphers: several cipher symbols can stand for one plain symbol.

A decoding "assignment" is an int array `a` of length k (the number of distinct
cipher symbols actually seen), where a[s] is the plain symbol cipher symbol s
stands for. Unlike a permutation, many cipher symbols may share a plain symbol,
so the search moves are:
  * reassign: send one cipher symbol to a different plain symbol
  * swap:     exchange the plain symbols of two cipher symbols
Both proposals are symmetric, so the plain Metropolis acceptance rule applies.

The number of cipher symbols is read directly off the ciphertext
(len(np.unique(ciphertext))); the sampler never needs to know how many
homophones each letter has.
"""

from typing import Optional

import numpy as np

from alphabet import N_SYMBOLS
from sampler_ext import RunResult, Schedule, constant


def allocate_homophones(probs: np.ndarray, n_symbols: int) -> np.ndarray:
    """Number of cipher symbols per plain symbol: at least one each, the rest by frequency."""
    counts = np.ones(len(probs), dtype=int)
    for _ in range(n_symbols - len(probs)):
        counts[np.argmax(probs / (counts + 1))] += 1
    return counts


def encrypt(plaintext: np.ndarray, counts: np.ndarray, rng: np.random.Generator):
    """Replace each plain symbol by a randomly chosen one of its cipher symbols.

    Returns (ciphertext, owner) where owner[c] is the plain symbol of cipher symbol c.
    """
    offsets = np.concatenate([[0], np.cumsum(counts)])
    owner = np.repeat(np.arange(len(counts)), counts)
    ciphertext = offsets[plaintext] + rng.integers(0, counts[plaintext])
    return ciphertext, owner


def relabel(ciphertext: np.ndarray):
    """Renumber the cipher symbols that occur as 0..k-1; return (observed_ids, ciphertext)."""
    observed, relabeled = np.unique(ciphertext, return_inverse=True)
    return observed, relabeled


def initial_assignment(ciphertext, k, probs, rng, perturb=0.3):
    """Match cipher-symbol frequencies to plain-symbol frequencies, then perturb randomly."""
    freq = np.bincount(ciphertext, minlength=k) / len(ciphertext)
    deficit = probs.astype(float).copy()
    assignment = np.empty(k, dtype=np.int64)
    for s in np.argsort(-freq):
        target = int(np.argmax(deficit))
        assignment[s] = target
        deficit[target] -= freq[s]
    n_perturb = int(perturb * k)
    chosen = rng.choice(k, size=n_perturb, replace=False)
    assignment[chosen] = rng.integers(0, N_SYMBOLS, size=n_perturb)
    return assignment


def metropolis(
    ciphertext: np.ndarray,
    k: int,
    model,
    probs: np.ndarray,
    n_iter: int,
    rng: np.random.Generator,
    schedule: Optional[Schedule] = None,
) -> RunResult:
    """Run one chain over many-to-one assignments and return the best one seen."""
    schedule = schedule or constant(1.0)
    assignment = initial_assignment(ciphertext, k, probs, rng)
    score = model.score(assignment[ciphertext])
    best, best_score = assignment.copy(), score
    trace = np.empty(n_iter)
    accepted = 0

    for it in range(n_iter):
        proposal = assignment.copy()
        if rng.random() < 0.5:
            i = rng.integers(k)
            proposal[i] = (proposal[i] + rng.integers(1, N_SYMBOLS)) % N_SYMBOLS
        else:
            i, j = rng.choice(k, size=2, replace=False)
            proposal[i], proposal[j] = assignment[j], assignment[i]
        new_score = model.score(proposal[ciphertext])

        delta = new_score - score
        if delta >= 0 or np.log(rng.random()) < schedule(it, n_iter) * delta:
            assignment, score = proposal, new_score
            accepted += 1
            if score > best_score:
                best, best_score = assignment.copy(), score
        trace[it] = score

    return RunResult(best, best_score, accepted / n_iter, trace)
