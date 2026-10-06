"""
The sampling-importance-resampling (SIR) step at the heart of DME (Algorithm 3 in the paper).
"""
from __future__ import annotations

import math

import numpy as np


def log_vocab_size(tokenizer) -> float:
    """log V for the length-respecting prior, V being the tokenizer's vocabulary size (incl. added tokens)."""
    return math.log(len(tokenizer))


def _log_weights(
        log_rewards, logprobs, beta: float, alpha: float = -1.0,
        num_tokens=None, log_vocab_size: float | None = None,
) -> np.ndarray:
    """
    log w_i = beta * log R(x_i) + alpha * log p(x_i|x) [- |x_i| * log V].

    alpha=-1 is DME. alpha=0 with a very large beta recovers (1,N) hill climbing (always pick the best child).
    num_tokens/log_vocab_size are only given when using the length-respecting prior
    """
    logweights = beta * np.asarray(log_rewards) + alpha * np.asarray(logprobs)
    if num_tokens is not None:
        if log_vocab_size is None:
            raise ValueError("log_vocab_size is needed for the length-respecting prior")
        logweights = logweights - np.asarray(num_tokens) * log_vocab_size
    return logweights


def _sample_log_categorical(log_w, rng: np.random.Generator | None = None) -> int:
    """Sample one index with probability proportional to exp(log_w), numerically stable via the Gumbel-max trick."""
    rng = np.random.default_rng() if rng is None else rng
    log_w = np.asarray(log_w)
    gumbels = rng.gumbel(size=log_w.shape)
    return int(np.argmax(log_w + gumbels))


def dme_select(
        log_rewards, logprobs, beta: float, alpha: float = -1.0,
        num_tokens=None, log_vocab_size: float | None = None, rng: np.random.Generator | None = None,
) -> int:
    """
    One DME step: given a batch of candidates sampled conditioned on the chain's current state, returns the index of
    the candidate that becomes the next state.
    """
    logweights = _log_weights(log_rewards, logprobs, beta, alpha, num_tokens, log_vocab_size)
    return _sample_log_categorical(logweights, rng)
