"""
Path marginalisation helpers: restrict a completion's logprob and token count to the tokens that generated the
extracted answer. Characters are mapped to tokens by decoding growing token prefixes, as text can be detokenised in
several ways (e.g. a single token can hold both whitespace and the start of the answer).
"""
from __future__ import annotations

from typing import Callable, Sequence


def prefix_len_fn(
        token_ids: Sequence[int], tokenizer, skip_special_tokens: bool = True, max_len: int | None = None,
) -> Callable[[int], int]:
    """n -> len(decode(token_ids[:n])), optionally capped at max_len (e.g. the completion's text length)."""
    token_ids = list(token_ids)
    cache: dict[int, int] = {}

    def prefix_len(n: int) -> int:
        if n not in cache:
            length = len(tokenizer.decode(token_ids[:n], skip_special_tokens=skip_special_tokens))
            cache[n] = length if max_len is None else min(length, max_len)
        return cache[n]

    return prefix_len


def _bisect_boundary(prefix_len: Callable[[int], int], n_tokens: int, char_target: int) -> int:
    """Smallest n in [0, n_tokens] with prefix_len(n) >= char_target, assuming prefix_len is non-decreasing."""
    lo, hi = 0, n_tokens
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix_len(mid) >= char_target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def token_span(
        prefix_len: Callable[[int], int], n_tokens: int, char_span: tuple[int, int] | None,
        include_straddling_start: bool = False,
) -> tuple[int, int]:
    """
    [t_start, t_end) token range covering char_span, (0, 0) if there is no span or no tokens.

    include_straddling_start=False (code experiments): bisection, a token straddling the span's start is excluded
    while one straddling its end is included.
    include_straddling_start=True (regex experiments): linear scan, any token overlapping the span is included whole,
    e.g. a " ^" token for a regex starting with "^". The span is always at least one token long.
    """
    if char_span is None or n_tokens == 0:
        return 0, 0
    char_start, char_end = char_span

    if not include_straddling_start:
        if char_end <= char_start:
            return 0, 0
        return _bisect_boundary(prefix_len, n_tokens, char_start), _bisect_boundary(prefix_len, n_tokens, char_end)

    # token i covers characters [prefix_len(i), prefix_len(i + 1))
    spans = [(prefix_len(i), prefix_len(i + 1)) for i in range(n_tokens)]
    start_idx = next(i for i, (s, e) in enumerate(spans) if e > char_start)
    end_idx = next((i for i, (s, e) in enumerate(spans) if s >= char_end), len(spans)) - 1
    return start_idx, max(end_idx, start_idx) + 1


def span_logprob(token_logprobs: Sequence[float], t_start: int, t_end: int) -> float:
    total = 0.0
    for i in range(t_start, t_end):
        total += token_logprobs[i]
    return total


def marginalise(
        token_ids: Sequence[int], token_logprobs: Sequence[float], tokenizer, char_span: tuple[int, int] | None,
        include_straddling_start: bool = False, skip_special_tokens: bool = True, max_len: int | None = None,
) -> tuple[float, int]:
    """(logprob, n_tokens) of the tokens covering char_span, (0.0, 0) if there is no span."""
    prefix_len = prefix_len_fn(token_ids, tokenizer, skip_special_tokens, max_len)
    t_start, t_end = token_span(prefix_len, len(token_ids), char_span, include_straddling_start)
    return span_logprob(token_logprobs, t_start, t_end), t_end - t_start
