"""
Answer extraction helpers. An extractor maps a raw model completion to the answer that is rewarded (and becomes the
chain state), plus the character span of the completion that the answer was generated from. The span is used for
path marginalisation: only the logprob of the tokens covering the span enters the importance weight,
not that of e.g. surrounding whitespace, reasoning, or commentary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CODE_FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class Extraction:
    answer: str | None  # None if nothing could be extracted
    span: tuple[int, int] | None  # [start, end) character span of the raw completion, None if no answer was found


def strip_whitespace(text: str) -> Extraction:
    """
    The answer is the completion with leading/trailing whitespace stripped, as used for the regex experiments. The
    span covers the stripped text, so whitespace before/after the answer does not affect its probability.
    """
    stripped = text.strip()
    if stripped == "":
        return Extraction(answer=stripped, span=None)
    start = text.index(stripped)
    return Extraction(answer=stripped, span=(start, start + len(stripped)))


def last_code_block(text: str, start: int = 0, end: int | None = None) -> Extraction:
    """
    The answer is the (stripped) contents of the last ```/```python fenced code block in text[start:end], as used for
    the math bounds experiments. The span covers the block's contents, excluding the fences. Use start/end to only
    search part of the completion, e.g. the final answer after a reasoning trace.
    """
    end = len(text) if end is None else end
    matches = list(CODE_FENCE_RE.finditer(text[start:end]))
    if not matches:
        return Extraction(answer=None, span=None)
    m = matches[-1]
    return Extraction(answer=m.group(1).strip(), span=(start + m.start(1), start + m.end(1)))
