"""Path marginalisation for LCB: the logprob of only the code block LCB's extract_code would pick, fences included."""
from __future__ import annotations

from lcb_runner.lm_styles import LMStyle

from dme.marginalise import marginalise


def find_code_span(model_output: str, lmstyle: LMStyle) -> tuple[str, tuple[int, int] | None]:
    """
    (code, char_span) where code is exactly what lcb_runner's extract_code returns and char_span covers the code plus
    its enclosing ``` fences, None if there is no code.
    """
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMaInstruct:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
        if len(indexlines) < 2:
            indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2:
            # extract_code falls off the end of the function here (no explicit return -> None) for this style;
            # treat as no-code.
            return "", None
    elif lmstyle == LMStyle.GenericBase:
        code = model_output.strip()
        if not code:
            return "", None
        start_char = model_output.find(code)
        return code, (start_char, start_char + len(code))
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2:
            return "", None

    content_a, content_b = indexlines[-2] + 1, indexlines[-1]
    code = "\n".join(outputlines[content_a:content_b])

    fence_a, fence_b = indexlines[-2], indexlines[-1] + 1  # inclusive of both fence lines
    span_start = sum(len(line) + 1 for line in outputlines[:fence_a])
    span_text = "\n".join(outputlines[fence_a:fence_b])
    return code, (span_start, span_start + len(span_text))


def code_span_logprob_and_num_tokens(output, char_span: tuple[int, int] | None, tokenizer) -> tuple[float, int]:
    """(logprob, n_tokens) of a vLLM CompletionOutput's tokens covering char_span, (0.0, 0) if there is no span."""
    token_ids = list(output.token_ids)
    per_token_logprobs = [output.logprobs[i][tid].logprob for i, tid in enumerate(token_ids)]
    return marginalise(token_ids, per_token_logprobs, tokenizer, char_span, max_len=len(output.text))
