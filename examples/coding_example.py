"""
Finding a Python function using DME.
Run from the repo root with any of the experiment envs that has vLLM, e.g.
    uv run --project experiments/regex python examples/coding_example.py --length-prior
"""
from __future__ import annotations

import math
import subprocess
import sys

import numpy as np

from dme.extract import last_code_block
from utils import parse_args, run_dme

TASK = "Write a Python function `second_largest(xs)` returning the second largest distinct value of a list of " \
       "integers, or None if there is none. Put the function in a single ```python code block."
TESTS = [([3, 1, 2], 2), ([5, 5, 4], 4), ([1], None), ([], None), ([-1, -2, -2], -2), ([7, 7], None)]


def log_reward(answers: list[str | None]) -> np.ndarray:
    """log R with R = exp(fraction of tests passed). Runs each program in a subprocess with a timeout; this is NOT a
    sandbox, run model-generated code in an isolated environment."""
    log_rewards = []
    for code in answers:
        if code is None:
            log_rewards.append(-math.inf)  # no code block, never selected unless every candidate is invalid
            continue
        test_script = code + "\n\n" + "\n".join(
            f"try:\n    print(second_largest({xs!r}) == {want!r})\nexcept Exception:\n    print(False)"
            for xs, want in TESTS
        )
        try:
            out = subprocess.run([sys.executable, "-c", test_script], capture_output=True, text=True, timeout=5)
            passed = out.stdout.split().count("True")
        except subprocess.TimeoutExpired:
            passed = 0
        log_rewards.append(passed / len(TESTS))
    return np.array(log_rewards, dtype=float)


def build_prompt(state: str | None, tokenizer) -> str:
    task = TASK
    if state:
        task += f"\n\nHere is a previous attempt, please improve on it:\n```python\n{state}\n```"
    messages = [{"role": "user", "content": task}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


if __name__ == "__main__":
    # to run for longer increase args.num_steps
    run_dme(
        parse_args(__doc__, default_model="Qwen/Qwen2.5-Coder-1.5B-Instruct"),
        extract=last_code_block,
        log_reward=log_reward,
        build_prompt=build_prompt,
        max_tokens=512,
        initial_state=None,
    )
