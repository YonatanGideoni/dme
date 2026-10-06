"""
Toy example of DME: finding a regex that accepts positive integers
Run from the repo root with any of the experiment envs that has vLLM, e.g.
    uv run --project experiments/regex python examples/regex_example.py
"""
from __future__ import annotations

import math
import re
import signal

import numpy as np

from dme.extract import strip_whitespace
from utils import parse_args, run_dme

PROMPT = (
    "Please output a regular expression that detects the following, with nothing preceding or succeeding it. "
    "Accepts positive integers, e.g. \"34\" and \"1\" but not \"-34\" or \"1.5\".\n\n"
    "The last proposed regex was {state}. The regex is: "
)
POS = ["34", "1", "1000000000", "7"]
NEG = ["-34", "-1", "1.5", "abc", "12a"]


def _timeout_handler(signum, frame):
    raise TimeoutError


def regex_accuracy(pattern: str, timeout: float = 0.1) -> float:
    try:
        r = re.compile(pattern)
    except (re.error, OverflowError):
        return 0.0

    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        correct = sum(bool(r.search(s)) for s in POS) + sum(not r.search(s) for s in NEG)
        result = correct / (len(POS) + len(NEG))
    except TimeoutError:
        result = 0.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    return result


def log_reward(answers: list[str]) -> np.ndarray:
    """log R with R = exp(accuracy)"""
    return np.array([regex_accuracy(a) for a in answers])


def build_prompt(state: str, tokenizer) -> str:
    return PROMPT.replace("{state}", state)


if __name__ == "__main__":
    # to run for longer increase args.num_steps
    run_dme(
        parse_args(__doc__, default_model="Qwen/Qwen3-0.6B", default_batch_size=64, include_straddling_start=True),
        extract=strip_whitespace,
        log_reward=log_reward,
        build_prompt=build_prompt,
        max_tokens=16,
        initial_state="",
    )
