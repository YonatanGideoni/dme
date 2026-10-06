"""
Shared pieces of the examples: the command line arguments and the DME loop. To run DME on your own problem you need
three things, see regex_example.py and coding_example.py:

1. A prompt builder, mapping the chain's current state (the last selected answer) to a prompt.
2. An extractor, mapping a raw completion to the answer, plus the character span it came from (used for path
   marginalisation). `dme.extract` has the two used in the paper: `strip_whitespace` (regex experiments) and
   `last_code_block` (code/math experiments).
3. A batched log-reward function, mapping answers to log R(answer).
"""
from __future__ import annotations

import argparse
import math
from typing import Any, Callable

import numpy as np
from vllm import LLM

from dme.extract import Extraction
from dme.vllm_sampler import DME


def parse_args(
        description: str, default_model: str, default_batch_size: int = 32, include_straddling_start: bool = False,
) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=default_model)
    p.add_argument("--beta", type=float, default=1000)
    p.add_argument("--batch-size", type=int, default=default_batch_size, help="# candidates sampled per step")
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--length-prior", action="store_true", help="Use the length-respecting uniform prior")
    p.add_argument("--no-marginalise", action="store_true", help="No path marginalisation")
    p.add_argument("--include-straddling-start", action=argparse.BooleanOptionalAction, default=include_straddling_start,
                   help="For path marginalisation, count a token holding both text before the answer and its first "
                        "character (e.g. ' ^' for a regex starting with '^') as part of the answer's text")
    return p.parse_args()


def run_dme(
        args: argparse.Namespace,
        extract: Callable[[str], Extraction],
        log_reward: Callable[[list[Any]], np.ndarray],
        build_prompt: Callable[[Any, Any], str],
        max_tokens: int,
        initial_state: Any = "",
) -> None:
    """
    Runs DME for args.num_steps steps and prints the best answer found. build_prompt(state, tokenizer) maps the
    current state to a prompt, the tokenizer is the model's (e.g. for chat templates).
    """
    llm = LLM(model=args.model)
    sampler = DME(
        llm,
        extract=extract,
        log_reward=log_reward,
        beta=args.beta,
        batch_size=args.batch_size,
        marginalise=not args.no_marginalise,
        length_prior=args.length_prior,
        include_straddling_start=args.include_straddling_start,
        max_tokens=max_tokens,
    )
    sampler.init_chains(initial_state)

    best_answer, best_log_reward = None, -math.inf
    for step in range(args.num_steps):
        candidates = sampler.step(lambda state: build_prompt(state, sampler.tokenizer))[0]  # a single chain
        for c in candidates:
            if c.log_reward > best_log_reward:
                best_answer, best_log_reward = c.answer, c.log_reward
        print(f"step {step + 1}: best log R so far {best_log_reward:.3f}, current state {sampler.chains[0]!r}"[:300],
              flush=True)

    print(f"\nBest answer (log R = {best_log_reward:.3f}):\n{best_answer}")
