"""
DME on a RegexEval regex. Also runs the (1,N) hill climbing baseline via --alpha 0 --beta 1e6.

Results are cached per hyperparameter setting under results/; several workers (e.g. a SLURM array) can share a cache
dir, each claims unfinished runs via file locks.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from vllm import LLM

from dme.cache import claimed_run_lock, validate_or_init_cache_config
from dme.extract import strip_whitespace
from dme.vllm_sampler import DME
from dme_regex.consts import DEFAULT_EXAMPLE_ID, DEFAULT_MODEL, MAX_OUT_TOKENS, PROMPT, SAMPLES_PER_RUN
from dme_regex.utils import (
    load_entry,
    RunResult,
    calc_fitness,
    compute_hits_and_stfh,
    print_final_summary,
    regex_accuracy,
    results_base,
    run_result_path,
    save_run_chains,
    save_run_completions,
    save_run_result,
)


def cache_dir_path(args: argparse.Namespace) -> Path:
    model_slug = args.model_name.split("/")[-1]
    path = (
            results_base("REGEX_TASK_RESULTS_DIR")
            / f"id{args.example_id}"
            / f"beta{args.beta}"
            / f"n_chains{args.n_chains}"
            / f"chain_bs{args.batch_size_per_chain}"
            / f"alpha{args.alpha}"
            / f"maxtok{args.max_out_tokens}"
            / f"marginalise_affixes{args.marginalise}"
    )
    if args.length_prior:
        path = path / "lencorr"
    return path / model_slug


def cache_config(args: argparse.Namespace, entry: dict) -> dict:
    # same keys as the original implementation's config.json, so old and new caches can be compared directly
    return {
        "problem": {
            "example_id": args.example_id,
            "target_expression": entry["expression"],
            "samples_per_run": args.samples_per_run,
            "text_postproc": "strip",
        },
        "mcmc": {
            "n_chains": args.n_chains,
            "batch_size_per_chain": args.batch_size_per_chain,
            "beta": args.beta,
            "alpha": args.alpha,
            "marginalise_affixes": args.marginalise,
            "length_correction": args.length_prior,
        },
        "vllm": {
            "model_name": args.model_name,
            "max_out_tokens": args.max_out_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "sampler": {"sampler": "hard"},
    }


def run_worker(args: argparse.Namespace) -> None:
    entry = load_entry(args.example_id)
    pos_examples: list[str] = entry["matches"]
    neg_examples: list[str] = entry["non_matches"]
    # str.replace avoids KeyErrors if the prompt or a regex contains { or }
    prompt = PROMPT.replace("{refined_prompt}", entry["refined_prompt"])

    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else cache_dir_path(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {cache_dir}", flush=True)
    validate_or_init_cache_config(cache_dir, cache_config(args, entry))

    def log_reward(regexes: list[str]) -> np.ndarray:
        return np.log(np.array([calc_fitness(r, pos_examples, neg_examples) for r in regexes]))

    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampler = DME(
        llm,
        extract=strip_whitespace,
        log_reward=log_reward,
        beta=args.beta,
        batch_size=args.batch_size_per_chain,
        n_chains=args.n_chains,
        alpha=args.alpha,
        marginalise=args.marginalise,
        length_prior=args.length_prior,
        include_straddling_start=True,  # a token straddling the strip boundary, e.g. " ^", is part of the regex
        max_tokens=args.max_out_tokens,
    )

    def build_prompt(prev_regex: str) -> str:
        return prompt.replace("{prev_regex}", prev_regex)

    samples_per_step = args.n_chains * args.batch_size_per_chain
    runs = list(range(args.num_runs))
    random.shuffle(runs)

    for r in runs:
        if run_result_path(cache_dir, r).exists():
            continue

        with claimed_run_lock(cache_dir / f"run_{r}.lock") as lock:
            if lock is None:
                continue

            sampler.init_chains()
            completions: list[str] = []
            chains_log: list[list[str]] = []
            total_samples = 0
            start = time.time()

            with torch.inference_mode():
                while total_samples < args.samples_per_run:
                    for candidates in sampler.step(build_prompt):
                        completions.extend(c.answer for c in candidates)
                    chains_log.append(list(sampler.chains))
                    total_samples += samples_per_step

                    if total_samples % (samples_per_step * 50) == 0:
                        chain_accs = [regex_accuracy(s, pos_examples, neg_examples) for s in sampler.chains]
                        print(f"[run {r}] total={total_samples:>8}  mean_chain_acc={np.mean(chain_accs):.4f}",
                              flush=True)

            run_time = time.time() - start
            print(f"[run {r}] done in {run_time:.1f}s", flush=True)

            total_hits, samples_till_first_hit = compute_hits_and_stfh(
                completions, pos_examples, neg_examples, samples_per_step
            )
            avg_fitness = float(np.mean([calc_fitness(s, pos_examples, neg_examples) for s in sampler.chains]))

            save_run_result(cache_dir, RunResult(
                run_id=r,
                hits=total_hits,
                samples=total_samples,
                avg_fitness=avg_fitness,
                samples_till_first_hit=samples_till_first_hit,
                run_time=round(run_time, 2),
            ))
            save_run_completions(cache_dir, r, completions)
            save_run_chains(cache_dir, r, chains_log)

            stfh_str = f"{samples_till_first_hit:.1f}" if samples_till_first_hit is not None else "none"
            print(f"  hits={total_hits}  samples={total_samples}  avg_fitness={avg_fitness:.4f}  stfh={stfh_str}")

    print_final_summary(cache_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example_id", type=int, default=DEFAULT_EXAMPLE_ID, help="s2e-lab/RegexEval example id")
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--max_out_tokens", type=int, default=MAX_OUT_TOKENS)
    p.add_argument("--samples_per_run", type=int, default=SAMPLES_PER_RUN)
    p.add_argument("--n_chains", type=int, default=1, help="Number of parallel DME chains")
    p.add_argument("--batch_size_per_chain", type=int, default=64,
                   help="Completions per chain per step; total batch = n_chains x batch_size_per_chain")
    p.add_argument("--beta", type=float, default=1000.0)
    p.add_argument("--alpha", type=float, default=-1.0,
                   help="Weight on the proposal logprob; -1 for DME, 0 (with --beta 1e6) for (1,N) hill climbing")
    p.add_argument("--marginalise", action=argparse.BooleanOptionalAction, default=True,
                   help="Path marginalisation: use only the logprob of the regex's tokens, not of the surrounding "
                        "whitespace")
    p.add_argument("--length_prior", action=argparse.BooleanOptionalAction, default=False,
                   help="Use the length-respecting uniform prior")
    p.add_argument("--num_runs", type=int, default=10)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return p.parse_args()


if __name__ == "__main__":
    run_worker(parse_args())
