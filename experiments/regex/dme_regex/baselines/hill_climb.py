"""(1+N) hill climbing: the prompt's hint is the best regex seen so far, only replaced by a strictly better one."""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
from vllm import LLM, SamplingParams

from dme.cache import claimed_run_lock, validate_or_init_cache_config
from dme_regex.consts import DEFAULT_EXAMPLE_ID, DEFAULT_MODEL, MAX_OUT_TOKENS, PROMPT, SAMPLES_PER_RUN
from dme_regex.utils import (
    load_entry,
    RunResult,
    compute_hits_and_stfh,
    print_final_summary,
    regex_accuracy,
    results_base,
    run_result_path,
    save_run_completions,
    save_run_result,
)


def _output_dir(
        example_id: int, batch_size: int, model_name: str, max_out_tokens: int, out_dir: Path | None
) -> Path:
    if out_dir is not None:
        return out_dir
    model_slug = model_name.split("/")[-1]
    return (
            results_base("REGEX_TASK_BASELINE_RESULTS_DIR") / "hill_climb"
            / f"id{example_id}" / f"bs{batch_size}" / f"maxtok{max_out_tokens}" / model_slug
    )


def _run_one(
        run_id: int,
        prompt_template: str,
        pos_examples: list[str],
        neg_examples: list[str],
        llm: LLM,
        sampling_params: SamplingParams,
        batch_size: int,
        samples_per_run: int,
        cache_dir: Path,
) -> None:
    best_regex = ""
    best_accuracy = 0.0
    completions: list[str] = []
    total_samples = 0
    start = time.perf_counter()

    while total_samples < samples_per_run:
        prompt = prompt_template.replace("{prev_regex}", best_regex)
        results = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        batch = [out.text.strip() for out in results[0].outputs]
        completions.extend(batch)
        total_samples += len(batch)

        for c in batch:
            acc = regex_accuracy(c, pos_examples, neg_examples)
            if acc > best_accuracy:
                best_accuracy = acc
                best_regex = c

    run_time = time.perf_counter() - start

    total_hits, stfh = compute_hits_and_stfh(completions, pos_examples, neg_examples, batch_size)
    avg_fitness = float(np.mean([regex_accuracy(c, pos_examples, neg_examples) for c in completions[-batch_size:]]))

    save_run_result(cache_dir, RunResult(
        run_id=run_id,
        hits=total_hits,
        samples=total_samples,
        avg_fitness=avg_fitness,
        samples_till_first_hit=stfh,
        run_time=round(run_time, 2),
    ))
    save_run_completions(cache_dir, run_id, completions)

    stfh_str = f"{stfh:.1f}" if stfh is not None else "none"
    print(
        f"[run {run_id}] done in {run_time:.1f}s  hits={total_hits}  "
        f"samples={total_samples}  best_acc={best_accuracy:.4f}  stfh={stfh_str}",
        flush=True,
    )


def run_worker(args: argparse.Namespace) -> None:
    entry = load_entry(args.example_id)
    pos_examples: list[str] = entry["matches"]
    neg_examples: list[str] = entry["non_matches"]
    prompt_template = PROMPT.replace("{refined_prompt}", entry["refined_prompt"])

    # same keys as the original implementation's config.json, fields irrelevant to hill climbing get sentinels
    config = {
        "problem": {
            "example_id": args.example_id,
            "target_expression": entry["expression"],
            "samples_per_run": args.samples_per_run,
            "text_postproc": "strip",
        },
        "mcmc": {
            "n_chains": 1, "batch_size_per_chain": args.batch_size, "beta": None, "alpha": -1.0,
            "marginalise_affixes": False, "length_correction": False,
        },
        "vllm": {
            "model_name": args.model_name,
            "max_out_tokens": args.max_out_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
        },
        "sampler": {"sampler": "hill_climb"},
    }

    cache_dir = _output_dir(args.example_id, args.batch_size, args.model_name, args.max_out_tokens, args.out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {cache_dir}", flush=True)
    validate_or_init_cache_config(cache_dir, config)

    llm = LLM(
        model=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        max_tokens=args.max_out_tokens, top_p=1.0, temperature=1.0, n=args.batch_size, logprobs=0,
    )

    runs = list(range(args.num_runs))
    random.shuffle(runs)

    for r in runs:
        if run_result_path(cache_dir, r).exists():
            continue

        with claimed_run_lock(cache_dir / f"run_{r}.lock") as lock:
            if lock is None:
                continue
            _run_one(
                run_id=r,
                prompt_template=prompt_template,
                pos_examples=pos_examples,
                neg_examples=neg_examples,
                llm=llm,
                sampling_params=sampling_params,
                batch_size=args.batch_size,
                samples_per_run=args.samples_per_run,
                cache_dir=cache_dir,
            )

    print_final_summary(cache_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--example_id", type=int, default=DEFAULT_EXAMPLE_ID)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--max_out_tokens", type=int, default=MAX_OUT_TOKENS)
    p.add_argument("--samples_per_run", type=int, default=SAMPLES_PER_RUN)
    p.add_argument("--batch_size", type=int, default=64,
                   help="Completions per vLLM call (all drawn from the current best-regex prompt).")
    p.add_argument("--num_runs", type=int, default=10)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return p.parse_args()


if __name__ == "__main__":
    run_worker(parse_args())
