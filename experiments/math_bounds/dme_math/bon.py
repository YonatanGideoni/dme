import argparse
import asyncio
import os
import time
from pathlib import Path

from dme_math.paths import RUNS_DIR, load_env

load_env()

from dme_math import tinker_utils
from dme_math.eval_utils import evaluate_in_parallel
from dme_math.problems.problem_utils import ProblemLoader
from dme_math.results_utils import RunRecord, save_run

MODEL_NAME = "openai/gpt-oss-20b"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", default="pack_circ26", help="Problem name, e.g. pack_circ26 (circle packing)")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=32768, help="Generous cap: counts analysis + final channels")
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--num-workers", type=int, default=os.cpu_count() - 4)
    p.add_argument("--output-dir", default=str(RUNS_DIR))
    return p.parse_args()


async def main():
    args = parse_args()
    assert os.environ.get("TINKER_API_KEY"), "TINKER_API_KEY not set (see .env.example)"

    problem, evaluator, config = ProblemLoader.load_problem(args.problem)
    instruction = problem.generate_instruction()

    print(f"Problem: {args.problem} ({config.get('description')})", flush=True)

    cache_path = tinker_utils.gen_cache_path(
        args.output_dir, args.problem, args.num_samples, args.temperature,
        args.max_tokens, args.reasoning_effort, MODEL_NAME,
    )
    used_cache = cache_path.exists()
    if used_cache:
        print(f"Found cached generations at {cache_path}, skipping Tinker sampling (no new cost).", flush=True)
        completions, prompt_n_tokens = tinker_utils.load_gen_cache(cache_path)
        generation_time = 0.0
    else:
        sampling_client, tokenizer = await tinker_utils.get_sampling_client(MODEL_NAME)
        print(f"Sampling {args.num_samples} completions from {MODEL_NAME} "
              f"(temperature={args.temperature}, reasoning_effort={args.reasoning_effort}, "
              f"max_tokens={args.max_tokens})...", flush=True)
        t0 = time.time()
        completions, prompt_n_tokens = await tinker_utils.sample_completions(
            sampling_client, tokenizer, instruction,
            num_samples=args.num_samples, temperature=args.temperature,
            max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
        )
        generation_time = time.time() - t0
        tinker_utils.save_gen_cache(cache_path, completions, prompt_n_tokens)

    n_no_code = sum(1 for c in completions if c.code is None)
    print(f"Generation done in {generation_time:.1f}s. {n_no_code}/{len(completions)} completions had no "
          f"extractable code block.", flush=True)

    print(f"Evaluating {len(completions)} candidates across {args.num_workers} worker processes "
          f"(each with its own {problem.get_max_execution_time():.0f}s sandbox timeout)...", flush=True)
    t0 = time.time()
    codes = [c.code for c in completions]
    results = evaluate_in_parallel(problem, evaluator, codes, args.num_workers)
    eval_time = time.time() - t0

    successes = [r for r in results if r is not None and r.success]
    print(f"Evaluation done in {eval_time:.1f}s. {len(successes)}/{len(results)} succeeded.", flush=True)

    if successes:
        metric_name = config.get("metric_name")
        conf_key = config.get("conf_metric_name")
        lower_is_better = config.get("lower_is_better", False)
        best = (min if lower_is_better else max)(successes, key=lambda r: r.metrics[conf_key])
        print(f"Best {metric_name}: {best.metrics[conf_key]}", flush=True)

    prefill_tokens = prompt_n_tokens * args.num_samples
    sample_tokens = sum(len(c.tokens) for c in completions)
    cost = tinker_utils.estimate_cost(MODEL_NAME, prefill_tokens, sample_tokens)
    cost_note = "already paid for, loaded from cache, no new cost this run" if used_cache else "this run"
    print(f"Estimated Tinker cost ({cost_note}): ${cost:.4f} "
          f"({prefill_tokens} prefill tokens, {sample_tokens} sample tokens)", flush=True)

    records = [
        RunRecord(
            algorithm_id=i,
            algorithm_code=c.code,
            full_completion=c.decoded,
            prompt_used=instruction,
            eval_result=result,
            token_usage={"input": prompt_n_tokens, "output": len(c.tokens), "cached_prompt_tokens": 0,
                         "total": prompt_n_tokens + len(c.tokens)},
            api_cost_usd={"total": tinker_utils.estimate_cost(MODEL_NAME, prompt_n_tokens, len(c.tokens))},
            tokens=c.tokens,
            logprobs=c.logprobs,
        )
        for i, (c, result) in enumerate(zip(completions, results))
    ]
    json_path, npz_path = save_run(
        args.problem, MODEL_NAME, config, records, generation_time, eval_time, args.output_dir,
    )
    print(f"Wrote results to {json_path}" + (f" and logprobs to {npz_path}" if npz_path else ""), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
