import argparse
import asyncio
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from dme_math.paths import RUNS_DIR, load_env

load_env()

from dme_math import tinker_utils
from dme_math.eval_utils import evaluate_in_parallel
from dme_math.problems.problem_utils import ProblemLoader
from dme_math.results_utils import RunRecord, load_run, record_from_json, save_run

MODEL_NAME = "openai/gpt-oss-20b"


@dataclass
class ArchiveEntry:
    code: str
    metric_value: float


def build_prompt(base_instruction: str, archive: list[ArchiveEntry], metric_name: str) -> str:
    if not archive:
        return base_instruction
    instr = base_instruction
    instr += "\nHere are some example solutions from previous attempts, please improve on them:"
    for i, entry in enumerate(archive):
        instr += f"\n\n# Solution {i + 1}:\n```\n{entry.code}\n```\n"
        instr += f"This solution, solution {i + 1}, got a {metric_name} of {entry.metric_value}.\n"
    instr += "\nPlease now provide your new, better solution."
    return instr


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", default="pack_circ26")
    p.add_argument("--num-algorithms", type=int, default=20, help="Completions sampled per generation")
    p.add_argument("--n-gens", type=int, default=10)
    p.add_argument("--k", type=int, default=3, help="Number of archive examples shown per prompt (random selection)")
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--num-workers", type=int, default=os.cpu_count() - 4)
    p.add_argument("--trial", type=int, default=1, help="Trial index, each trial is an independent SCS run")
    p.add_argument("--output-dir", default=None, help="Defaults to runs/scs/trial<trial>")
    return p.parse_args()


async def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = str(RUNS_DIR / "scs" / f"trial{args.trial}")
    assert os.environ.get("TINKER_API_KEY"), "TINKER_API_KEY not set (see .env.example)"

    problem, evaluator, config = ProblemLoader.load_problem(args.problem)
    metric_name = config.get("metric_name")
    conf_key = config.get("conf_metric_name")
    lower_is_better = config.get("lower_is_better", False)

    sampling_client, tokenizer = await tinker_utils.get_sampling_client(MODEL_NAME)

    base_instruction = problem.generate_instruction()

    run_name = (f"results_{args.problem}_scs_numalgs{args.num_algorithms}_ngens{args.n_gens}_k{args.k}_"
                f"{MODEL_NAME.replace('/', '-')}")
    json_path = Path(args.output_dir) / f"{run_name}.json"
    npz_path = Path(args.output_dir) / f"{run_name}.logprobs.npz"

    archive: list[ArchiveEntry] = []  # only the previous generation's successes
    all_records, algo_id = [], 0
    total_gen_time = total_eval_time = 0.0
    best_reward, best_record = float("-inf"), None
    start_gen = 0

    if json_path.exists():
        prev = load_run(json_path)
        all_records = [record_from_json(d, npz_path) for d in prev["results"]]
        algo_id = len(all_records)
        start_gen = algo_id // args.num_algorithms
        total_gen_time = prev["metadata"].get("generation_time", 0.0)
        total_eval_time = prev["metadata"].get("eval_time", 0.0)
        for r in all_records:
            if r.eval_result is not None and r.eval_result.success:
                metric_value = r.eval_result.metrics[conf_key]
                reward = -metric_value if lower_is_better else metric_value
                if reward > best_reward:
                    best_reward, best_record = reward, r
        if start_gen > 0:
            last_gen_records = all_records[(start_gen - 1) * args.num_algorithms: start_gen * args.num_algorithms]
            archive = [ArchiveEntry(code=r.algorithm_code, metric_value=r.eval_result.metrics[conf_key])
                       for r in last_gen_records if r.eval_result is not None and r.eval_result.success]
        print(f"Resuming from {json_path}: {len(all_records)} candidates already done "
              f"({start_gen}/{args.n_gens} generations complete).")

    for gen in range(start_gen, args.n_gens):
        print(f"=== gen {gen + 1}/{args.n_gens} ===")

        # build one prompt per program, boosts diversity
        prompts = [
            build_prompt(base_instruction, random.sample(archive, min(args.k, len(archive))) if archive else [],
                         metric_name)
            for _ in range(args.num_algorithms)
        ]

        t0 = time.time()
        sample_results = await asyncio.gather(*[
            tinker_utils.sample_completions(
                sampling_client, tokenizer, prompt,
                num_samples=1, temperature=1.0, max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            for prompt in prompts
        ])
        completions = [completions_for_prompt[0] for completions_for_prompt, _ in sample_results]
        prompt_n_tokens_list = [n for _, n in sample_results]
        gen_time = time.time() - t0
        total_gen_time += gen_time

        t0 = time.time()
        results = evaluate_in_parallel(problem, evaluator, [c.code for c in completions], args.num_workers)
        eval_time = time.time() - t0
        total_eval_time += eval_time

        new_archive = []
        n_success = 0
        for completion, result, prompt, prompt_n_tokens in zip(completions, results, prompts, prompt_n_tokens_list):
            record = RunRecord(
                algorithm_id=algo_id,
                algorithm_code=completion.code,
                full_completion=completion.decoded,
                prompt_used=prompt,
                eval_result=result,
                token_usage={"input": prompt_n_tokens, "output": len(completion.tokens),
                             "cached_prompt_tokens": 0, "total": prompt_n_tokens + len(completion.tokens)},
                api_cost_usd={"total": tinker_utils.estimate_cost(MODEL_NAME, prompt_n_tokens,
                                                                   len(completion.tokens))},
                tokens=completion.tokens,
                logprobs=completion.logprobs,
            )
            all_records.append(record)
            algo_id += 1

            if result is not None and result.success:
                n_success += 1
                metric_value = result.metrics[conf_key]
                new_archive.append(ArchiveEntry(code=completion.code, metric_value=metric_value))
                reward = -metric_value if lower_is_better else metric_value
                if reward > best_reward:
                    best_reward, best_record = reward, record

        archive = new_archive

        print(f"  gen done: {n_success}/{len(completions)} succeeded, gen={gen_time:.1f}s eval={eval_time:.1f}s")

        # Checkpoint after every generation, so a crash only loses the generation in
        # progress, not the whole trial
        save_run(args.problem, MODEL_NAME, config, all_records, total_gen_time, total_eval_time,
                 args.output_dir, run_name=run_name)

    if best_record is not None:
        print(f"\nBest {metric_name}: {best_record.eval_result.metrics[conf_key]}")

    prefill_tokens = sum(r.token_usage["input"] for r in all_records)
    sample_tokens = sum(r.token_usage["output"] for r in all_records)
    cost = tinker_utils.estimate_cost(MODEL_NAME, prefill_tokens, sample_tokens)
    print(f"Total: {len(all_records)} candidates, {args.n_gens} gens x {args.num_algorithms} algs. "
          f"Estimated Tinker cost: ${cost:.4f} ({prefill_tokens} prefill, {sample_tokens} sample tokens)")
    print(f"Wrote results to {json_path}" + (f" and logprobs to {npz_path}" if npz_path.exists() else ""))


if __name__ == "__main__":
    asyncio.run(main())
