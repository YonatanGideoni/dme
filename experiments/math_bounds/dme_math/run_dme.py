"""
DME on the AlphaEvolve math bounds problems, sampling from GPT-OSS 20B via Tinker. Each chain's state is its current
program and that program's bound, shown to the model as a hint to improve on. Checkpoints after every step and
resumes from the checkpoint if rerun with the same arguments.
"""
import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dme_math.paths import RUNS_DIR, load_env

load_env()

import numpy as np

from dme.core import log_vocab_size, dme_select
from dme.extract import last_code_block
from dme.marginalise import marginalise as marginalise_span
from dme_math import tinker_utils
from dme_math.eval_utils import evaluate_in_parallel
from dme_math.problems.problem_utils import ProblemLoader
from dme_math.results_utils import RunRecord, load_run, record_from_json, save_run

MODEL_NAME = "openai/gpt-oss-20b"


@dataclass
class ChainState:
    code: str | None = None
    metric_value: float | None = None


def final_channel_bounds(decoded: str) -> tuple[int, int]:
    """[start, end) of GPT-OSS's final answer channel, from the whole completion if there is no final channel marker.
    The end is the first <|end|>/<|return|> after the start."""
    idx = decoded.rfind(tinker_utils.FINAL_CHANNEL_MARKER)
    search_start = idx + len(tinker_utils.FINAL_CHANNEL_MARKER) if idx != -1 else 0
    tail = decoded[search_start:]
    end_candidates = [i for i in (tail.find("<|end|>"), tail.find("<|return|>")) if i != -1]
    return search_start, search_start + min(end_candidates, default=len(tail))


def marginalise(completion: tinker_utils.Completion, tokenizer) -> tuple[float, int]:
    """Path marginalisation: cumulative logprob and token count restricted to the extracted code span.
    Returns (0.0, 0) if no code span was found."""
    if completion.logprobs is None:
        return 0.0, 0
    span = last_code_block(completion.decoded, *final_channel_bounds(completion.decoded)).span
    # Tinker's decoded text keeps special tokens, so token prefixes are decoded the same way, without clamping
    return marginalise_span(completion.tokens, completion.logprobs, tokenizer, span, skip_special_tokens=False)


def build_prompt(base_instruction: str, chain: ChainState, metric_name: str) -> str:
    if chain.code is None:
        return base_instruction
    return (f"{base_instruction}\n\nHere is an example solution from a previous attempt, please improve on it:\n\n"
            f"```\n{chain.code}\n```\n"
            f"This solution got a {metric_name} of {chain.metric_value}.\n\n"
            f"Please now provide your new, better solution.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", default="pack_circ26")
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=20, help="Completions sampled per chain per step")
    p.add_argument("--num-steps", type=int, default=63)
    p.add_argument("--beta", type=float, default=1e6)
    p.add_argument("--lencorr", action=argparse.BooleanOptionalAction, default=False,
                   help="Use the length-respecting uniform prior")
    p.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--num-workers", type=int, default=os.cpu_count() - 4)
    p.add_argument("--output-dir", default=str(RUNS_DIR / "dme"))
    return p.parse_args()


async def main():
    args = parse_args()
    assert os.environ.get("TINKER_API_KEY"), "TINKER_API_KEY not set (see .env.example)"

    problem, evaluator, config = ProblemLoader.load_problem(args.problem)
    metric_name = config.get("metric_name")
    conf_key = config.get("conf_metric_name")
    lower_is_better = config.get("lower_is_better", False)

    sampling_client, tokenizer = await tinker_utils.get_sampling_client(MODEL_NAME)
    log_v = log_vocab_size(tokenizer) if args.lencorr else None

    base_instruction = problem.generate_instruction()

    lencorr_str = "on" if args.lencorr else "off"
    run_name = (f"results_{args.problem}_dme_beta{args.beta:g}_lencorr{lencorr_str}_"
                f"nchains{args.n_chains}_bs{args.batch_size}_{MODEL_NAME.replace('/', '-')}")
    json_path = Path(args.output_dir) / f"{run_name}.json"
    npz_path = Path(args.output_dir) / f"{run_name}.logprobs.npz"
    chainstate_path = Path(args.output_dir) / f"{run_name}.chainstate.json"

    chains = [ChainState() for _ in range(args.n_chains)]
    all_records, algo_id = [], 0
    total_gen_time = total_eval_time = 0.0
    best_reward, best_record = float("-inf"), None
    start_step = 0
    step_size = args.n_chains * args.batch_size

    if json_path.exists():
        prev = load_run(json_path)
        all_records = [record_from_json(d, npz_path) for d in prev["results"]]
        algo_id = len(all_records)
        start_step = algo_id // step_size
        total_gen_time = prev["metadata"].get("generation_time", 0.0)
        total_eval_time = prev["metadata"].get("eval_time", 0.0)
        for r in all_records:
            if r.eval_result is not None and r.eval_result.success:
                metric_value = r.eval_result.metrics[conf_key]
                reward = -metric_value if lower_is_better else metric_value
                if reward > best_reward:
                    best_reward, best_record = reward, r
        if chainstate_path.exists():
            chains = [ChainState(**c) for c in json.loads(chainstate_path.read_text())]
        print(f"Resuming from {json_path}: {len(all_records)} candidates already done "
              f"({start_step}/{args.num_steps} steps complete).", flush=True)

    for step in range(start_step, args.num_steps):
        print(f"=== step {step + 1}/{args.num_steps} ===", flush=True)
        chain_prompts = [build_prompt(base_instruction, c, metric_name) for c in chains]

        t0 = time.time()
        sample_results = await asyncio.gather(*[
            tinker_utils.sample_completions(
                sampling_client, tokenizer, prompt,
                num_samples=args.batch_size, temperature=1.0, max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
            for prompt in chain_prompts
        ])
        completions_by_chain = [completions for completions, _ in sample_results]
        prompt_n_tokens_by_chain = [n for _, n in sample_results]
        gen_time = time.time() - t0
        total_gen_time += gen_time

        t0 = time.time()
        flat_completions = [c for chain in completions_by_chain for c in chain]
        flat_results = evaluate_in_parallel(problem, evaluator, [c.code for c in flat_completions], args.num_workers)
        eval_time = time.time() - t0
        total_eval_time += eval_time

        offset = 0
        for c, (completions, prompt_n_tokens) in enumerate(zip(completions_by_chain, prompt_n_tokens_by_chain)):
            results = flat_results[offset:offset + len(completions)]
            offset += len(completions)

            rewards = np.empty(len(completions))
            logprobs = np.empty(len(completions))
            num_tokens = np.empty(len(completions)) if args.lencorr else None
            for i, (completion, result) in enumerate(zip(completions, results)):
                lp, nt = marginalise(completion, tokenizer)
                logprobs[i] = lp
                if args.lencorr:
                    num_tokens[i] = nt
                if result is not None and result.success:
                    metric_value = result.metrics[conf_key]
                    rewards[i] = -metric_value if lower_is_better else metric_value
                else:
                    rewards[i] = float("-inf")

            # the reward is R = e^bound (sign flipped if lower is better), so log R is the bound itself
            picked = dme_select(rewards, logprobs, args.beta, num_tokens=num_tokens, log_vocab_size=log_v)
            picked_result = results[picked]
            chains[c] = ChainState(
                code=completions[picked].code,
                metric_value=picked_result.metrics.get(conf_key) if picked_result is not None else None,
            )
            print(f"  chain {c}: picked {picked}/{len(completions)}, reward={rewards[picked]:.4g}, "
                  f"success={bool(picked_result and picked_result.success)}", flush=True)

            for completion, result in zip(completions, results):
                record = RunRecord(
                    algorithm_id=algo_id,
                    algorithm_code=completion.code,
                    full_completion=completion.decoded,
                    prompt_used=chain_prompts[c],
                    eval_result=result,
                    token_usage={"input": prompt_n_tokens, "output": len(completion.tokens),
                                 "cached_prompt_tokens": 0, "total": prompt_n_tokens + len(completion.tokens)},
                    api_cost_usd={"total": tinker_utils.estimate_cost(MODEL_NAME, prompt_n_tokens,
                                                                       len(completion.tokens))},
                    tokens=completion.tokens,
                    logprobs=completion.logprobs,
                )
                all_records.append(record)
                if result is not None and result.success:
                    reward = -result.metrics[conf_key] if lower_is_better else result.metrics[conf_key]
                    if reward > best_reward:
                        best_reward, best_record = reward, record
                algo_id += 1

        print(f"  step done: gen={gen_time:.1f}s eval={eval_time:.1f}s", flush=True)

        # Checkpoint after every step, so a crash only loses the step in progress. Chain state
        # isn't recoverable from the saved records alone, so it's persisted in its own small sidecar.
        save_run(args.problem, MODEL_NAME, config, all_records, total_gen_time, total_eval_time,
                 args.output_dir, run_name=run_name)
        chainstate_path.write_text(json.dumps([asdict(c) for c in chains]))

    if best_record is not None:
        print(f"\nBest {metric_name}: {best_record.eval_result.metrics[conf_key]}", flush=True)

    prefill_tokens = sum(r.token_usage["input"] for r in all_records)
    sample_tokens = sum(r.token_usage["output"] for r in all_records)
    cost = tinker_utils.estimate_cost(MODEL_NAME, prefill_tokens, sample_tokens)
    print(f"Total: {len(all_records)} candidates, {args.num_steps} steps, {args.n_chains} chains. "
          f"Estimated Tinker cost: ${cost:.4f} ({prefill_tokens} prefill, {sample_tokens} sample tokens)", flush=True)
    print(f"Wrote results to {json_path}" + (f" and logprobs to {npz_path}" if npz_path.exists() else ""), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
