"""
DME on LiveCodeBench questions. A chain's state is its current program; the first step uses LCB's plain code
generation prompt, later steps LCB's self-repair prompt with the chain's program and its first failing test. A
question run stops at the first hit (a program passing all unit tests) or once the sample budget is used up.

Many question runs are processed concurrently with an async vLLM engine. Results are cached per question and
hyperparameter setting under results/; several workers can share them, each claims unfinished runs via file locks.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lcb_runner.lm_styles import LanguageModelStore, LMStyle
from lcb_runner.prompts.code_generation import format_prompt_generation
from lcb_runner.prompts.self_repair import format_prompt_self_repair
from vllm import SamplingParams

from dme.core import log_vocab_size, dme_select
from dme_lcb.backend import AsyncGenerator, AsyncGrader
from dme_lcb.code_span import code_span_logprob_and_num_tokens, find_code_span
from dme_lcb.grading import load_problems_by_id
from dme_lcb.utils import (
    RESULTS_DIR,
    DMERunResult,
    claim_runs,
    get_base_parser,
    is_run_done,
    log,
    question_set_tag,
    save_list,
    save_result,
    write_campaign_config,
)


def calc_fitness(accuracy: float) -> float:
    """R = exp(accuracy), accuracy being the fraction of unit tests passed."""
    return math.exp(accuracy)


@dataclass
class ChainState:
    code: str = ""
    accuracy: float = 0.0
    metadata: dict = field(default_factory=dict)
    hit: bool = False  # true if any of the chain's proposed candidates was ever a hit (accuracy=1)


@dataclass
class QuestionRun:
    question_id: str
    run_idx: int
    problem: object
    cache_dir: Path
    sample: dict
    total_tests: int
    chains: list[ChainState] = field(default_factory=list)
    n_evalled: int = 0
    completions_log: list[str] = field(default_factory=list)
    accuracies_log: list[float] = field(default_factory=list)  # parallel to completions_log
    chains_log: list[list[str]] = field(default_factory=list)  # per-round selected code per chain
    # rich_cache-only, parallel to completions_log:
    raw_outputs_log: list[str] = field(default_factory=list)
    logprobs_log: list[float] = field(default_factory=list)  # code-span-only (SIR weight input)
    full_logprobs_log: list[float] = field(default_factory=list)  # whole raw completion (cumulative_logprob)
    metadatas_log: list[dict] = field(default_factory=list)


def cache_dir_path(question_id: str, args: argparse.Namespace) -> Path:
    model_slug = args.model.split("/")[-1]
    path = (
            RESULTS_DIR / "dme" / f"qset_{question_set_tag(args.question_ids_file)}"
            / f"id{question_id}"
            / f"beta{args.beta}"
            / f"n_chains{args.n_chains}"
            / f"chain_bs{args.batch_size_per_chain}"
            / f"alpha{args.alpha}"
            / f"maxtok{args.max_out_tokens}"
    )
    if args.length_prior:
        path = path / "lencorr"
    return path / f"samples{args.samples_per_run}" / model_slug


def build_prompt(problem, model_style: LMStyle, state: ChainState) -> str:
    """Plain codegen prompt at init, otherwise the self repair prompt"""
    if not state.code or state.accuracy == 1.0:
        prompt = format_prompt_generation(problem, model_style)
    else:
        metadata_str = json.dumps(state.metadata)
        prompt = format_prompt_self_repair(problem.question_content, model_style, state.code, False, metadata_str)
    assert isinstance(prompt, str), f"expected a raw string prompt for {model_style}, got {type(prompt)}"
    return prompt


def compute_hits_and_stfh(accuracies: list[float], micro_batch_size: int) -> tuple[int, float | None]:
    """
    Scan accuracies in generation order; return (n_hits, samples_till_first_hit). All samples within a micro-batch
    of micro_batch_size = n_chains * batch_size_per_chain are treated as simultaneous; the hit is attributed to the
    midpoint of its micro-batch.
    """
    n_hits = 0
    stfh: float | None = None
    for i, acc in enumerate(accuracies):
        if acc == 1.0:
            n_hits += 1
            if stfh is None:
                batch_idx = i // micro_batch_size
                stfh = (batch_idx + 0.5) * micro_batch_size
    return n_hits, stfh


def build_question_runs(question_ids, problems_by_id, args) -> list[QuestionRun]:
    question_runs = []
    for run_idx in range(args.num_runs):
        for qid in question_ids:
            problem = problems_by_id[qid]
            question_runs.append(QuestionRun(
                question_id=qid,
                run_idx=run_idx,
                problem=problem,
                cache_dir=cache_dir_path(qid, args),
                sample=problem.get_evaluation_sample(),
                total_tests=len(problem.public_test_cases) + len(problem.private_test_cases),
                chains=[ChainState() for _ in range(args.n_chains)],
            ))
    return question_runs


def save_run_logs(q: QuestionRun, args) -> None:
    save_list(q.cache_dir, q.run_idx, "completions", q.completions_log)
    save_list(q.cache_dir, q.run_idx, "chains", q.chains_log)
    if args.rich_cache:
        save_list(q.cache_dir, q.run_idx, "raw_outputs", q.raw_outputs_log)
        save_list(q.cache_dir, q.run_idx, "logprobs", q.logprobs_log)
        save_list(q.cache_dir, q.run_idx, "full_logprobs", q.full_logprobs_log)
        save_list(q.cache_dir, q.run_idx, "accuracies", q.accuracies_log)
        save_list(q.cache_dir, q.run_idx, "metadatas", q.metadatas_log)


def finalize_question_run(q: QuestionRun, start_time: float, args) -> None:
    hits, stfh = compute_hits_and_stfh(q.accuracies_log, args.n_chains * args.batch_size_per_chain)
    avg_fitness = float(sum(calc_fitness(c.accuracy) for c in q.chains) / len(q.chains))
    save_run_logs(q, args)
    save_result(q.cache_dir, q.run_idx, DMERunResult(
        run_id=q.run_idx,
        hits=hits,
        samples=q.n_evalled,
        avg_fitness=avg_fitness,
        samples_till_first_hit=stfh,
        run_time=round(time.perf_counter() - start_time, 2),
    ))


async def round_for_chain(
        chain: ChainState, ci: int, q: QuestionRun, round_idx: int, generator, grader, tokenizer,
        log_v: float | None, args,
) -> None:
    prompt = build_prompt(q.problem, args.model_style, chain)
    sp = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=args.max_out_tokens, n=args.batch_size_per_chain, logprobs=0,
    )
    gen_start = time.perf_counter()
    outputs = await generator.generate(prompt, sp, f"{q.question_id}-run{q.run_idx}-c{ci}-r{round_idx}")
    gen_time = time.perf_counter() - gen_start

    # find_code_span mirrors extract_code's exact fence-picking logic and returns a char span covering the code plus
    # its enclosing ``` fences, used for path marginalisation
    spans = [find_code_span(o.text, args.model_style) for o in outputs]
    codes = [code for code, _span in spans]
    marginals = [code_span_logprob_and_num_tokens(o, span, tokenizer) for o, (_code, span) in zip(outputs, spans)]
    logprobs = [lp for lp, _n in marginals]

    eval_start = time.perf_counter()
    accuracies, metadatas = await grader.grade(q.sample, codes, q.total_tests, args.timeout)
    eval_time = time.perf_counter() - eval_start

    num_tokens = [n for _lp, n in marginals] if args.length_prior else None
    # np.log over an array, as in the original, math.log can differ in the last bit
    log_rewards = np.log(np.array([calc_fitness(a) for a in accuracies]))
    selected = dme_select(log_rewards, logprobs, args.beta, args.alpha, num_tokens, log_v)
    chain.code, chain.accuracy, chain.metadata = codes[selected], accuracies[selected], metadatas[selected]
    chain.hit = chain.hit or max(accuracies) == 1.0

    q.completions_log.extend(codes)
    q.accuracies_log.extend(accuracies)
    if args.rich_cache:
        q.raw_outputs_log.extend(o.text for o in outputs)
        q.logprobs_log.extend(logprobs)
        q.full_logprobs_log.extend(o.cumulative_logprob for o in outputs)
        q.metadatas_log.extend(metadatas)
    log(f"{q.question_id} run{q.run_idx} c{ci} r{round_idx}: gen={gen_time:.2f}s eval={eval_time:.2f}s")


async def run_question(q: QuestionRun, generator, grader, tokenizer, args) -> None:
    start_time = time.perf_counter()
    log_v = log_vocab_size(tokenizer) if args.length_prior else None
    round_idx = 0

    while not any(c.hit for c in q.chains) and q.n_evalled < args.samples_per_run:
        round_idx += 1
        await asyncio.gather(*[
            round_for_chain(chain, ci, q, round_idx, generator, grader, tokenizer, log_v, args)
            for ci, chain in enumerate(q.chains)
        ])
        q.n_evalled += args.n_chains * args.batch_size_per_chain
        q.chains_log.append([c.code for c in q.chains])

    finalize_question_run(q, start_time, args)
    log(
        f"{q.question_id} run{q.run_idx}: done n_evalled={q.n_evalled}  "
        f"mean_chain_acc={sum(c.accuracy for c in q.chains) / len(q.chains):.4f}"
    )


def run_campaign(args) -> None:
    args.model_style = LanguageModelStore[args.model].model_style
    question_ids = args.question_ids_file.read_text().split()
    problems_by_id = load_problems_by_id()
    write_campaign_config(RESULTS_DIR / "dme", args)

    log(f"loading {args.model}...")
    generator = AsyncGenerator(
        args.model, args.gpu_memory_utilization, args.tensor_parallel_size, args.dtype, args.max_out_tokens,
    )
    grader = AsyncGrader(args.num_process_evaluate)
    log("model loaded")

    async def _run():
        tokenizer = await generator.get_tokenizer()
        all_question_runs = build_question_runs(question_ids, problems_by_id, args)
        pending = [q for q in all_question_runs if not is_run_done(q.cache_dir, q.run_idx)]
        log(f"{len(all_question_runs) - len(pending)}/{len(all_question_runs)} already done, "
            f"{len(pending)} remaining")

        for i in range(0, len(pending), args.parallel_slice):
            with claim_runs(pending[i: i + args.parallel_slice]) as claimed:
                if claimed:
                    await asyncio.gather(*[run_question(q, generator, grader, tokenizer, args) for q in claimed])
        log("=== campaign complete ===")

    asyncio.run(_run())
    grader.shutdown()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(parents=[get_base_parser()], description=__doc__)
    p.add_argument("--n_chains", type=int, default=1)
    p.add_argument("--batch_size_per_chain", type=int, default=8)
    p.add_argument("--beta", type=float, default=1000.0)
    p.add_argument("--alpha", type=float, default=-1.0)
    p.add_argument("--samples_per_run", type=int, default=300, help="Sample budget per question run")
    p.add_argument("--length_prior", action=argparse.BooleanOptionalAction, default=False,
                   help="Use the length-respecting uniform prior")
    return p.parse_args()


if __name__ == "__main__":
    run_campaign(parse_args())
