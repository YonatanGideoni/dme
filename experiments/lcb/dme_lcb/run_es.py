"""
(1,N) and (1+N) evolution strategy baselines on LiveCodeBench. Each round samples N children from the parent via
LCB's self-repair prompt (plain code generation prompt in the first round). (1,N) always replaces the parent with the
best child, (1+N) only if the best child is strictly better. A run stops at the first hit or once the budget is used.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lcb_runner.lm_styles import LanguageModelStore, LMStyle
from lcb_runner.prompts.code_generation import format_prompt_generation
from lcb_runner.prompts.self_repair import format_prompt_self_repair
from lcb_runner.utils.extraction_utils import extract_code
from vllm import SamplingParams

from dme_lcb.backend import AsyncGenerator, AsyncGrader
from dme_lcb.grading import load_problems_by_id
from dme_lcb.utils import (
    RESULTS_DIR,
    ESRunResult,
    claim_runs,
    get_base_parser,
    is_run_done,
    log,
    question_set_tag,
    save_list,
    save_result,
    write_campaign_config,
)

STRATEGIES = ["elitist", "non_elitist"]


@dataclass
class RunState:
    run_id: int
    parent_code: str = ""
    parent_accuracy: float = 0.0
    parent_metadata: dict = field(default_factory=dict)
    best_code: str = ""
    best_accuracy: float = 0.0
    n_evalled: int = 0
    hit: bool = False
    n_evalled_at_hit: int | None = None
    completions_log: list[dict] = field(default_factory=list)


@dataclass
class QuestionRun:
    question_id: str
    strategy: str
    run_idx: int
    problem: object
    cache_dir: Path
    sample: dict
    total_tests: int
    state: RunState
    # rich_cache-only, one entry per (round, candidate), all candidates
    all_candidates_log: list[dict] = field(default_factory=list)


def cache_dir_path(question_id: str, strategy: str, args: argparse.Namespace) -> Path:
    model_slug = args.model.split("/")[-1]
    return (
            RESULTS_DIR / "evo_strategies" / f"qset_{question_set_tag(args.question_ids_file)}"
            / f"id{question_id}" / strategy / f"bs{args.batch_size}" / f"maxtok{args.max_out_tokens}"
            / f"temp{args.temperature}" / f"maxevals{args.max_evals}" / model_slug
    )


def build_repair_prompt(problem, model_style: LMStyle, state: RunState) -> str:
    result = state.parent_accuracy == 1.0
    metadata_str = json.dumps(state.parent_metadata)
    return format_prompt_self_repair(problem.question_content, model_style, state.parent_code, result, metadata_str)


def update_state(state: RunState, strategy: str, codes, accuracies, metadatas) -> None:
    best_idx = int(np.argmax(accuracies))
    best_code, best_acc, best_meta = codes[best_idx], accuracies[best_idx], metadatas[best_idx]

    state.n_evalled += len(codes)
    state.completions_log.append({"round_best_code": best_code, "round_best_accuracy": best_acc})

    if best_acc > state.best_accuracy:
        state.best_accuracy = best_acc
        state.best_code = best_code

    if strategy == "non_elitist" or best_acc > state.parent_accuracy:
        state.parent_code, state.parent_accuracy, state.parent_metadata = best_code, best_acc, best_meta

    if state.best_accuracy == 1.0 and not state.hit:
        state.hit = True
        state.n_evalled_at_hit = state.n_evalled


def build_question_runs(question_ids, problems_by_id, args) -> list[QuestionRun]:
    question_runs = []
    for run_idx in range(args.num_runs):
        for qid in question_ids:
            for strategy in args.strategies:
                problem = problems_by_id[qid]
                question_runs.append(QuestionRun(
                    question_id=qid,
                    strategy=strategy,
                    run_idx=run_idx,
                    problem=problem,
                    cache_dir=cache_dir_path(qid, strategy, args),
                    sample=problem.get_evaluation_sample(),
                    total_tests=len(problem.public_test_cases) + len(problem.private_test_cases),
                    state=RunState(run_id=run_idx),
                ))
    return question_runs


def finalize_question_run(q: QuestionRun, start_time: float, args) -> None:
    save_list(q.cache_dir, q.run_idx, "completions", q.state.completions_log)
    if args.rich_cache:
        save_list(q.cache_dir, q.run_idx, "all_candidates", q.all_candidates_log)
    save_result(q.cache_dir, q.run_idx, ESRunResult(
        run_id=q.run_idx,
        strategy=q.strategy,
        hit=q.state.hit,
        n_evalled=q.state.n_evalled,
        best_accuracy=q.state.best_accuracy,
        n_evalled_at_hit=q.state.n_evalled_at_hit,
        run_time=round(time.perf_counter() - start_time, 2),
    ))


async def run_question(q: QuestionRun, generator, grader, args) -> None:
    start_time = time.perf_counter()
    round_idx = 0
    while not q.state.hit and q.state.n_evalled < args.max_evals:
        round_idx += 1
        prompt = (
            format_prompt_generation(q.problem, args.model_style)
            if not q.state.parent_code
            else build_repair_prompt(q.problem, args.model_style, q.state)
        )
        sp = SamplingParams(
            temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_out_tokens, n=args.batch_size,
        )
        outputs = await generator.generate(prompt, sp, f"{q.question_id}-{q.strategy}-run{q.run_idx}-r{round_idx}")
        codes = [extract_code(o.text, args.model_style) for o in outputs]
        accuracies, metadatas = await grader.grade(q.sample, codes, q.total_tests, args.timeout)

        update_state(q.state, q.strategy, codes, accuracies, metadatas)
        if args.rich_cache:
            q.all_candidates_log.extend(
                {"round": round_idx, "code": c, "accuracy": a, "metadata": m}
                for c, a, m in zip(codes, accuracies, metadatas)
            )

    finalize_question_run(q, start_time, args)
    log(f"{q.question_id} {q.strategy} run{q.run_idx}: "
        f"done hit={q.state.hit} n_evalled={q.state.n_evalled} best={q.state.best_accuracy:.3f}")


def run_campaign(args) -> None:
    args.model_style = LanguageModelStore[args.model].model_style
    question_ids = args.question_ids_file.read_text().split()
    problems_by_id = load_problems_by_id()
    write_campaign_config(RESULTS_DIR / "evo_strategies", args)

    log(f"loading {args.model}...")
    generator = AsyncGenerator(
        args.model, args.gpu_memory_utilization, args.tensor_parallel_size, args.dtype, args.max_out_tokens,
    )
    grader = AsyncGrader(args.num_process_evaluate)
    log("model loaded")

    async def _run():
        all_question_runs = build_question_runs(question_ids, problems_by_id, args)
        pending = [q for q in all_question_runs if not is_run_done(q.cache_dir, q.run_idx)]
        log(f"{len(all_question_runs) - len(pending)}/{len(all_question_runs)} already done, "
            f"{len(pending)} remaining")

        for i in range(0, len(pending), args.parallel_slice):
            with claim_runs(pending[i: i + args.parallel_slice]) as claimed:
                if claimed:
                    await asyncio.gather(*[run_question(q, generator, grader, args) for q in claimed])
        log("=== campaign complete ===")

    asyncio.run(_run())
    grader.shutdown()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(parents=[get_base_parser()], description=__doc__)
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES,
                   help="elitist = (1+N), non_elitist = (1,N)")
    p.add_argument("--batch_size", type=int, default=8, help="N children generated per round")
    p.add_argument("--max_evals", type=int, default=300, help="Sample budget per question run")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    return p.parse_args()


if __name__ == "__main__":
    run_campaign(parse_args())
