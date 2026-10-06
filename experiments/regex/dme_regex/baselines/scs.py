from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from dme_regex.consts import MAX_OUT_TOKENS, SAMPLES_PER_RUN
from dme_regex.utils import regex_accuracy, results_base, save_run_completions

# Seed prompt: no {prev_regex} hint, SCS optimizes the prompt itself.
SEED_PROMPT = (
    "Please output a regular expression that detects the following, "
    "with nothing preceding or succeeding it. {refined_prompt}\n\nThe regex is: "
)

DEFAULT_EXAMPLE_ID = 2447
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

# copied almost verbatim from gepa_baseline.py's REFLECTION_PROMPT_TEMPLATE
REFLECTION_HEADER = """You are optimizing the exact prompt text fed to a small language model so that it outputs a regular expression matching all (hidden) positive examples and rejecting all negative examples.

## Critical structural fact

The prompt you produce is concatenated character-for-character directly in front of the eval model's generation. There is no chat template, no separator, and no "user turn" between your text and the model's output — the model's completion begins at the exact character where your text ends. For example, if your prompt ends in "...The regex is: " and the model then generates "^abc$", the model literally saw "...The regex is: " and continued it with "^abc$".

## What you must produce — and must NOT produce

Your output must always be a natural-language INSTRUCTION/PROMPT that elicits a regex from the eval model — never a bare regex, never the eval model's own completion, and never just a corrected version of the best completion from the previous attempts below. A bare regex is not a valid replacement for the prompt: it removes the instruction context the eval model needs and will not improve the score. If you find yourself about to output something that itself compiles as a regex with no surrounding instruction, stop — that is the failure mode this warning exists to prevent.

## Task description

The task description that must stay intact in your new prompt:

```
{base_eval_prompt}
```"""

REFLECTION_TASK_FOOTER = """## Your Task

Analyze the previous attempts and their scores above:
- **Failure patterns**: What specific errors or failure modes appear (e.g. the eval model rambling instead of emitting a regex immediately, wrong output format, truncation)?
- **Success patterns**: What phrasing led to correct or close regexes, and should be preserved?
- **Root causes**: What about the current instruction confuses the eval model?

Based on your analysis, propose an improved instruction/prompt that:
1. Addresses the identified failure patterns and root causes
2. Preserves successful phrasing from previous attempts
3. Makes the eval model as likely as possible to emit, immediately and with nothing else, a regex matching the task description in the original seed prompt (do not change or narrow the described task itself — only how the instruction is phrased)

## Output Format

Provide ONLY the improved instruction/prompt within ``` blocks. It must be a complete, drop-in replacement for the eval prompt — natural-language instruction text, not a regex."""


@dataclass
class EvalPromptResult:
    generation: int
    prompt: str
    score: float
    hits: int
    samples: int
    best_completion: str
    best_completion_score: float
    completions: list[str] = dataclasses.field(default_factory=list)


@dataclass
class SCSRunStats:
    hits: int = 0
    samples: int = 0
    samples_till_first_hit: int | None = None


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _output_dir(example_id: int, model_name: str, max_out_tokens: int, out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir
    model_slug = model_name.split("/")[-1]
    base = results_base("REGEX_TASK_BASELINE_RESULTS_DIR")
    return (
            base / "scs"
            / f"id{example_id}" / f"maxtok{max_out_tokens}" / model_slug
    )


class PromptGenerator:
    def __init__(
            self,
            model_name: str,
            num_prompts: int,
            max_tokens: int,
            max_model_len: int,
            tensor_parallel_size: int,
            gpu_memory_utilization: float,
            temperature: float,
            top_p: float,
            think_budget: int,
    ) -> None:
        self.think_budget = think_budget
        self.answer_budget = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # force-close tokens appended after a (possibly truncated) thinking phase
        self._think_close_ids = self.tokenizer.encode("</think>\n\n", add_special_tokens=False)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

        self.num_prompts = num_prompts

    def generate_eval_prompts(
            self,
            base_eval_prompt: str,
            previous_generation: list[EvalPromptResult],
            n_examples_from_archive: int,
            # as model might generate bad outputs+we want good prompts, prompt generation doesn't count towards budget
            leeway_extra: int = 5
    ) -> list[str]:
        if not previous_generation:
            meta_prompts = [self._build_meta_prompt(
                base_eval_prompt=base_eval_prompt,
                n_examples_from_archive=n_examples_from_archive,
            )]
            n_per_prompt = self.num_prompts + leeway_extra
        else:
            meta_prompts = [
                self._build_meta_prompt(
                    base_eval_prompt=base_eval_prompt,
                    previous_generation=previous_generation,
                    n_examples_from_archive=n_examples_from_archive
                )
                for _ in range(self.num_prompts + leeway_extra)]
            n_per_prompt = 1

        completions = self._generate_meta_completions(meta_prompts, n_per_prompt)

        prompts, extr_successful = zip(*[self._extract_prompt(text, base_eval_prompt)
                                         for text in completions])
        succ_extracted_prompts = [p for p, success in zip(prompts, extr_successful) if success]
        unsucc_extracted_prompts = [p for p, success in zip(prompts, extr_successful) if not success]
        # prioritise well-formed prompts (properly extracted from ``` fence)
        unsafe_needed = self.num_prompts - len(succ_extracted_prompts)
        res_prompts = (succ_extracted_prompts[:self.num_prompts] +
                       unsucc_extracted_prompts[:max(unsafe_needed, 0)])
        assert len(res_prompts) == self.num_prompts

        return [self._ensure_completion_guard(p) for p in res_prompts]

    def _generate_meta_completions(self, meta_prompts: list[str], n_per_prompt: int) -> list[str]:
        """
        Returns len(meta_prompts) * n_per_prompt completion texts.

        Raw-text completion makes the instruct meta model "continue the document" — echoing the
        archive format and musing to itself in untagged CoT that eats the whole token budget —
        instead of answering, so this uses a chat-template two-pass thinking budget instead
        (mirrors gepa_baseline's BudgetedReflectionLM):

        Phase 1 — thinking: generate with stop=["</think>"] and max_tokens=think_budget.
        Phase 2 — answer: extend the phase-1 token ids with a forced "</think>\\n\\n" close, then
            generate only the answer.  Unlike GEPA's server-based version, no re-tokenization is
            needed — phase-1 output token ids are reused directly.
        """
        prompt_ids = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            for p in meta_prompts
        ]
        think_params = SamplingParams(
            max_tokens=self.think_budget,
            temperature=self.temperature,
            top_p=self.top_p,
            n=n_per_prompt,
            stop=["</think>"],
        )
        think_results = self.llm.generate(
            [{"prompt_token_ids": ids} for ids in prompt_ids],
            sampling_params=think_params,
            use_tqdm=False,
        )

        answer_inputs = [
            {"prompt_token_ids": list(ids) + list(out.token_ids) + self._think_close_ids}
            for ids, result in zip(prompt_ids, think_results)
            for out in result.outputs
        ]
        answer_params = SamplingParams(
            max_tokens=self.answer_budget,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        answer_results = self.llm.generate(answer_inputs, sampling_params=answer_params, use_tqdm=False)
        return [result.outputs[0].text for result in answer_results]

    @staticmethod
    def _build_meta_prompt(
            base_eval_prompt: str,
            n_examples_from_archive: int,
            previous_generation: list[EvalPromptResult] = None,
    ) -> str:
        instr = REFLECTION_HEADER.replace("{base_eval_prompt}", base_eval_prompt)

        if previous_generation and n_examples_from_archive > 0:
            k = min(n_examples_from_archive, len(previous_generation))
            chosen_prompts = random.sample(previous_generation, k)
            instr += "\n\n## Previous attempts\n\nExample prompts from previous attempts, with their mean score (higher is better):"
            for i, result in enumerate(chosen_prompts):
                instr += f"\n\n# Prompt {i + 1}:\n```\n{result.prompt}\n```\n"
                instr += (
                    f"This prompt, prompt {i + 1}, got a mean score (higher is better) of {result.score:.6f}. "
                    f"Its best completion in the batch was: {result.best_completion!r}\n"
                )

            instr += "\n" + REFLECTION_TASK_FOOTER

        return instr

    @staticmethod
    def _extract_prompt(completion: str, fallback_prompt: str) -> tuple[str, bool]:
        text = completion.strip()
        fenced = re.search(r"```(?:text|prompt)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        if not text:
            text = fallback_prompt
        return text, bool(fenced)

    @staticmethod
    def _ensure_completion_guard(prompt: str) -> str:
        # same guard as gepa_baseline: generated prompts are often useful instructions but lack
        # the trailing cue that makes the eval model emit the regex immediately, without it the
        # baseline stands basically no chance of working well
        if not prompt.rstrip().endswith(":"):
            prompt = prompt + "\n\nThe regex is: "
        return prompt


class PromptEvaluator:
    def __init__(
            self,
            model_name: str,
            pos_examples: list[str],
            neg_examples: list[str],
            max_out_tokens: int,
            tensor_parallel_size: int,
            gpu_memory_utilization: float,
            max_model_len: int,
    ) -> None:
        self.pos_examples = pos_examples
        self.neg_examples = neg_examples
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.max_out_tokens = max_out_tokens
        self.max_model_len = max_model_len

    def evaluate_prompts(
            self,
            prompts: list[str],
            batch_size_per_eval_prompt: int,
            stats: SCSRunStats,
            generation: int,
    ) -> list[EvalPromptResult]:
        eval_prompts = [{"prompt_token_ids": self._token_ids(prompt)} for prompt in prompts]
        llm_res = self.llm.generate(
            eval_prompts,
            sampling_params=SamplingParams(
                max_tokens=self.max_out_tokens,
                top_p=1.0,
                temperature=1.0,
                n=batch_size_per_eval_prompt,
            ),
            use_tqdm=False,
        )

        results: list[EvalPromptResult] = []
        for i, result in enumerate(llm_res):
            completions = [out.text for out in result.outputs]
            accuracies = [self._fitness(text) for text in completions]
            hits = [text for text, acc in zip(completions, accuracies) if acc == 1.0]

            for offset, (text, acc) in enumerate(zip(completions, accuracies), start=1):
                if stats.samples_till_first_hit is None and acc == 1.0:
                    print(f'Hit!  regex={text.strip()!r}', flush=True)
                    stats.samples_till_first_hit = stats.samples + offset

            stats.samples += len(completions)
            stats.hits += len(hits)

            best_idx = int(np.argmax(accuracies))

            results.append(
                EvalPromptResult(
                    generation=generation,
                    prompt=prompts[i],
                    score=float(np.mean(accuracies)),
                    hits=len(hits),
                    samples=len(completions),
                    best_completion=completions[best_idx],
                    best_completion_score=float(accuracies[best_idx]),
                    completions=completions,
                )
            )

        return results

    def _token_ids(self, prompt: str) -> list[int]:
        token_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        # truncate from the left so the trailing "The regex is: " cue survives overlong prompts
        max_prompt_len = self.max_model_len - self.max_out_tokens
        return token_ids[-max_prompt_len:]

    def _fitness(self, completion: str) -> float:
        return regex_accuracy(completion.strip(), self.pos_examples, self.neg_examples)


def run_benchmark(args: argparse.Namespace) -> None:
    out_dir = _output_dir(args.example_id, args.model_name, args.max_out_tokens, args.out_dir)

    result_path = out_dir / "run_result.json"
    if result_path.exists():
        print(f"Result already exists at {result_path}, skipping.", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading s2e-lab/RegexEval (id={args.example_id})...", flush=True)
    ds = load_dataset("s2e-lab/RegexEval", split="train")
    entry = next((ex for ex in ds if ex["id"] == args.example_id), None)
    if entry is None:
        raise ValueError(f"Example id={args.example_id} not found in s2e-lab/RegexEval")

    pos_examples: list[str] = entry["matches"]
    neg_examples: list[str] = entry["non_matches"]
    target_expression: str = entry["expression"]
    seed_prompt = SEED_PROMPT.replace("{refined_prompt}", entry["refined_prompt"])

    random.seed(args.seed)

    evaluator = PromptEvaluator(
        model_name=args.model_name,
        pos_examples=pos_examples,
        neg_examples=neg_examples,
        max_out_tokens=args.max_out_tokens,
        tensor_parallel_size=args.task_tensor_parallel_size,
        gpu_memory_utilization=args.task_gpu_memory_utilization,
        max_model_len=args.task_max_model_len,
    )
    generator = PromptGenerator(
        model_name=args.meta_model_name,
        num_prompts=args.num_prompts_to_eval_per_generation,
        max_tokens=args.meta_max_tokens,
        max_model_len=args.meta_max_model_len,
        tensor_parallel_size=args.meta_tensor_parallel_size,
        gpu_memory_utilization=args.meta_gpu_memory_utilization,
        temperature=args.meta_temperature,
        top_p=args.meta_top_p,
        think_budget=args.think_budget,
    )

    print("SCS baseline (regex)")
    print(f"  example_id                  : {args.example_id}  ({target_expression!r})")
    print(f"  evaluation model            : {args.model_name}")
    print(f"  meta model                  : {args.meta_model_name}")
    print(f"  max samples                 : {args.max_samples:,}")
    print(f"  eval prompts per generation : {args.num_prompts_to_eval_per_generation}")
    print(f"  completions per eval prompt : {args.batch_size_per_eval_prompt}")
    print(f"  generations per trial       : {args.generations_per_trial}")
    print(f"  archive examples per prompt : {args.n_examples_from_archive}")
    print(f"  output dir                  : {out_dir}")
    print()

    stats = SCSRunStats()
    previous_generation: list[EvalPromptResult] = []
    all_results: list[EvalPromptResult] = []
    all_completions: list[str] = []
    start = time.perf_counter()  # exclude LLM startup; measure only the optimization loop

    trial = 0
    while stats.samples < args.max_samples:
        trial += 1
        print(f'Starting trial {trial}', flush=True)
        print(f'Hits / total = {stats.hits} / {stats.samples}')
        for generation in range(args.generations_per_trial):
            eval_prompts = generator.generate_eval_prompts(
                base_eval_prompt=seed_prompt,
                previous_generation=previous_generation,
                n_examples_from_archive=args.n_examples_from_archive,
            )
            generation_results = evaluator.evaluate_prompts(
                prompts=eval_prompts,
                batch_size_per_eval_prompt=args.batch_size_per_eval_prompt,
                stats=stats,
                generation=generation,
            )

            previous_generation = generation_results
            all_results.extend(generation_results)
            for res in generation_results:
                all_completions.extend(res.completions)

    elapsed_s = time.perf_counter() - start

    save_run_completions(out_dir, 0, all_completions)
    print(f"Completions saved to {out_dir}/run_0_completions.jsonl.gz")

    timing = {
        "total_samples": stats.samples,
        "total_time_s": round(elapsed_s, 3),
        "samples_per_second": round(stats.samples / elapsed_s, 2) if elapsed_s > 0 else None,
    }
    _save_json(out_dir / "timing.json", timing)

    output = {
        "config": {
            "example_id": args.example_id,
            "target_expression": target_expression,
            "model_name": args.model_name,
            "meta_model_name": args.meta_model_name,
            "max_out_tokens": args.max_out_tokens,
            "max_samples": args.max_samples,
            "num_prompts_to_eval_per_generation": args.num_prompts_to_eval_per_generation,
            "batch_size_per_eval_prompt": args.batch_size_per_eval_prompt,
            "n_examples_from_archive": args.n_examples_from_archive,
            "generations_per_trial": args.generations_per_trial,
            "think_budget": args.think_budget,
            "seed_prompt": seed_prompt,
            "seed": args.seed,
        },
        "hits": stats.hits,
        "samples": stats.samples,
        "samples_till_first_hit": stats.samples_till_first_hit,
        "elapsed_seconds": round(elapsed_s, 2),
        "generations": generation,
        "eval_prompt_results": [
            {k: v for k, v in asdict(result).items() if k != "completions"}
            for result in all_results
        ],
    }
    _save_json(result_path, output)

    stfh = stats.samples_till_first_hit if stats.samples_till_first_hit is not None else "none"
    print(f"\nFinal hits / total samples: {stats.hits} / {stats.samples}")
    print(f"Samples till first hit: {stfh}")
    print(f"Run time: {elapsed_s:.2f}s  ({stats.samples / elapsed_s:.0f} samples/s)")
    print(f"Results saved to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCS baseline for the regex task")

    p.add_argument("--example_id", type=int, default=DEFAULT_EXAMPLE_ID)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--max_out_tokens", type=int, default=MAX_OUT_TOKENS)
    p.add_argument("--max_samples", type=int, default=SAMPLES_PER_RUN)

    p.add_argument("--meta_model_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--num_prompts_to_eval_per_generation", type=int, default=20)
    p.add_argument("--batch_size_per_eval_prompt", type=int, default=64)
    p.add_argument(
        "--n_examples_from_archive",
        type=int,
        default=3,
        help="Number of previous-generation eval prompts and scores to condition on.",
    )
    p.add_argument("--generations_per_trial", type=int, default=10)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=None)

    p.add_argument("--task_tensor_parallel_size", type=int, default=1)
    p.add_argument("--task_gpu_memory_utilization", type=float, default=0.15)
    p.add_argument("--task_max_model_len", type=int, default=4096)
    p.add_argument("--meta_tensor_parallel_size", type=int, default=1)
    p.add_argument("--meta_gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--meta_max_tokens", type=int, default=512)
    p.add_argument("--meta_max_model_len", type=int, default=8192)
    p.add_argument("--meta_temperature", type=float, default=0.7)
    p.add_argument("--meta_top_p", type=float, default=0.95)
    p.add_argument("--think_budget", type=int, default=4096,
                   help="Max thinking tokens per meta completion.")
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
