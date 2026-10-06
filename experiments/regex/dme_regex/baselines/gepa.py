from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    optimize_anything,
)

import numpy as np
import requests
from dme_regex.consts import MAX_OUT_TOKENS, SAMPLES_PER_RUN
from dme_regex.utils import regex_accuracy, results_base, save_run_completions
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Seed prompt: no {prev_regex} hint, GEPA optimizes the prompt itself.
SEED_PROMPT = (
    "Please output a regular expression that detects the following, "
    "with nothing preceding or succeeding it. {refined_prompt}\n\nThe regex is: "
)

DEFAULT_EXAMPLE_ID = 2447
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

# Custom reflection prompt template (replaces optimize_anything's auto-built one from
# objective/background, the two are mutually exclusive). The auto-built template's generic
# "system component" framing let the reflection model collapse the optimized prompt into a bare
# regex (the answer) instead of an instruction, which then makes the model give bad nonsensical outputs.
# This template makes the prompt/completion adjacency and the instruction-vs-answer distinction explicit.
REFLECTION_PROMPT_TEMPLATE = """You are optimizing the exact prompt text fed to a small language model so that it outputs a regular expression matching all (hidden) positive examples and rejecting all negative examples.

## Critical structural fact

The component you produce is concatenated character-for-character directly in front of the eval model's generation. There is no chat template, no separator, and no "user turn" between your text and the model's output — the model's completion begins at the exact character where your text ends. For example, if your component ends in "...The regex is: " and the model then generates "^abc$", the model literally saw "...The regex is: " and continued it with "^abc$".

## What you must produce — and must NOT produce

Your output must always be a natural-language INSTRUCTION/PROMPT that elicits a regex from the eval model — never a bare regex, never the eval model's own completion, and never just a corrected version of "best_regex_in_batch" from the evaluation data below. A bare regex is not a valid replacement for the component: it removes the instruction context the eval model needs and will not improve the score. If you find yourself about to output something that itself compiles as a regex with no surrounding instruction, stop — that is the failure mode this warning exists to prevent.

## Current Component

The prompt text currently being optimized:

```
<curr_param>
```

## Evaluation Results

Performance data from evaluating the current component:

```
<side_info>
```

## Your Task

Analyze the evaluation results:
- **Failure patterns**: What specific errors or failure modes appear (e.g. the eval model rambling instead of emitting a regex immediately, wrong output format, truncation)?
- **Success patterns**: What phrasing led to correct or close regexes, and should be preserved?
- **Root causes**: What about the current instruction confuses the eval model?

Based on your analysis, propose an improved instruction/prompt that:
1. Addresses the identified failure patterns and root causes
2. Preserves successful phrasing from the current version
3. Makes the eval model as likely as possible to emit, immediately and with nothing else, a regex matching the task description in the original seed prompt (do not change or narrow the described task itself — only how the instruction is phrased)

## Output Format

Provide ONLY the improved instruction/prompt within ``` blocks. It must be a complete, drop-in replacement for the current component — natural-language instruction text, not a regex."""


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def wait_for_openai_server(base_url: str, timeout_seconds: int = 300) -> None:
    deadline = time.time() + timeout_seconds
    headers = {"Authorization": "Bearer EMPTY"}
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/models", headers=headers, timeout=5)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for OpenAI-compatible server at {base_url}")


def init_vllm_server(
        model_name: str,
        host: str,
        port: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        tensor_parallel_size: int,
        log_path: Path = Path("vllm.log"),
) -> str:
    base_url = f"http://{host}:{port}/v1"
    if is_port_open(host, port):
        print(f"[reflection] Found existing server on {host}:{port}")
        return base_url
    print(f"[reflection] No server found on {host}:{port}")
    print(f"[reflection] Launching vLLM server for {model_name}")
    cmd = [
        "vllm", "serve", model_name,
        "--host", host, "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--api-key", "EMPTY",
    ]
    print("[reflection] Command:")
    print(" ".join(cmd))
    log_file = open(log_path, "ab")
    subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    print("[reflection] Waiting for server startup...")
    wait_for_openai_server(base_url)
    print("[reflection] Server is ready")
    return base_url


# ---------------------------------------------------------------------------
# Budgeted reflection LM (two-pass thinking budget)
# ---------------------------------------------------------------------------

class BudgetedReflectionLM:
    """
    Wraps a vLLM OpenAI-compatible server with a two-pass thinking budget:

    Phase 1 — thinking: call /chat/completions with stop=["</think>"] and
        max_tokens=think_budget.  The model generates <think>\\n...thoughts...
        and stops before (or at budget, whichever comes first).

    Phase 2 — answer: build a raw text prefix that ends with
        <think>\\n...thoughts...\\n</think>\\n\\n, then call /completions so
        the model generates only the answer (no repeat thinking).

    This prevents the think-blob from filling the context and corrupting
    GEPA's candidate pool when Qwen3 extended-thinking is enabled.
    """

    def __init__(
            self,
            model_name: str,
            api_base: str,
            api_key: str,
            temperature: float = 0.7,
            top_p: float = 0.95,
            think_budget: int = 4096,
            answer_budget: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.think_budget = think_budget
        self.answer_budget = answer_budget
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _phase2_token_ids(self, user_prompt: str, thinking_content: str) -> list[int]:
        """Token IDs for the phase 2 prefix: ends with </think>\\n\\n.

        Passing token IDs (instead of raw text) to /v1/completions avoids
        mis-tokenization of special tokens like <|im_start|> and <think>.
        """
        base = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        # thinking_content from phase 1 = "<think>\n{thoughts}" (no closing tag)
        # Re-tokenize ONLY the thinking portion (special tokens already present in base).
        THINK_CLOSE = 151668  # </think>
        NEWLINE = 198  # \n
        thinking_ids = self._tokenizer(
            thinking_content.rstrip("\n") + "\n",
            add_special_tokens=False,
        ).input_ids
        return list(base) + thinking_ids + [THINK_CLOSE, NEWLINE]

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if not isinstance(prompt, str):
            raise TypeError("BudgetedReflectionLM only accepts string prompts")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # --- Phase 1: thinking ---
        r1 = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.think_budget,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stop": ["</think>"],
            },
            timeout=600,
        )
        r1.raise_for_status()
        thinking_content: str = r1.json()["choices"][0]["message"]["content"]

        # --- Phase 2: answer ---
        # Pass pre-tokenized IDs to avoid mis-tokenization of special tokens.
        phase2_ids = self._phase2_token_ids(prompt, thinking_content)
        r2 = requests.post(
            f"{self.api_base}/completions",
            headers=headers,
            json={
                "model": self.model_name,
                "prompt": phase2_ids,
                "max_tokens": self.answer_budget,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stop": ["<|im_end|>"],
            },
            timeout=600,
        )
        r2.raise_for_status()
        answer: str = r2.json()["choices"][0]["text"]

        # Return only the answer — GEPA's extractor only needs the proposed instruction.
        # Returning the thinking block too causes the extractor to pick up <think> content
        # when the answer has no backticks.
        return answer


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class GEPARunStats:
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
        base / "gepa"
        / f"id{example_id}" / f"maxtok{max_out_tokens}" / model_slug
    )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RegexEvaluator:
    """
    Drop-in GEPA evaluator for the regex task.

    Each call generates `num_completions` regexes from the current prompt,
    scores them by regex_accuracy, and returns (mean_accuracy, side_info).
    Startup time (LLM loading) is excluded from timing tracked by the caller.
    """

    def __init__(
            self,
            model_name: str,
            pos_examples: list[str],
            neg_examples: list[str],
            max_out_tokens: int,
            num_completions: int,
            tensor_parallel_size: int,
            gpu_memory_utilization: float,
            max_model_len: int,
            cache_path: Path | None = None,
    ) -> None:
        self.pos_examples = pos_examples
        self.neg_examples = neg_examples
        self.cache_path = cache_path

        self.completions: list[str] = []
        self.stats = GEPARunStats()
        if self.cache_path and self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                self.stats = GEPARunStats(**data)
                print(f"[evaluator] Resuming from cache: {self.stats}")
            except Exception as e:
                print(f"[evaluator] Warning: Failed to load cache ({e}). Starting fresh.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.sampling_params = SamplingParams(
            max_tokens=max_out_tokens,
            top_p=1.0,
            temperature=1.0,
            n=num_completions,
        )

    def __call__(self, prompt: str, **_: Any) -> tuple[float, dict[str, Any]]:
        if not prompt.rstrip().endswith(":"):
            # the prompt often ends up being useful instructions but without the right formatting,
            # telling the eval model to output a regex. without this GEPA stands basically no chance
            # of working well, so we manually add it here
            prompt = prompt + "\n\nThe regex is: "
        token_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        results = self.llm.generate(
            [{"prompt_token_ids": token_ids}],
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        completions = [out.text for out in results[0].outputs]
        self.completions.extend(completions)

        accuracies = [
            regex_accuracy(c.strip(), self.pos_examples, self.neg_examples)
            for c in completions
        ]
        hits = [c for c, a in zip(completions, accuracies) if a == 1.0]

        for offset, (c, a) in enumerate(zip(completions, accuracies), start=1):
            if self.stats.samples_till_first_hit is None and a == 1.0:
                self.stats.samples_till_first_hit = self.stats.samples + offset

        self.stats.samples += len(completions)
        self.stats.hits += len(hits)

        if hits:
            print(f"Hit!  regex={hits[0].strip()!r}", flush=True)

        if self.cache_path:
            _save_json(self.cache_path, dataclasses.asdict(self.stats))

        best_idx = int(np.argmax(accuracies))
        best_regex_in_batch = completions[best_idx].strip()
        best_regex_in_batch_accuracy = float(accuracies[best_idx])

        mean_accuracy = float(np.mean(accuracies))
        side_info: dict[str, Any] = {
            "prompt": prompt,
            "note": (
                "'some_completions' are the literal, unedited text the eval model generated "
                "immediately after 'prompt' above, with no separator or chat turn in between. "
                "'best_regex_in_batch' is just whichever completion in this batch "
                "scored highest; it is not a known-correct answer to copy in as the new prompt "
                "(see its accuracy, which is usually well below 1.0)."
            ),
            "some_completions": completions[:10],
            "score": mean_accuracy,
            "score_per_some_completions": accuracies[:10],
            "correct_regexes": hits,
            "best_regex_in_batch": best_regex_in_batch,
            "best_regex_in_batch_accuracy": best_regex_in_batch_accuracy,
        }
        return mean_accuracy, side_info


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    print(f"Loading s2e-lab/RegexEval (id={args.example_id})...", flush=True)
    ds = load_dataset("s2e-lab/RegexEval", split="train")
    entry = next((ex for ex in ds if ex["id"] == args.example_id), None)
    if entry is None:
        raise ValueError(f"Example id={args.example_id} not found in s2e-lab/RegexEval")

    pos_examples: list[str] = entry["matches"]
    neg_examples: list[str] = entry["non_matches"]
    target_expression: str = entry["expression"]
    seed_prompt = SEED_PROMPT.replace("{refined_prompt}", entry["refined_prompt"])

    out_dir = _output_dir(args.example_id, args.model_name, args.max_out_tokens, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("GEPA baseline (regex)")
    print(f"  example_id            : {args.example_id}  ({target_expression!r})")
    print(f"  evaluation model      : {args.model_name}")
    print(f"  reflection model      : {args.reflection_model_name}")
    print(f"  max samples           : {args.max_samples:,}")
    print(f"  completions per prompt: {args.num_completions}")
    print(f"  output dir            : {out_dir}")
    print()

    evaluator = RegexEvaluator(
        model_name=args.model_name,
        pos_examples=pos_examples,
        neg_examples=neg_examples,
        max_out_tokens=args.max_out_tokens,
        num_completions=args.num_completions,
        tensor_parallel_size=args.task_tensor_parallel_size,
        gpu_memory_utilization=args.task_gpu_memory_utilization,
        max_model_len=args.max_eval_model_len,
        cache_path=out_dir / "stats_cache.json",
    )

    reflection_api_base = init_vllm_server(
        model_name=args.reflection_model_name,
        host=args.reflection_host,
        port=args.reflection_port,
        gpu_memory_utilization=args.reflection_gpu_memory_utilization,
        max_model_len=args.max_reflection_model_len,
        tensor_parallel_size=args.reflection_tensor_parallel_size,
        log_path=out_dir / "reflection_vllm.log",
    )

    reflection_lm = BudgetedReflectionLM(
        model_name=args.reflection_model_name,
        api_base=reflection_api_base,
        api_key="EMPTY",
        temperature=args.reflection_temperature,
        top_p=args.reflection_top_p,
        think_budget=args.think_budget,
        answer_budget=args.answer_budget,
    )

    t0 = time.perf_counter()  # exclude LLM startup; measure only optimize_anything
    result = optimize_anything(
        evaluator=evaluator,
        seed_candidate=seed_prompt,
        valset=None,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(out_dir / "gepa_artifacts"),
                max_metric_calls=math.ceil(args.max_samples / args.num_completions),
            ),
            reflection=ReflectionConfig(
                reflection_lm=reflection_lm,
                reflection_prompt_template=REFLECTION_PROMPT_TEMPLATE,
            ),
        ),
    )
    elapsed_s = time.perf_counter() - t0

    save_run_completions(out_dir, 0, evaluator.completions)
    print(f"Completions saved to {out_dir}/run_0_completions.jsonl.gz")

    stats = evaluator.stats
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
            "reflection_model_name": args.reflection_model_name,
            "max_out_tokens": args.max_out_tokens,
            "max_samples": args.max_samples,
            "num_completions": args.num_completions,
            "max_eval_model_len": args.max_eval_model_len,
            "seed": args.seed,
            "seed_prompt": seed_prompt,
        },
        "hits": stats.hits,
        "samples": stats.samples,
        "samples_till_first_hit": stats.samples_till_first_hit,
        "elapsed_seconds": round(elapsed_s, 2),
        "best_candidate": getattr(result, "best_candidate", None),
    }
    _save_json(out_dir / "run_result.json", output)

    stfh = stats.samples_till_first_hit if stats.samples_till_first_hit is not None else "none"
    print(f"\nFinal hits / total samples: {stats.hits} / {stats.samples}")
    print(f"Samples till first hit: {stfh}")
    print(f"Run time: {elapsed_s:.2f}s  ({stats.samples / elapsed_s:.0f} samples/s)")
    print(f"Results saved to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GEPA baseline for the regex task")

    # problem
    p.add_argument("--example_id", type=int, default=DEFAULT_EXAMPLE_ID)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--max_out_tokens", type=int, default=MAX_OUT_TOKENS)
    p.add_argument("--max_samples", type=int, default=SAMPLES_PER_RUN)

    # reflection / refiner
    p.add_argument("--reflection_model_name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--reflection_temperature", type=float, default=0.7)
    p.add_argument("--reflection_top_p", type=float, default=0.95)
    p.add_argument("--reflection_host", default="127.0.0.1")
    p.add_argument("--reflection_port", type=int, default=8001)
    p.add_argument("--reflection_gpu_memory_utilization", type=float, default=0.7)
    p.add_argument("--max_reflection_model_len", type=int, default=8192)
    p.add_argument("--reflection_tensor_parallel_size", type=int, default=1)

    # evaluation
    p.add_argument(
        "--num_completions",
        type=int,
        default=64,
        help="Regex completions generated per evaluator call.",
    )
    p.add_argument("--task_tensor_parallel_size", type=int, default=1)
    p.add_argument("--task_gpu_memory_utilization", type=float, default=0.15)
    p.add_argument("--max_eval_model_len", type=int, default=8208)

    # thinking budget
    p.add_argument("--think_budget", type=int, default=4096,
                   help="Max thinking tokens for the reflection model (phase 1).")
    p.add_argument("--answer_budget", type=int, default=1024,
                   help="Max answer tokens for the reflection model (phase 2).")

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
