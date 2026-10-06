from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from dme_regex.consts import MAX_OUT_TOKENS, SAMPLES_PER_RUN
from dme_regex.utils import calc_fitness, regex_accuracy, results_base, save_run_completions

# Same prompt as random sampling, no previous-regex hint.
NO_PREV_REGEX_PROMPT = (
    "Please output a regular expression that detects the following, "
    "with nothing preceding or succeeding it. {refined_prompt}\n\nThe regex is: "
)

DEFAULT_EXAMPLE_ID = 2447
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dataclass
class RunStats:
    hits: int = 0
    samples: int = 0
    samples_till_first_hit: int | None = None


class AdvantageState:
    """Shared state: passes exact RS-GRPO advantages from reward fn to compute_loss."""
    exact_advantages: torch.Tensor | None = None


class CustomMetricsCallback(TrainerCallback):
    def __init__(self, stats: RunStats):
        self.stats = stats

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["custom/hits"] = self.stats.hits
            logs["custom/samples"] = self.stats.samples
            if self.stats.samples_till_first_hit is not None:
                logs["custom/samples_till_first_hit"] = self.stats.samples_till_first_hit


class RSGRPOTrainer(GRPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if AdvantageState.exact_advantages is not None:
            inputs["advantages"] = -AdvantageState.exact_advantages.to(inputs["prompt_ids"].device)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _default_out_dir(example_id: int, model_name: str, max_out_tokens: int) -> Path:
    model_slug = model_name.split("/")[-1]
    return (
        results_base("REGEX_TASK_BASELINE_RESULTS_DIR") / "rs_grpo"
        / f"id{example_id}" / f"maxtok{max_out_tokens}" / model_slug
    )


def run_benchmark(args: argparse.Namespace) -> None:
    out_dir = (
        _default_out_dir(args.example_id, args.model_name, args.max_out_tokens)
        if args.out_dir is None else args.out_dir
    )

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
    prompt = NO_PREV_REGEX_PROMPT.replace("{refined_prompt}", entry["refined_prompt"])

    if args.report_to == "wandb":
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    max_steps = math.ceil(args.max_samples / args.grpo_num_generations)

    print(f"RS-GRPO baseline (regex)")
    print(f"  example_id         : {args.example_id}  ({target_expression!r})")
    print(f"  model              : {args.model_name}")
    print(f"  max samples        : {args.max_samples:,}  (targeting {max_steps * args.grpo_num_generations:,})")
    print(f"  max steps          : {max_steps}")
    print(f"  num_generations    : {args.grpo_num_generations}")
    print(f"  beta               : {args.beta}")
    print(f"  out_dir            : {out_dir}")
    print()

    stats = RunStats()
    all_completions: list[str] = []

    def rs_reward_func(prompts, completions, **kwargs):
        scores = []
        for c in completions:
            raw = c[0]["content"] if isinstance(c, list) and isinstance(c[0], dict) else c
            text = raw.strip()
            all_completions.append(raw)

            acc = regex_accuracy(text, pos_examples, neg_examples)
            score = calc_fitness(text, pos_examples, neg_examples)
            scores.append(score)

            stats.samples += 1
            if acc == 1.0:
                stats.hits += 1
                if stats.samples_till_first_hit is None:
                    stats.samples_till_first_hit = stats.samples
                    print(f"Hit at sample {stats.samples}!  regex={text!r}", flush=True)

        # Exact RS-GRPO advantages: A = 1/beta * (exp(beta*r) / mean(exp(beta*r)) - 1)
        num_gens = args.grpo_num_generations
        if len(scores) % num_gens == 0:
            batch_size = len(scores) // num_gens
            scores_t = torch.tensor(scores, dtype=torch.float32).view(batch_size, num_gens)
            exp_r = torch.exp(args.beta * scores_t)
            advs = (1.0 / args.beta) * (exp_r / exp_r.mean(dim=1, keepdim=True) - 1.0)
            AdvantageState.exact_advantages = advs.view(-1)
        else:
            AdvantageState.exact_advantages = None

        return scores

    train_dataset = Dataset.from_dict({"prompt": [prompt] * max_steps})

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        num_generations=args.grpo_num_generations,
        max_completion_length=args.max_out_tokens,
        per_device_train_batch_size=args.grpo_num_generations,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        use_liger_kernel=False,
        logging_steps=1,
        report_to=args.report_to,
        beta=0.0,
        seed=args.seed,
        save_strategy="no",
        gradient_checkpointing=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    trainer = RSGRPOTrainer(
        model=args.model_name,
        processing_class=tokenizer,
        reward_funcs=[rs_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )
    trainer.add_callback(CustomMetricsCallback(stats))

    t0 = time.perf_counter()
    trainer.train()
    elapsed_s = time.perf_counter() - t0

    # Cache completions and timing (mirrors random_sampling.py convention).
    save_run_completions(out_dir, 0, all_completions)

    timing = {
        "total_samples": stats.samples,
        "total_time_s": round(elapsed_s, 3),
        "samples_per_second": round(stats.samples / elapsed_s, 2) if elapsed_s > 0 else None,
    }
    _save_json(out_dir / "timing.json", timing)

    result = {
        "config": {
            "example_id": args.example_id,
            "target_expression": target_expression,
            "model_name": args.model_name,
            "max_out_tokens": args.max_out_tokens,
            "max_samples": args.max_samples,
            "actual_samples": stats.samples,
            "max_steps": max_steps,
            "grpo_num_generations": args.grpo_num_generations,
            "beta": args.beta,
            "seed": args.seed,
        },
        "hits": stats.hits,
        "samples": stats.samples,
        "samples_till_first_hit": stats.samples_till_first_hit,
        "elapsed_seconds": round(elapsed_s, 2),
    }
    _save_json(out_dir / "run_result.json", result)

    stfh = stats.samples_till_first_hit if stats.samples_till_first_hit is not None else "none"
    print(f"\nFinal hits / total samples: {stats.hits} / {stats.samples}")
    print(f"Samples till first hit: {stfh}")
    print(f"Run time: {elapsed_s:.2f}s  ({stats.samples / elapsed_s:.0f} samples/s)")
    print(f"Results saved to {out_dir}")

    if args.report_to == "wandb":
        wandb.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact RS-GRPO baseline for the regex task")
    p.add_argument("--example_id", type=int, default=DEFAULT_EXAMPLE_ID)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--max_out_tokens", type=int, default=MAX_OUT_TOKENS)
    p.add_argument("--max_samples", type=int, default=SAMPLES_PER_RUN)
    p.add_argument("--grpo_num_generations", type=int, default=64,
                   help="Group size G for GRPO; total batch = G completions per step.")
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=8.0,
                   help="Risk-sensitivity parameter for RS-GRPO advantage.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--wandb_project", default="rs-grpo-regex-baseline")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"],
                   help="Where to log training metrics")
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
