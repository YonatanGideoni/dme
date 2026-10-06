from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from dme_regex.consts import MAX_OUT_TOKENS, SAMPLES_PER_RUN
from dme_regex.utils import calc_fitness, regex_accuracy, results_base, save_run_completions

SEED_PROMPT = (
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


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _default_out_dir(example_id: int, model_name: str, max_out_tokens: int) -> Path:
    model_slug = model_name.split("/")[-1]
    return (
        results_base("REGEX_TASK_BASELINE_RESULTS_DIR") / "gfn"
        / f"id{example_id}" / f"maxtok{max_out_tokens}" / model_slug
    )


class CustomMetricsCallback(TrainerCallback):
    def __init__(self, stats: RunStats):
        self.stats = stats

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["custom/hits"] = self.stats.hits
            logs["custom/samples"] = self.stats.samples
            if self.stats.samples_till_first_hit is not None:
                logs["custom/samples_till_first_hit"] = self.stats.samples_till_first_hit


class GFNTrainer(GRPOTrainer):
    def __init__(
        self, *args, target_beta: float, subtb_lambda: float = 1.0,
        pos_examples: list[str] | None = None, neg_examples: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_beta = target_beta
        self.subtb_lambda = subtb_lambda
        self.pos_examples = pos_examples
        self.neg_examples = neg_examples

    def _forward_token_and_eos_logps(self, model, input_ids, attention_mask, completion_ids, logits_to_keep):
        """
        One forward pass returning both (a) log-prob of each actually-generated token
        (what _get_per_token_logps_and_entropies would give) and (b) log-prob the model
        assigns to eos at each position, need both for the GFN subTB loss
        """
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        logits = model(**model_inputs).logits
        logits = logits[:, :-1, :][:, -logits_to_keep:, :] / self.temperature
        log_probs = logits.float().log_softmax(dim=-1)
        token_logps = torch.gather(log_probs, dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)
        eos_logps = log_probs[..., self.processing_class.eos_token_id]
        return token_logps, eos_logps

    def _prefix_log_rewards(self, completion_ids, completion_mask):
        """
        log_r_prefix[i, k] = beta * log(calc_fitness(decode(completion_ids[i, :k])))
        for k = 0..n_i (n_i = sample i's real generated length); shape (bsz, seq_len+1).
        Positions beyond each sample's own n_i are left at 0 -- unused, masked out downstream.
        """
        bsz, seq_len = completion_ids.shape
        lengths = completion_mask.sum(dim=1).long().tolist()
        log_r_prefix = torch.zeros(bsz, seq_len + 1, dtype=torch.float32, device=completion_ids.device)
        ids_cpu = completion_ids.detach().cpu()
        tokenizer = self.processing_class
        for i in range(bsz):
            for k in range(lengths[i] + 1):
                text = tokenizer.decode(ids_cpu[i, :k].tolist(), skip_special_tokens=True).strip()
                fitness = calc_fitness(text, self.pos_examples, self.neg_examples)
                log_r_prefix[i, k] = self.target_beta * math.log(fitness)
        return log_r_prefix

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("GFNTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        bsz, seq_len = completion_ids.shape
        device = completion_ids.device

        token_logps, eos_logps = self._forward_token_and_eos_logps(
            model, input_ids, attention_mask, completion_ids, logits_to_keep,
        )
        token_logps = token_logps.float()

        log_r_prefix = self._prefix_log_rewards(completion_ids, completion_mask)  # (bsz, seq_len+1)

        # boundary: log Fhat(terminal) = log R(terminal) exactly, i.e. P(eos|terminal) := 1,
        # regardless of whether termination was a real eos or the token cap
        n = completion_mask.sum(dim=1).long()  # (bsz,) real generated length per sample
        eos_logps_full = torch.zeros(bsz, seq_len + 1, dtype=torch.float32, device=device)
        eos_logps_full[:, :seq_len] = eos_logps.float()
        eos_logps_full[torch.arange(bsz, device=device), n] = 0.0

        delta = (
            log_r_prefix[:, :-1] - eos_logps_full[:, :-1] + token_logps
            - (log_r_prefix[:, 1:] - eos_logps_full[:, 1:])
        )
        delta_cumsum = torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1).cumsum(1)  # (bsz, seq_len+1)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        mask = positions >= n.unsqueeze(1)  # True = already past this sample's real content

        batch_loss = 0.0
        total_lambda = 0.0
        for subtraj_len in range(1, seq_len + 1):
            subtb_term = (delta_cumsum[:, subtraj_len:] - delta_cumsum[:, :-subtraj_len]) ** 2
            subtb_term[mask[:, subtraj_len - 1:]] = 0
            batch_loss = batch_loss + self.subtb_lambda ** (subtraj_len - 1) * subtb_term.sum()
            total_lambda = total_lambda + self.subtb_lambda ** (subtraj_len - 1) * (~mask[:, subtraj_len - 1:]).sum()
        loss = batch_loss / total_lambda

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["gfn/subtb_delta_abs"].append(delta.detach().abs().mean().item())
        return loss


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
    prompt = SEED_PROMPT.replace("{refined_prompt}", entry["refined_prompt"])

    if args.report_to == "wandb":
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    max_steps = math.ceil(args.max_samples / args.grpo_num_generations)

    print(f"GFN (SubTB) baseline (regex)")
    print(f"  example_id         : {args.example_id}  ({target_expression!r})")
    print(f"  model              : {args.model_name}")
    print(f"  max samples        : {args.max_samples:,}  (targeting {max_steps * args.grpo_num_generations:,})")
    print(f"  max steps          : {max_steps}")
    print(f"  num_generations    : {args.grpo_num_generations}")
    print(f"  target beta        : {args.beta}")
    print(f"  subtb_lambda       : {args.subtb_lambda}")
    print(f"  out_dir            : {out_dir}")
    print()

    stats = RunStats()
    all_completions: list[str] = []

    def gfn_reward_func(prompts, completions, **kwargs):
        log_fitness = []
        for c in completions:
            raw = c[0]["content"] if isinstance(c, list) and isinstance(c[0], dict) else c
            text = raw.strip()
            all_completions.append(raw)

            acc = regex_accuracy(text, pos_examples, neg_examples)
            fitness = calc_fitness(text, pos_examples, neg_examples)
            log_fitness.append(math.log(fitness))

            stats.samples += 1
            if acc == 1.0:
                stats.hits += 1
                if stats.samples_till_first_hit is None:
                    stats.samples_till_first_hit = stats.samples
                    print(f"Hit at sample {stats.samples}!  regex={text!r}", flush=True)

        return [math.exp(lf) for lf in log_fitness]  # raw fitness; only used for TRL's own reward logging

    train_dataset = Dataset.from_dict({"prompt": [prompt] * max_steps})

    peft_config = LoraConfig(
        r=512,
        lora_alpha=32,
        lora_dropout=0.1,
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
        beta=0.0,  # disable GRPO's KL penalty / reference model
        mask_truncated_completions=False,  # keep truncated completions
        seed=args.seed,
        save_strategy="no",
        gradient_checkpointing=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    trainer = GFNTrainer(
        model=args.model_name,
        processing_class=tokenizer,
        reward_funcs=[gfn_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        target_beta=args.beta,
        subtb_lambda=args.subtb_lambda,
        pos_examples=pos_examples,
        neg_examples=neg_examples,
    )
    trainer.add_callback(CustomMetricsCallback(stats))

    t0 = time.perf_counter()
    trainer.train()
    elapsed_s = time.perf_counter() - t0

    # Cache completions and timing (mirrors random_sampling.py / rs_grpo.py convention).
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
            "subtb_lambda": args.subtb_lambda,
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
    p = argparse.ArgumentParser(description="GFlowNet (SubTB) baseline for the regex task")
    p.add_argument("--example_id", type=int, default=DEFAULT_EXAMPLE_ID)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--max_out_tokens", type=int, default=MAX_OUT_TOKENS)
    p.add_argument("--max_samples", type=int, default=SAMPLES_PER_RUN)
    p.add_argument("--grpo_num_generations", type=int, default=64,
                   help="Group size G; total rollout batch = G completions per step.")
    p.add_argument("--learning_rate", type=float, default=1e-7)
    p.add_argument("--beta", type=float, default=1000.0,
                   help="Target-distribution beta: R(y) = calc_fitness(y)^beta (matches the MCMC sampler's target).")
    p.add_argument("--subtb_lambda", type=float, default=1.0,
                   help="SubTB(lambda) sub-trajectory length weighting (1.0 = equal weight, matches eq. 3 exactly).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--wandb_project", default="gfn-regex-baseline")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"],
                   help="Where to log training metrics")
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
