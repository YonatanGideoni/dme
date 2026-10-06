from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from dme_regex.consts import MAX_OUT_TOKENS, SAMPLES_PER_RUN
from dme_regex.utils import regex_accuracy, results_base, wilson_ci

# No previous-regex hint — pure i.i.d. sampling.
NO_PREV_REGEX_PROMPT = (
    "Please output a regular expression that detects the following, "
    "with nothing preceding or succeeding it. {refined_prompt}\n\nThe regex is: "
)

DEFAULT_EXAMPLE_ID = 2447
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
BENCHMARK_MULTIPLIER = 3    # samples per worker = BENCHMARK_MULTIPLIER × SAMPLES_PER_RUN
N_BOOTSTRAP = 10_000
RNG_SEED = 42


# ---------------------------------------------------------------------------
# Bootstrap stats (mirrors password/baselines/random_sampling.py)
# ---------------------------------------------------------------------------

def _simulate_runs(
    hit_flags: np.ndarray,
    run_size: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(hit_flags)
    draws = hit_flags[rng.integers(0, N, size=(n_resamples, run_size))]
    had_hit = draws.any(axis=1)
    pos = draws.argmax(axis=1).astype(float) + 1.0
    pos[~had_hit] = np.nan
    return had_hit, pos


def bootstrap_success_rate(
    hit_flags: np.ndarray,
    run_size: int,
    n_resamples: int = N_BOOTSTRAP,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    had_hit, _ = _simulate_runs(hit_flags, run_size, n_resamples, rng)
    boot = had_hit.astype(float)
    alpha = (1 - ci) / 2
    return float(boot.mean()), *np.quantile(boot, [alpha, 1 - alpha]).tolist()


def bootstrap_mean_stfh(
    hit_flags: np.ndarray,
    run_size: int,
    n_resamples: int = N_BOOTSTRAP,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float] | None:
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    _, first_hit_pos = _simulate_runs(hit_flags, run_size, n_resamples, rng)
    successful = first_hit_pos[~np.isnan(first_hit_pos)]
    if len(successful) == 0:
        return None
    alpha = (1 - ci) / 2
    return float(successful.mean()), *np.quantile(successful, [alpha, 1 - alpha]).tolist()


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _output_dir(example_id: int, model_name: str, max_out_tokens: int, out_dir: Path | None) -> Path:
    if out_dir is not None:
        return out_dir
    model_slug = model_name.split("/")[-1]
    base = results_base("REGEX_TASK_BASELINE_RESULTS_DIR")
    return (
        base / "random"
        / f"id{example_id}" / f"maxtok{max_out_tokens}" / model_slug
    )


def _chunk_path(out_dir: Path, worker_id: int) -> Path:
    return out_dir / f"completions_{worker_id}.jsonl.gz"


def _timing_path(out_dir: Path, worker_id: int) -> Path:
    return out_dir / f"timing_{worker_id}.json"


def _save_chunk(
    out_dir: Path, worker_id: int, completions: list[str], elapsed_s: float
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    path = _chunk_path(out_dir, worker_id)
    tmp = path.with_suffix(".gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
        for c in completions:
            f.write(json.dumps(c) + "\n")
    tmp.replace(path)

    timing = {
        "total_samples": len(completions),
        "total_time_s": round(elapsed_s, 3),
        "samples_per_second": round(len(completions) / elapsed_s, 2),
    }
    tp = _timing_path(out_dir, worker_id)
    tp.write_text(json.dumps(timing, indent=2), encoding="utf-8")


def _load_all_chunks(out_dir: Path) -> list[str]:
    all_completions = []
    for path in sorted(out_dir.glob("completions_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            all_completions.extend(json.loads(line) for line in f)
    return all_completions


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    batch_size: int,
    max_out_tokens: int,
) -> list[str]:
    token_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    results = llm.generate(
        [{"prompt_token_ids": token_ids}],
        sampling_params=SamplingParams(
            max_tokens=max_out_tokens,
            top_p=1.0,
            temperature=1.0,
            n=batch_size,
        ),
        use_tqdm=False,
    )
    return [out.text.strip() for out in results[0].outputs]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(
    all_completions: list[str],
    pos_examples: list[str],
    neg_examples: list[str],
    samples_per_run: int,
) -> None:
    hit_flags = np.array(
        [regex_accuracy(c.strip(), pos_examples, neg_examples) == 1.0 for c in all_completions],
        dtype=bool,
    )
    n = len(hit_flags)
    n_hits = int(hit_flags.sum())
    hit_prob = n_hits / n

    w_lo, w_hi = wilson_ci(n_hits, n)
    print(f"\n── Empirical hit probability ────────────────────────────────────────")
    print(f"  hits / samples : {n_hits} / {n:,}")
    print(f"  P(hit)         : {hit_prob:.4e}")
    print(f"  Wilson 95% CI  : [{w_lo:.4e}, {w_hi:.4e}]")

    if n_hits == 0:
        print("\n  No hits observed — cannot estimate run-level statistics.")
        return

    rng = np.random.default_rng(RNG_SEED)

    sr, sr_lo, sr_hi = bootstrap_success_rate(hit_flags, samples_per_run, rng=rng)
    print(f"\n── Per-run success rate  P(≥1 hit in {samples_per_run:,} samples) ──────────")
    print(f"  Point estimate : {sr:.4f}")
    print(f"  95% CI         : [{sr_lo:.4f}, {sr_hi:.4f}]")
    print(f"  (bootstrap over {N_BOOTSTRAP:,} simulated runs)")

    result = bootstrap_mean_stfh(hit_flags, samples_per_run, rng=rng)
    print(f"\n── Mean samples till first hit  (successful runs only) ──────────────")
    if result is None:
        print("  No bootstrap run produced a hit.")
    else:
        fh, fh_lo, fh_hi = result
        print(f"  Point estimate : {fh:,.1f}")
        print(f"  95% CI         : [{fh_lo:,.1f}, {fh_hi:,.1f}]")
        print(f"  (bootstrap over {N_BOOTSTRAP:,} simulated runs)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    total_samples = args.benchmark_multiplier * args.samples_per_run

    print(f"Loading s2e-lab/RegexEval (id={args.example_id})...", flush=True)
    ds = load_dataset("s2e-lab/RegexEval", split="train")
    entry = next((ex for ex in ds if ex["id"] == args.example_id), None)
    if entry is None:
        raise ValueError(f"Example id={args.example_id} not found in s2e-lab/RegexEval")

    pos_examples: list[str] = entry["matches"]
    neg_examples: list[str] = entry["non_matches"]
    target_expression: str = entry["expression"]
    prompt = NO_PREV_REGEX_PROMPT.replace("{refined_prompt}", entry["refined_prompt"])

    out_dir = _output_dir(args.example_id, args.model_name, args.max_out_tokens, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Random-sampling baseline")
    print(f"  example_id      : {args.example_id}  ({target_expression!r})")
    print(f"  model           : {args.model_name}")
    print(f"  samples_per_run : {args.samples_per_run:,}")
    print(f"  samples/worker  : {total_samples:,}  ({args.benchmark_multiplier}×)")
    print(f"  worker_id       : {args.worker_id}")
    print(f"  batch_size      : {args.batch_size}")
    print(f"  output dir      : {out_dir}")
    print()

    chunk_path = _chunk_path(out_dir, args.worker_id)
    if chunk_path.exists():
        print(f"Chunk {args.worker_id} already exists, skipping generation.", flush=True)
    elif not args.stats_only:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        llm = LLM(
            model=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        completions: list[str] = []
        samples_done = 0
        t0 = time.perf_counter()  # start timing after vLLM is loaded
        log_every = max(args.batch_size, total_samples // 20)

        while samples_done < total_samples:
            batch = min(args.batch_size, total_samples - samples_done)
            completions.extend(_generate(llm, tokenizer, prompt, batch, args.max_out_tokens))
            samples_done += batch

            if samples_done % log_every == 0 or samples_done == total_samples:
                print(
                    f"  [{samples_done:>{len(str(total_samples))}}/{total_samples}]  "
                    f"elapsed: {time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

        elapsed_s = time.perf_counter() - t0
        _save_chunk(out_dir, args.worker_id, completions, elapsed_s)
        print(
            f"Chunk {args.worker_id} saved to {chunk_path}  "
            f"({elapsed_s:.1f}s, {len(completions)/elapsed_s:.0f} samples/s)"
        )

    # Aggregate all available chunks and print stats.
    all_completions = _load_all_chunks(out_dir)
    n_chunks = len(list(out_dir.glob("completions_*.jsonl.gz")))
    print(f"\nAggregating {n_chunks} chunk(s) — {len(all_completions):,} total completions.")
    print_stats(all_completions, pos_examples, neg_examples, args.samples_per_run)
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Best-of-N (IID random sampling) baseline for the regex task")
    p.add_argument("--example_id", type=int, default=DEFAULT_EXAMPLE_ID)
    p.add_argument("--model_name", default=DEFAULT_MODEL)
    p.add_argument("--max_out_tokens", type=int, default=MAX_OUT_TOKENS)
    p.add_argument("--samples_per_run", type=int, default=SAMPLES_PER_RUN)
    p.add_argument("--benchmark_multiplier", type=int, default=BENCHMARK_MULTIPLIER,
                   help="Samples per worker = benchmark_multiplier × samples_per_run.")
    p.add_argument("--batch_size", type=int, default=1024,
                   help="Completions per vLLM call.")
    p.add_argument("--worker_id", type=int, default=0,
                   help="Index of this worker; each saves its own chunk file.")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--stats_only", action="store_true",
                   help="Skip generation; aggregate existing chunks and print stats.")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return p.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
