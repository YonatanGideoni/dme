from __future__ import annotations

import json
import math
import os
import re
import signal
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
from datasets import load_dataset
from scipy.stats import norm

from dme.cache import load_jsonl_gz, save_json, save_jsonl_gz
from dme_regex.consts import DATASET

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_entry(example_id: int) -> dict:
    """RegexEval entry with fields id, expression (reference regex), refined_prompt, matches, non_matches."""
    print(f"Loading {DATASET} (id={example_id})...", flush=True)
    ds = load_dataset(DATASET, split="train")
    entry = next((ex for ex in ds if ex["id"] == example_id), None)
    if entry is None:
        raise ValueError(f"Example id={example_id} not found in {DATASET}")
    print(f"Target  : {entry['expression']!r}", flush=True)
    print(f"Examples: {len(entry['matches'])} pos / {len(entry['non_matches'])} neg", flush=True)
    return entry


def wilson_ci(k: int, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    z = norm.ppf(1 - (1 - ci) / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * (p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5 / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _handler(signum, frame):
    raise TimeoutError


def regex_accuracy(pattern: str, pos_examples: list, neg_examples: list, timeout: float = 0.1) -> float:
    """
    Returns the regex' accuracy on the given examples, treating pos and neg equally. Timeout is in seconds, e.g. for
    catastrophic backtracking. Uses SIGALRM, so only call it from the main thread.
    """
    try:
        r = re.compile(pattern)
    except (re.error, OverflowError):
        return 0.0

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        correct = (sum(bool(r.search(s)) for s in pos_examples)
                   + sum(not r.search(s) for s in neg_examples))
        result = correct / (len(pos_examples) + len(neg_examples))
    except TimeoutError:
        result = 0.0
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)

    return result


def calc_fitness(pattern: str, pos_examples: list, neg_examples: list) -> float:
    """Fitness R = exp(regex_accuracy), range [e^0, e^1] = [1, e]. A hit has fitness == e."""
    return math.exp(regex_accuracy(pattern, pos_examples, neg_examples))


def compute_hits_and_stfh(
        completions: list[str], pos_examples: list[str], neg_examples: list[str], micro_batch_size: int,
) -> tuple[int, float | None]:
    """
    Scan completions in generation order; return (n_hits, samples_till_first_hit).

    A hit is regex_accuracy == 1.0. All completions within a micro-batch of micro_batch_size are treated as
    simultaneous; the hit is attributed to the midpoint of its micro-batch.
    """
    n_hits = 0
    stfh: float | None = None
    for i, comp in enumerate(completions):
        if regex_accuracy(comp, pos_examples, neg_examples) == 1.0:
            n_hits += 1
            if stfh is None:
                batch_idx = i // micro_batch_size
                stfh = (batch_idx + 0.5) * micro_batch_size
    return n_hits, stfh


# ---------------------------------------------------------------------------
# Per-run results, for the methods that run several independent runs per cache dir (DME, (1,N), (1+N))
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_id: int
    hits: int
    samples: int
    avg_fitness: float
    samples_till_first_hit: float | None
    run_time: float | None = None


_RUN_RESULT_FIELDS = {f.name for f in fields(RunResult)}


def run_result_path(cache_dir: Path, run_id: int) -> Path:
    return cache_dir / f"run_{run_id}_result.json"


def save_run_result(cache_dir: Path, result: RunResult) -> None:
    save_json(run_result_path(cache_dir, result.run_id), asdict(result))


def load_all_run_results(cache_dir: Path) -> list[RunResult]:
    results = []
    for path in sorted(cache_dir.glob("run_*_result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append(RunResult(**{k: v for k, v in data.items() if k in _RUN_RESULT_FIELDS}))
    return results


def run_completions_path(cache_dir: Path, run_id: int) -> Path:
    return cache_dir / f"run_{run_id}_completions.jsonl.gz"


def save_run_completions(cache_dir: Path, run_id: int, completions: list[str]) -> None:
    save_jsonl_gz(run_completions_path(cache_dir, run_id), completions)


def load_run_completions(cache_dir: Path, run_id: int) -> list[str]:
    return load_jsonl_gz(run_completions_path(cache_dir, run_id))


def save_run_chains(cache_dir: Path, run_id: int, chains_log: list[list[str]]) -> None:
    """Each line is a JSON array [n_chains] of the selected regex per chain at that step."""
    save_jsonl_gz(cache_dir / f"run_{run_id}_chains.jsonl.gz", chains_log)


def results_base(env_var: str) -> Path:
    return Path(os.environ.get(env_var, RESULTS_DIR))


def print_final_summary(cache_dir: Path) -> None:
    results = load_all_run_results(cache_dir)
    if not results:
        print("\nNo completed run results found.")
        return

    results.sort(key=lambda r: r.run_id)
    print("\n=== Per-run totals ===")
    for r in results:
        stfh = f"{r.samples_till_first_hit:.1f}" if r.samples_till_first_hit is not None else "none"
        print(
            f"run={r.run_id:>3}  hits={r.hits:>4}  samples={r.samples:>8}  "
            f"avg_fitness={r.avg_fitness:.4f}  stfh={stfh}"
        )

    overall_hits = sum(r.hits for r in results)
    overall_samples = sum(r.samples for r in results)
    all_stfh = [r.samples_till_first_hit for r in results if r.samples_till_first_hit is not None]

    print("\n=== Overall ===")
    stfh_str = "none"
    if all_stfh:
        stfh_str = (
            f"mean={np.mean(all_stfh):.1f}  median={np.median(all_stfh):.1f} "
            f"({len(all_stfh)}/{len(results)} runs with hit)"
        )
    print(f"runs={len(results)}  hits={overall_hits}  samples={overall_samples}  stfh={stfh_str}")
    print(f"success rate={len(all_stfh) / len(results):.3f}")
