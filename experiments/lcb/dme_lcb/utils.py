from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from dme.cache import claimed_run_lock, save_json, save_jsonl_gz

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(os.environ.get("DME_LCB_RESULTS_DIR", EXPERIMENT_DIR / "results"))
QUESTION_IDS_DIR = EXPERIMENT_DIR / "question_ids"
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_result_path(cache_dir: Path, run_id: int) -> Path:
    return cache_dir / f"run_{run_id}_result.json"


def is_run_done(cache_dir: Path, run_id: int) -> bool:
    return run_result_path(cache_dir, run_id).exists()


def save_result(cache_dir: Path, run_id: int, result) -> None:
    save_json(run_result_path(cache_dir, run_id), asdict(result))


def save_list(cache_dir: Path, run_id: int, name: str, rows: list) -> None:
    """Per-run log, run_{run_id}_{name}.jsonl.gz, one row per candidate (or per round for round-level logs)."""
    save_jsonl_gz(cache_dir / f"run_{run_id}_{name}.jsonl.gz", rows)


def question_set_tag(question_ids_file: Path) -> str:
    """The question ids file's stem, a cache-path segment so different question subsets never collide."""
    return question_ids_file.stem


@contextlib.contextmanager
def claim_runs(question_runs: list) -> Iterator[list]:
    """Claim (via file locks) the given question runs that aren't done or held by another worker."""
    claimed = []
    with contextlib.ExitStack() as stack:
        for q in question_runs:
            if is_run_done(q.cache_dir, q.run_idx):
                continue
            q.cache_dir.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(claimed_run_lock(q.cache_dir / f"run_{q.run_idx}.lock"))
            if lock is None:
                continue
            claimed.append(q)
        yield claimed


def get_base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--question_ids_file", type=Path, default=QUESTION_IDS_DIR / "random_sample_300.txt",
                   help="Whitespace-separated LCB question ids. Each file gets its own cache subtree.")
    p.add_argument("--num_runs", type=int, default=5, help="Independent runs per question")
    p.add_argument("--max_out_tokens", type=int, default=2000)
    p.add_argument("--timeout", type=int, default=6, help="Per unit test timeout (s)")
    p.add_argument("--num_process_evaluate", type=int, default=16, help="Grader worker processes")
    p.add_argument("--parallel_slice", type=int, default=16, help="Number of question runs processed concurrently")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--dtype", default="half")
    p.add_argument("--rich_cache", action=argparse.BooleanOptionalAction, default=True,
                   help="Also save every candidate's raw output, logprobs, accuracy and grading metadata")
    return p


def write_campaign_config(results_dir: Path, args: argparse.Namespace) -> None:
    """Record of the CLI args a campaign was launched with, for provenance (not validated against)."""
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "model_style"}
    path = results_dir / f"campaign_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


@dataclass
class DMERunResult:
    run_id: int
    hits: int
    samples: int
    avg_fitness: float
    samples_till_first_hit: float | None
    run_time: float | None = None


@dataclass
class ESRunResult:
    run_id: int
    strategy: str  # "non_elitist" ((1,N)) or "elitist" ((1+N))
    hit: bool
    n_evalled: int
    best_accuracy: float
    n_evalled_at_hit: int | None
    run_time: float | None = None
