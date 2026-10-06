import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dme_math.problems.problem_utils import EvaluationResult


@dataclass
class RunRecord:
    algorithm_id: int
    algorithm_code: str | None
    full_completion: str
    prompt_used: str
    eval_result: EvaluationResult | None
    token_usage: dict
    api_cost_usd: dict
    tokens: list[int] | None = None
    logprobs: list[float] | None = None


def _record_to_json(r: RunRecord) -> dict:
    if r.eval_result is not None:
        success = r.eval_result.success
        execution_time = r.eval_result.execution_time
        error_message = r.eval_result.error_message
        metrics = r.eval_result.metrics
    else:
        success, execution_time, error_message, metrics = False, 0.0, "No code block found in completion", {}
    return {
        "success": success,
        "execution_time": execution_time,
        "error_message": error_message,
        "metrics": metrics,
        "algorithm_id": r.algorithm_id,
        "algorithm_code": r.algorithm_code,
        "full_completion": r.full_completion,
        "prompt_used": r.prompt_used,
        "token_usage": r.token_usage,
        "api_cost_usd": r.api_cost_usd,
        "has_logprobs": r.tokens is not None and r.logprobs is not None,
    }


def _build_metadata(problem_name: str, model_name: str, config: dict, records: list[RunRecord],
                     generation_time: float, eval_time: float) -> dict:
    successful = sum(1 for r in records if r.eval_result is not None and r.eval_result.success)
    total_tokens = {"input": 0, "output": 0, "cached_prompt_tokens": 0, "total": 0}
    total_cost = {"input": 0.0, "output": 0.0, "cached_prompt": 0.0, "total": 0.0}
    for r in records:
        for k in total_tokens:
            total_tokens[k] += r.token_usage.get(k, 0)
        for k in total_cost:
            total_cost[k] += r.api_cost_usd.get(k, 0.0)
    return {
        "problem_name": problem_name,
        "model_name": model_name,
        "total_algorithms": len(records),
        "successful_algorithms": successful,
        "success_rate": successful / len(records) if records else 0.0,
        "generation_time": generation_time,
        "eval_time": eval_time,
        "total_time": generation_time + eval_time,
        "tokens": total_tokens,
        "cost_usd": total_cost,
        "config": config,
    }


def _save_logprobs_sidecar(npz_path: Path, records: list[RunRecord]) -> bool:
    """CSR-style sidecar: concatenated int32 tokens + float16 logprobs, an
    offsets array (offsets[i]:offsets[i+1] is record i's slice), and the
    algorithm_id each slice belongs to. Writes nothing if no record carries logprobs."""
    with_logprobs = [r for r in records if r.tokens is not None and r.logprobs is not None]
    if not with_logprobs:
        return False

    lengths = [len(r.tokens) for r in with_logprobs]
    offsets = np.zeros(len(with_logprobs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    tokens = np.concatenate([np.asarray(r.tokens, dtype=np.int32) for r in with_logprobs])
    logprobs = np.concatenate([np.asarray(r.logprobs, dtype=np.float16) for r in with_logprobs])
    algorithm_ids = np.asarray([r.algorithm_id for r in with_logprobs], dtype=np.int32)

    np.savez_compressed(npz_path, tokens=tokens, logprobs=logprobs, offsets=offsets, algorithm_ids=algorithm_ids)
    return True


def save_run(
        problem_name: str, model_name: str, config: dict, records: list[RunRecord],
        generation_time: float, eval_time: float, out_dir: str | Path, run_name: str | None = None,
) -> tuple[Path, Path | None]:
    """Writes <run_name>.json and, if any record carries tokens/logprobs, <run_name>.logprobs.npz
    Returns (json_path, npz_path_or_None)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        model_slug = model_name.replace("/", "-")
        run_name = f"results_{problem_name}_{len(records)}_{model_slug}"

    json_path = out_dir / f"{run_name}.json"
    npz_path = out_dir / f"{run_name}.logprobs.npz"

    payload = {
        "metadata": _build_metadata(problem_name, model_name, config, records, generation_time, eval_time),
        "results": [_record_to_json(r) for r in records],
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    wrote_sidecar = _save_logprobs_sidecar(npz_path, records)
    return json_path, (npz_path if wrote_sidecar else None)


def load_run(json_path: str | Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def record_from_json(d: dict, npz_path: Path) -> RunRecord:
    """Reconstructs a RunRecord from a saved results dict entry, re-attaching
    logprobs/tokens from the npz sidecar (if present) instead of leaving them
    unset -- used to resume a checkpointed run without silently losing them."""
    eval_result = EvaluationResult(success=d["success"], execution_time=d["execution_time"],
                                    error_message=d["error_message"], metrics=d["metrics"])
    tokens = logprobs = None
    if d.get("has_logprobs") and npz_path.exists():
        lp_data = load_logprobs(npz_path, d["algorithm_id"])
        if lp_data is not None:
            tokens, logprobs = lp_data
    return RunRecord(
        algorithm_id=d["algorithm_id"], algorithm_code=d["algorithm_code"],
        full_completion=d["full_completion"], prompt_used=d["prompt_used"],
        eval_result=eval_result, token_usage=d["token_usage"], api_cost_usd=d["api_cost_usd"],
        tokens=tokens, logprobs=logprobs,
    )


def load_logprobs(npz_path: str | Path, algorithm_id: int) -> tuple[list[int], list[float]] | None:
    """Returns (tokens, logprobs) for one algorithm_id, or None if it has no
    logprobs recorded (e.g. code extraction failed for that completion)."""
    data = np.load(npz_path)
    matches = np.nonzero(data["algorithm_ids"] == algorithm_id)[0]
    if len(matches) == 0:
        return None
    i = matches[0]
    offsets = data["offsets"]
    start, end = offsets[i], offsets[i + 1]
    return data["tokens"][start:end].tolist(), data["logprobs"][start:end].tolist()
