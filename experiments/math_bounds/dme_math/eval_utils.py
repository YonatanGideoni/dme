import concurrent.futures

from dme_math.problems.problem_utils import BaseEvaluator, BaseProblem, EvaluationResult


def _evaluate_one(idx_and_code: tuple[int, str | None], problem: BaseProblem, evaluator: BaseEvaluator):
    """Runs in a worker process. `problem`/`evaluator` are pickled once per task by ProcessPoolExecutor."""
    idx, code = idx_and_code
    if code is None:
        return idx, None
    return idx, evaluator.evaluate_algorithm(code, problem)


def evaluate_in_parallel(
        problem: BaseProblem, evaluator: BaseEvaluator, codes: list[str | None], num_workers: int,
) -> list[EvaluationResult | None]:
    results: list[EvaluationResult | None] = [None] * len(codes)
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as pool:
        future_to_idx = {
            pool.submit(_evaluate_one, (i, code), problem, evaluator): i
            for i, code in enumerate(codes)
        }
        for j, future in enumerate(concurrent.futures.as_completed(future_to_idx), 1):
            idx, result = future.result()
            results[idx] = result
            if j % 5 == 0 or j == len(codes):
                print(f"  evaluated {j}/{len(codes)}", flush=True)
    return results
