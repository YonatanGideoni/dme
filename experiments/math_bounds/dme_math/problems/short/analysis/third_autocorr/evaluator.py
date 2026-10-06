import numpy as np

from dme_math.problems.problem_utils import BaseEvaluator, ProblemConfig, helper


class ThirdAutocorrEvaluator(BaseEvaluator):

    def __init__(self, config: ProblemConfig):
        self.config = config
        self.params = config["problem_parameters"]

    def get_helper_prelude(self, problem) -> str:
        return "import numpy as np"

    # ---------------- helpers ----------------

    @staticmethod
    @helper
    def compute_upper_bound(step_heights: np.ndarray) -> float:
        convolution = np.convolve(step_heights, step_heights)
        return abs(2 * len(step_heights) * np.max(convolution) / (np.sum(step_heights) ** 2))

    # ---------------- BaseEvaluator hooks ----------------

    def parse_output(self, raw_output, problem):
        return np.asarray(raw_output, dtype=float).reshape(-1)

    def validate_output(self, step_heights: np.ndarray, problem) -> None:
        if step_heights is None:
            raise ValueError("Output is None")

        self.assert_all_finite(step_heights, name="step_heights")
        if step_heights.ndim != 1 or step_heights.size == 0:
            raise ValueError(f"Expected a 1D non-empty array, got shape {step_heights.shape}")
        if np.sum(step_heights) == 0:
            raise ValueError("Sum of step heights is zero (division by zero in C3)")

    def compute_metrics(self, step_heights: np.ndarray, problem):
        ub = self.compute_upper_bound(step_heights)
        return {
            "upper_bound": float(ub),
            "step_heights": step_heights.tolist(),
        }

    def default_failure_metrics(self, problem):
        return {"upper_bound": float("inf")}
