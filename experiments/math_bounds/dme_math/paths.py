import os
from pathlib import Path

from dotenv import load_dotenv

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("DME_MATH_RUNS_DIR", EXPERIMENT_DIR / "runs"))
PLOTS_DIR = Path(os.environ.get("DME_MATH_PLOTS_DIR", EXPERIMENT_DIR / "plots"))


def load_env() -> None:
    """Loads TINKER_API_KEY etc. from experiments/math_bounds/.env, see .env.example."""
    load_dotenv(EXPERIMENT_DIR / ".env")
