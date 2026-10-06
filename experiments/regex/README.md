# Regular expressions

Code for the paper's regex experiments, DME and baselines.

## Installing dependencies

From this directory, run `uv sync`. All commands below are run from this directory.

## Running

The regexes are from the [RegexEval](https://huggingface.co/datasets/s2e-lab/RegexEval) dataset. The paper's 20 test
regexes and 3 validation regexes are listed in `dme_regex/consts.py` (`TEST_IDS`, `VALIDATION_IDS`). To run DME on
regex `<ID>` with the paper's settings, run:

`uv run python -m dme_regex.run_dme --example_id <ID>`

The defaults are Qwen3-0.6B, a 16 token output limit, 100k samples per run, `beta=1000`, a batch size of 64, path
marginalisation, and the uniform prior (`--length_prior` enables the length-respecting one), and 10 runs per regex.
Results are cached under `results/`, and several workers can share them, e.g. in a SLURM array: each worker claims
unfinished runs using file locks.

The baselines are run with:
- (1,N) hill climbing: `uv run python -m dme_regex.run_dme --example_id <ID> --alpha 0 --beta 1000000 --no-marginalise`
- (1+N) hill climbing: `uv run python -m dme_regex.baselines.hill_climb --example_id <ID>`
- Best-of-N: `uv run python -m dme_regex.baselines.bon --example_id <ID>`, which generates 300k IID samples and estimates
  the per-run success rate by bootstrapping. Change `--benchmark_multiplier` to `k` to generated 100k x `k` samples.
- SCS: `uv run python -m dme_regex.baselines.scs --example_id <ID> --seed <SEED>`
- GEPA: `uv run python -m dme_regex.baselines.gepa --example_id <ID> --seed <SEED>`, which starts a vLLM server for the
  reflection model.
- GFlowNets: `uv run python -m dme_regex.baselines.gfn --example_id <ID> --seed <SEED>`
- RS-GRPO: `uv run python -m dme_regex.baselines.rs_grpo --example_id <ID> --seed <SEED>`

Each SCS/GEPA/GFlowNets/RS-GRPO invocation is a single run, the paper uses 10 seeds for quick methods and 3 for 
the others. Every script's defaults are the tuned hyperparameters from the paper's appendix, see `--help` for all options.
