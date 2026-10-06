# Math bounds

Code for the paper's math bounds experiments, DME, best-of-N and SCS.

The problems (prompts, verifiers and evaluation), the best-of-N (IID RS) baseline and the SCS baseline are adapted
from the codebase of [Simple Baselines are Competitive with Code Evolution](https://arxiv.org/abs/2602.16805)
([code](https://github.com/YonatanGideoni/code_evo_simple_baselines)), with sampling moved to GPT-OSS 20B via Tinker.
The problem statements follow [AlphaEvolve](https://arxiv.org/abs/2506.13131).

## Installing dependencies

From this directory, run `uv sync`. All commands below are run from this directory.

## Running

The math bounds experiments sample from GPT-OSS 20B via Tinker. Copy `.env.example` to `.env` and set your
`TINKER_API_KEY` in it. The problems are in `dme_math/problems/`; only the problem's name has to be given, so for
`dme_math/problems/short/geometry/pack_circ26` it's enough to specify `pack_circ26`. The paper's problems are
`pack_circ26 maxmin_dist_ratio kissing_11d heilbronn_triangles uncert_ineq first_autocorr sec_autocorr sumdiff_sets
min_overlap`, with `third_autocorr` used for tuning.

To run DME on problem `<PROBLEM>`, run:

`uv run python -m dme_math.run_dme --problem <PROBLEM> --num-workers <N-WORKERS>`

The defaults are the paper's settings: 4 chains, 63 steps, 20 programs per chain per step (5040 programs in total),
`beta=1e6` and the uniform prior. Runs are checkpointed after every step and resume when rerun with the same
arguments. `<N-WORKERS>` is the number of CPU processes used to evaluate the generated programs.

The baselines are run with:
- Best-of-N: `uv run python -m dme_math.bon --problem <PROBLEM> --num-samples 5000 --num-workers <N-WORKERS>`
- SCS: `uv run python -m dme_math.scs --problem <PROBLEM> --trial <TRIAL> --num-workers <N-WORKERS>`, where each trial
  generates 10 generations of 20 programs.

To compute the probability of improvement given a 5k program budget, run DME with more chains (not 4) via 
`--n-chains X`, and similarly generate more samples/trials for Best-of-N and SCS respectively. Then run:

`uv run python -m dme_math.prob_improvement`

It uses every run in `runs/`: each DME run file is split into its chains and DME runs of `--dme-chains-per-run`
chains (default 4) are compared, all best-of-N files are pooled, and all SCS trials are used.

Outputs are saved in `runs/` and `plots/`.
