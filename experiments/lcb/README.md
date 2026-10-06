# LiveCodeBench

Code for the LiveCodeBench experiments, DME and the (1,N)/(1+N) evolution strategies.

## Installing dependencies

From this directory, run `uv sync`. All commands below are run from this directory.

## Running

To run DME and the two evolution strategy baselines on the paper's 300 question subset of
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench), run:

`uv run python -m dme_lcb.run_dme --length_prior --num_process_evaluate <N-WORKERS> --parallel_slice <N-QUESTIONS>`

`uv run python -m dme_lcb.run_es --num_process_evaluate <N-WORKERS> --parallel_slice <N-QUESTIONS>`

Where:

- `<N-WORKERS>` is the number of CPU processes used to run the generated programs' unit tests.
- `<N-QUESTIONS>` is the number of questions processed concurrently. The paper's setup used 24 workers and 50
  questions on an A100.

The defaults are Qwen2.5-Coder-1.5B-Instruct, 5 runs per question, a budget of 300 programs per
run, a batch size of 8, and for DME `beta=1000`, except that the length-respecting prior is off by default.
The question subset is given by `--question_ids_file`, which defaults to `question_ids/random_sample_300.txt`. For
shorter, quick experiments use `--question_ids_file question_ids/never_solved_sample_100.txt`, which is a distinct
LCB subset of 100 questions that were empirically never solved by best-of-N sampling given a 300 generated
programs budget.

Results are cached per question under `results/` (or `$DME_LCB_RESULTS_DIR`).

Generated programs are run in subprocesses with LiveCodeBench's reliability guard, which is not a security sandbox.
We recommend running these experiments in a proper container.

## Third-party code

`lcb_runner/` contains a subset of [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)'s `lcb_runner`,
Copyright (c) 2024 LiveCodeBench, used under the MIT License (see `lcb_runner/LICENSE`), with modifications listed at
the top of each file.
