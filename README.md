# Code for "Distribution Matching Evolutionary Algorithms for Rare Event Sampling"

[![arXiv](https://img.shields.io/badge/arXiv-2610.03833-b31b1b.svg)](https://arxiv.org/abs/2610.03833)

This is the code for ["Distribution Matching Evolutionary Algorithms for Rare Event Sampling"](https://arxiv.org/abs/2610.03833). The
repo contains:
1. A small library implementing DME, `src/dme` using vLLM
2. Examples, `examples/`, showing how to use DME on your own problems.
3. The paper's three experiments and their baselines: regular expressions (`experiments/regex`), LiveCodeBench
   (`experiments/lcb`), and the math bounds (`experiments/math_bounds`).

## Quickstart

### Installing dependencies

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Each experiment is its own uv project with its own
environment. For example, to install the environment for the regex:
```
cd experiments/regex   # or experiments/lcb, experiments/math_bounds
uv sync
```

The math bounds specifically use the [Tinker](https://thinkingmachines.ai/tinker/) API, with no local GPU required.
Experiment commands are run from the experiment's directory with `uv run`.

In case vLLM hangs, set `VLLM_WORKER_MULTIPROC_METHOD=spawn` for any vLLM-based experiments.

### Using DME on your own problem

At each step DME samples a batch of candidates conditioned on the current state (code, regex, etc.) and picks the 
next state with probability proportional to `R(x)^beta / p(x|state)`. In code:
```python
from dme import dme_select

next_idx = dme_select(log_rewards, logprobs, beta)  # log R(x_i) and log p(x_i|state) per candidate
```

`examples/` demonstrates the full loop over two toy problems:
```
uv run --project experiments/regex python examples/regex_example.py
uv run --project experiments/regex python examples/coding_example.py --length-prior
```
To adapt it to your own problem, you need to give a prompt builder (current state -> prompt), an extractor (completion
-> answer) and a batched log-reward function. Two extractors are provided in `dme.extract`: `strip_whitespace`,
which takes the stripped completion as the answer, as in the regex experiments, and `last_code_block`, which takes the
last fenced code block as the answer, as in the LiveCodeBench and math bounds experiments. 

`--length-prior` enables the length-respecting uniform prior.

### Running the paper's experiments

Each experiment's directory has a README with the commands and settings to reproduce the paper's results:
- [Regular expressions](experiments/regex/README.md): DME, (1,N), (1+N), best-of-N, SCS, GEPA, GFlowNets and RS-GRPO
  on RegexEval.
- [LiveCodeBench](experiments/lcb/README.md): DME and the (1,N)/(1+N) evolution strategies.
- [Math bounds](experiments/math_bounds/README.md): DME, best-of-N and SCS on AlphaEvolve problems, plus the
  probability of improvement calculations.

### Third-party code

`experiments/lcb/lcb_runner` contains a subset of [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)'s
`lcb_runner`, Copyright (c) 2024 LiveCodeBench, used under the MIT License, with modifications listed at the top of each file. The math bounds
problems and their best-of-N (IID RS) and SCS baselines are adapted from the codebase of
[Simple Baselines are Competitive with Code Evolution](https://arxiv.org/abs/2602.16805)
([code](https://github.com/YonatanGideoni/code_evo_simple_baselines)); the problem statements follow
[AlphaEvolve](https://arxiv.org/abs/2506.13131).

## Bibtex
```
@article{gideoni2026dme,
  title={Distribution Matching Evolutionary Algorithms for Rare Event Sampling},
  author={Gideoni, Yonatan and Gal, Yarin},
  journal={arXiv preprint arXiv:2610.03833},
  year={2026}
}
```

If you use the LiveCodeBench experiments, please also cite:
```
@article{jain2024livecodebench,
  author    = {Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica},
  title     = {LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  year      = {2024},
  journal   = {arXiv preprint},
}
```
