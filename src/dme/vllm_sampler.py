"""
A generic DME sampler on top of vLLM: n_chains independent chains, each step samples batch_size completions per chain
conditioned on the chain's state (via a user-given prompt builder), scores them, and resamples the next state with
the SIR weights. vLLM is lazily imported so the rest of the package doesn't depend on it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from dme.core import dme_select, log_vocab_size
from dme.extract import Extraction
from dme.marginalise import marginalise


@dataclass
class Candidate:
    text: str  # raw completion
    answer: Any  # extracted answer, the candidate's chain state if selected
    log_reward: float
    logprob: float  # log p(x'|x), restricted to the answer's span if marginalising
    num_tokens: int | None  # |x'| for the length-respecting prior, None if the prior isn't used


class DME:
    """
    DME with a vLLM model as the proposer.

    Args:
        llm: a vllm.LLM.
        extract: raw completion text -> Extraction, e.g. dme.extract.strip_whitespace or dme.extract.last_code_block.
        log_reward: batched answers -> log R per answer, -inf for invalid answers. Batched so rewards can be computed
            in parallel and in the same numerical way as a vectorised implementation.
        beta: inverse temperature of the target distribution pi(x) ~ R(x)^beta.
        alpha: weight on the proposal's logprob, -1 for DME. alpha=0 with a large beta gives (1,N) hill climbing.
        marginalise: path marginalisation, use only the logprob of the tokens covering the extracted answer instead
            of the whole completion's.
        length_prior: use the length-respecting uniform prior, which adds a -|x'| log V term to the log weights.
        include_straddling_start: how the answer's character span is mapped to tokens, see dme.marginalise.token_span.
        seed: seeds the resampling step only; vLLM's sampling is seeded separately (vllm.LLM(seed=...)).
    """

    def __init__(
            self,
            llm,
            extract: Callable[[str], Extraction],
            log_reward: Callable[[list[Any]], np.ndarray],
            beta: float,
            batch_size: int,
            n_chains: int = 1,
            alpha: float = -1.0,
            marginalise: bool = True,
            length_prior: bool = False,
            include_straddling_start: bool = False,
            max_tokens: int = 1024,
            seed: int | None = None,
    ):
        from vllm import SamplingParams

        self.llm = llm
        self.extract = extract
        self.log_reward = log_reward
        self.beta = beta
        self.alpha = alpha
        self.batch_size = batch_size
        self.n_chains = n_chains
        self.marginalise = marginalise
        self.length_prior = length_prior
        self.include_straddling_start = include_straddling_start
        # temperature/top_p must stay 1, the importance weights need the model's raw, unmodified logprobs
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens, top_p=1.0, temperature=1.0, n=batch_size, logprobs=0,
        )
        self.tokenizer = llm.get_tokenizer()
        self.log_vocab_size = log_vocab_size(self.tokenizer) if length_prior else None
        self.rng = np.random.default_rng(seed) if seed is not None else None
        self.chains: list[Any] = []
        self.init_chains()

    def init_chains(self, initial_state: Any = "") -> None:
        self.chains = [initial_state] * self.n_chains

    def generate(self, prompts: Sequence[str]) -> list[list[Candidate]]:
        """batch_size scored candidates per prompt."""
        results = self.llm.generate(list(prompts), sampling_params=self.sampling_params, use_tqdm=False)
        all_candidates = []
        for result in results:
            extractions = [self.extract(out.text) for out in result.outputs]
            log_rewards = self.log_reward([e.answer for e in extractions])
            candidates = []
            for out, extraction, log_r in zip(result.outputs, extractions, log_rewards):
                logprob, num_tokens = self._logprob_and_num_tokens(out, extraction)
                candidates.append(Candidate(
                    text=out.text, answer=extraction.answer, log_reward=log_r, logprob=logprob,
                    num_tokens=num_tokens if self.length_prior else None,
                ))
            all_candidates.append(candidates)
        return all_candidates

    def select(self, candidates: list[Candidate]) -> int:
        num_tokens = [c.num_tokens for c in candidates] if self.length_prior else None
        return dme_select(
            np.array([c.log_reward for c in candidates]), np.array([c.logprob for c in candidates]),
            self.beta, self.alpha, num_tokens, self.log_vocab_size, self.rng,
        )

    def step(self, build_prompt: Callable[[Any], str]) -> list[list[Candidate]]:
        """One DME step across all chains, build_prompt maps a chain's state to its prompt."""
        chain_candidates = self.generate([build_prompt(state) for state in self.chains])
        for c, candidates in enumerate(chain_candidates):
            self.chains[c] = candidates[self.select(candidates)].answer
        return chain_candidates

    def _logprob_and_num_tokens(self, out, extraction: Extraction) -> tuple[float, int]:
        if not self.marginalise:
            return out.cumulative_logprob, len(out.token_ids)
        token_ids = list(out.token_ids)
        per_token_logprobs = [out.logprobs[i][tid].logprob for i, tid in enumerate(token_ids)]
        return marginalise(
            token_ids, per_token_logprobs, self.tokenizer, extraction.span,
            include_straddling_start=self.include_straddling_start, max_len=len(out.text),
        )
