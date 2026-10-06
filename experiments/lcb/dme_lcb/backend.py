"""Async vLLM generation and process-pool unit-test grading, so many questions' chains run concurrently."""
from __future__ import annotations

import asyncio
import inspect
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from itertools import count

from transformers import AutoConfig
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.outputs import CompletionOutput
from vllm.sampling_params import RequestOutputKind

from dme_lcb.grading import grade_one


class AsyncGenerator:
    def __init__(
            self, model: str, gpu_memory_utilization: float, tensor_parallel_size: int, dtype: str,
            max_out_tokens: int, enforce_eager: bool = False, prompt_buffer: int = 4096,
    ):
        engine_args = AsyncEngineArgs(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            enforce_eager=enforce_eager,
            max_model_len=min(
                max_out_tokens + prompt_buffer, AutoConfig.from_pretrained(model).max_position_embeddings,
            ),
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._counter = count()
        self._tokenizer = None

    async def generate(self, prompt: str, sampling_params: SamplingParams, tag: str) -> list[CompletionOutput]:
        # force full aggregation, not per-token streaming
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        request_id = f"{tag}-{next(self._counter)}"
        final_output = None
        async for request_output in self.engine.generate(prompt, sampling_params, request_id):
            final_output = request_output
        assert final_output is not None and len(final_output.outputs) == sampling_params.n, (
            f"expected {sampling_params.n} completions, got {len(final_output.outputs) if final_output else 0}"
        )
        return final_output.outputs

    async def get_tokenizer(self):
        # AsyncLLMEngine.get_tokenizer() is a coroutine in some vLLM versions and synchronous in others
        if self._tokenizer is None:
            result = self.engine.get_tokenizer()
            self._tokenizer = await result if inspect.isawaitable(result) else result
        return self._tokenizer


def _init_grader_worker(workdir: str) -> None:
    # generated programs sometimes write files to their cwd, keep them out of the caller's directory
    os.chdir(workdir)


class AsyncGrader:
    def __init__(self, num_workers: int, fast: bool = False):
        self._workdir = tempfile.TemporaryDirectory(prefix="dme_lcb_grading_")
        # spawn, not fork: avoids workers inheriting vLLM's already-loaded memory footprint
        self.pool = ProcessPoolExecutor(
            max_workers=num_workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_grader_worker, initargs=(self._workdir.name,),
        )
        # fast=True stops at the first failing test instead of scoring every test case
        self.fast = fast

    async def grade(self, sample: dict, codes: list[str], total_tests: int, timeout: int):
        loop = asyncio.get_running_loop()
        futures = [
            loop.run_in_executor(self.pool, grade_one, sample, code, total_tests, timeout, self.fast)
            for code in codes
        ]
        pairs = await asyncio.gather(*futures)
        accuracies = [acc for acc, _meta in pairs]
        metadatas = [meta for _acc, meta in pairs]
        return accuracies, metadatas

    def shutdown(self):
        self.pool.shutdown(wait=True)
        self._workdir.cleanup()
