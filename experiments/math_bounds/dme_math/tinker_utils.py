import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path

import tinker
from tinker import types

# Tinker's backend rejects a single request with num_samples > 128
MAX_SAMPLES_PER_REQUEST = 128

# $/M tokens (checked 2026-09 via https://tinker-docs.thinkingmachines.ai/tinker/models/).
# Client-side cost estimate only -- ignores any prefix-caching discount
# Base-context entries only, no long context pricing
MODEL_PRICES = {
    "openai/gpt-oss-20b": {"prefill": 0.18, "sample": 0.45},
    "openai/gpt-oss-120b": {"prefill": 0.33, "sample": 0.84},
    "Qwen/Qwen3-8B": {"prefill": 0.195, "sample": 0.60},
    "Qwen/Qwen3.5-4B": {"prefill": 0.33, "sample": 1.005},
    "Qwen/Qwen3.5-9B": {"prefill": 0.66, "sample": 1.995},
    "Qwen/Qwen3.5-9B-Base": {"prefill": 0.66, "sample": 1.995},
    "Qwen/Qwen3.5-35B-A3B-Base": {"prefill": 0.54, "sample": 1.335},
    "Qwen/Qwen3.5-397B-A17B": {"prefill": 3.00, "sample": 7.50},
    "Qwen/Qwen3.6-27B": {"prefill": 1.86, "sample": 5.595},
    "Qwen/Qwen3.6-35B-A3B": {"prefill": 0.54, "sample": 1.335},
    "Qwen/Qwen3.8-27B": {"prefill": 1.86, "sample": 5.595},
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {"prefill": 0.39, "sample": 0.99},
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16": {"prefill": 0.39, "sample": 0.99},
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": {"prefill": 1.14, "sample": 2.88},
    "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16": {"prefill": 4.98, "sample": 12.45},
    "moonshotai/Kimi-K2.6": {"prefill": 2.205, "sample": 5.49},
    "deepseek-ai/DeepSeek-V3.1": {"prefill": 1.695, "sample": 4.215},
    "thinkingmachines/Inkling": {"prefill": 3.74, "sample": 9.36},
    "thinkingmachines/Inkling-Small": {"prefill": 1.16, "sample": 2.88},
}

CODE_FENCE_RE = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)
FINAL_CHANNEL_MARKER = "<|channel|>final<|message|>"


@dataclass
class Completion:
    tokens: list[int]
    logprobs: list[float] | None
    decoded: str
    code: str | None
    stop_reason: str

    @property
    def cumulative_logprob(self) -> float | None:
        return sum(self.logprobs) if self.logprobs else None


def extract_final_channel_text(decoded: str) -> str:
    """Extract only final answers after end of reasoning (analysis channel for GPT OSS 20B)"""
    idx = decoded.rfind(FINAL_CHANNEL_MARKER)
    if idx == -1:
        return decoded
    tail = decoded[idx + len(FINAL_CHANNEL_MARKER):]
    return tail.split("<|end|>")[0].split("<|return|>")[0]


def extract_code(text: str) -> str | None:
    """Pull the last fenced code block out of the model's final-channel text."""
    matches = CODE_FENCE_RE.findall(text)
    return matches[-1].strip() if matches else None


async def get_sampling_client(model_name: str):
    """Returns (sampling_client, tokenizer) for the given base model."""
    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(base_model=model_name)
    tokenizer = sampling_client.get_tokenizer()
    return sampling_client, tokenizer


def build_chat_prompt(tokenizer, instruction: str, reasoning_effort: str) -> list[int]:
    messages = [{"role": "user", "content": instruction}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=False,
        reasoning_effort=reasoning_effort,
    )


async def sample_completions(
        sampling_client, tokenizer, instruction: str, *, num_samples: int, temperature: float,
        max_tokens: int, reasoning_effort: str,
) -> tuple[list[Completion], int]:
    """Returns (completions, prompt_n_tokens)."""
    prompt_tokens = build_chat_prompt(tokenizer, instruction, reasoning_effort)
    prompt = types.ModelInput.from_ints(prompt_tokens)
    sampling_params = types.SamplingParams(max_tokens=max_tokens, temperature=temperature)

    chunk_sizes = []
    remaining = num_samples
    while remaining > 0:
        chunk_sizes.append(min(MAX_SAMPLES_PER_REQUEST, remaining))
        remaining -= chunk_sizes[-1]

    results = await asyncio.gather(*(
        sampling_client.sample_async(prompt=prompt, num_samples=n, sampling_params=sampling_params)
        for n in chunk_sizes
    ))

    completions = []
    for result in results:
        for seq in result.sequences:
            decoded = tokenizer.decode(seq.tokens)
            final_text = extract_final_channel_text(decoded)
            completions.append(Completion(
                tokens=seq.tokens,
                logprobs=seq.logprobs,
                decoded=decoded,
                code=extract_code(final_text),
                stop_reason=seq.stop_reason,
            ))
    return completions, len(prompt_tokens)


def estimate_cost(model_name: str, prefill_tokens: int, sample_tokens: int) -> float:
    prices = MODEL_PRICES[model_name]
    return prefill_tokens / 1e6 * prices["prefill"] + sample_tokens / 1e6 * prices["sample"]


def gen_cache_path(
        output_dir: str | Path, problem: str, num_samples: int, temperature: float,
        max_tokens: int, reasoning_effort: str, model_name: str,
) -> Path:
    model_slug = model_name.replace("/", "-")
    fname = f"gencache_{problem}_{num_samples}_{temperature:g}_{max_tokens}_{reasoning_effort}_{model_slug}.json"
    return Path(output_dir) / "gen_cache" / fname


def save_gen_cache(path: Path, completions: list[Completion], prompt_n_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "prompt_n_tokens": prompt_n_tokens,
        "completions": [
            {"tokens": c.tokens, "logprobs": c.logprobs, "decoded": c.decoded,
             "code": c.code, "stop_reason": c.stop_reason}
            for c in completions
        ],
    }
    path.write_text(json.dumps(data))


def load_gen_cache(path: Path) -> tuple[list[Completion], int]:
    data = json.loads(path.read_text())
    completions = [Completion(**c) for c in data["completions"]]
    return completions, data["prompt_n_tokens"]
