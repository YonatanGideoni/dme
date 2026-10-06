# Adapted from LiveCodeBench (https://github.com/LiveCodeBench/LiveCodeBench), upstream commit 28fef95.
# Copyright (c) 2024 LiveCodeBench, MIT License, see LICENSE in this directory.
# Modifications: added CodeQwenInstruct and DeepSeekR1 self-repair templates, get_check_prompt tolerates sparse/unknown error metadata instead of raising, Anthropic prompt constants are defined inline instead of imported from the anthropic package.

import json

# values of anthropic.HUMAN_PROMPT / anthropic.AI_PROMPT, only used for Claude-style prompts
HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"

from lcb_runner.lm_styles import LMStyle


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once."

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you are helping a user correct a error program for code competition. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the entire executable program. You must put the entire fixed executable program within code delimiters."

    SYSTEM_MESSAGE_MAGIC = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n"

    SYSTEM_MESSAGE_WIZARD = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    SYSTEM_MESSAGE_PHIND = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. You must put the entired fixed program within code delimiters only for once., for example:
```python
# YOUR CODE HERE
```"""

    # Mirrors code_generation.py's SYSTEM_MESSAGE_CODEQWEN wrapper (a bare
    # "helpful assistant" message, not a repair-specific one -- this repo's
    # existing convention for this model style).
    SYSTEM_MESSAGE_CODEQWEN = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    # Mirrors code_generation.py's SYSTEM_MESSAGE_DEEPSEEK_R1 verbatim.
    SYSTEM_MESSAGE_DEEPSEEK_R1 = (
        "<｜begin▁of▁sentence｜>A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>"
    )

    FORMATTING_REPEAT = f"First reason about the code providing a textual explanation of what is wrong with the code and then generate a fixed of the program enclosed code delimiters."

    FORMATTING_MESSAGE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


# def truncate_io(io):
#     if len(str(io)) > 200:
#         io = str(io)[:200] + "...."
#     return io


def get_check_prompt(question: str, result, metadata):
    ## assumes i/o examples are already truncated!
    ## less pressure on storing 10 MB json because on a single large input-output pair
    # result_by_test_case = result
    # assert len(metadata) == 1, f"metadata = {metadata}"
    # metadata = metadata[0]
    metadata = json.loads(metadata)
    if "error_code" not in metadata:
        return ""
    # testing_util.run_test() has an outer catch-all exception handler (used
    # for e.g. a ValueError while parsing malformed input) that also reports
    # error_code=-4 but only sets error_message, not error/inputs/expected --
    # a different, sparser shape than the "normal" per-test failure metadata
    # below. Fall back gracefully instead of KeyError'ing on it.
    error = metadata.get("error", metadata.get("error_message", "unknown error"))
    inputs = metadata.get("inputs", "N/A")
    expected = metadata.get("expected", "N/A")
    if metadata["error_code"] == -1:
        # time limit exceeded
        message = f"The above code is incorrect and got the following compilation error.\n{error}"
    elif metadata["error_code"] == -2:
        # wrong answer
        message = f"The above code is incorrect and got a wrong answer.\nInput: {inputs}\nGenerated Output: {metadata.get('output', 'N/A')}\nExpected: {expected}"
    elif metadata["error_code"] == -3:
        # time limit exceeded
        message = f"The above code is incorrect and got time limit exceeded.\n{error}\nInput: {inputs}\nExpected: {expected}"
    elif metadata["error_code"] == -4:
        # runtime error
        message = f"The above code is incorrect and got a runtime error.\nInput: {inputs}\nExpected: {expected}\n{error}"
    else:
        # e.g. error_code=-5 (TestRunnerError), raised by
        # evaluate_generations_by_problem's own outer exception handler for
        # unexpected harness failures (observed: MemoryError while parsing a
        # large problem's test data). Previously raised NotImplementedError
        # here and crashed the whole process instead of just failing this
        # one candidate -- fall back gracefully like the other branches.
        message = f"The above code is incorrect and failed with an unexpected error.\n{error}"
    return message


def get_generic_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Question:\n{question}\n\n"
    prompt += f"### Answer:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata) + "\n"
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_cllama_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Question\n{question}\n\n"
    prompt += f"### Answer\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_deepseekcode_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Instruction: You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question}\n\n"
    prompt += f"### Response:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_magicoder_question_template_answer(question: str, code, result, metadata):
    prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question}\n\n"
    prompt += f"@@ Response \n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_mixtral_question_template_answer(question: str, code, result, metadata):
    prompt = f"Question:\n"
    prompt += f"{question}\n\n"
    prompt += f"Answer:\n\n"
    prompt += f"```python\n\n{code}\n``\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_wizard_question_template_answer(question: str, code, result, metadata):
    prompt = f"""### Instruction: You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once., for example:
    ```python
    # YOUR CODE HERE
    ```
"""
    prompt += f"{question}\n\n"
    prompt += f"### Response:```python\n\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_phind_question_template_answer(question: str, code, result, metadata):
    prompt = f"{question}\n\n"
    prompt += f"```python\n{code}\n``` \n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"\n\n### Assistant"
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt

def get_codeqwen_question_template_answer(question: str, code, result, metadata):
    prompt = f"Question: {question}\n\n"
    prompt += f"```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata) + "\n\n"
    prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += f"```python\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    return prompt


def get_deepseek_r1_question_template_answer(question: str, code, result, metadata):
    prompt = f"Question: {question}\n\n"
    prompt += f"```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata) + "\n\n"
    prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"<｜Assistant｜>"
    return prompt


def get_qwen_question_template_answer(question: str, code, result, metadata):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "abacusai/Dracarys-72B-Instruct", padding_side="left", use_fast=False
    )
    prompt = f"""### Instruction: You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once., for example:
    ```python
    # YOUR CODE HERE
    ```\n\n
"""
    prompt += f"Question:\n{question}\n\n"
    prompt += f"```python\n{code}\n``` \n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"\n\n### Assistant"
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"

    messages = [
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        padding=False,
    )
    return prompt

def format_prompt_self_repair(
    question: str, LanguageModelStyle: LMStyle, code, result, metadata
) -> str:
    if result:
        # The code is accepted, no need to change anything.
        return ""
    if LanguageModelStyle == LMStyle.OpenAIChat:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                )
                + "\n\n"
                + PromptConstants.FORMATTING_REPEAT,
            },
        ]
        return chat_messages
    if LanguageModelStyle == LMStyle.LLaMa3:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ),
            },
        ]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    elif LanguageModelStyle == LMStyle.Claude:
        prompt = f"{HUMAN_PROMPT}\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n{get_generic_question_template_answer(question, code, result, metadata).rstrip()}\n{AI_PROMPT}"
        return prompt
    elif LanguageModelStyle == LMStyle.Claude3:
        system = PromptConstants.SYSTEM_MESSAGE_GENERIC
        prompt = [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ).rstrip(),
            }
        ]
        return system, prompt
    elif LanguageModelStyle == LMStyle.MistralWeb:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question, code, result, metadata),
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.Gemini:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{get_generic_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.StarCoderInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{get_generic_question_template_answer(question, code, result,metadata)}"
        return prompt

    elif LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK}\n\n{get_deepseekcode_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.CodeQwenInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN}\n\n{get_codeqwen_question_template_answer(question, code, result, metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.DeepSeekR1:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK_R1}"
        prompt += get_deepseek_r1_question_template_answer(question, code, result, metadata)
        return prompt
    elif LanguageModelStyle == LMStyle.CodeLLaMaInstruct:
        prompt = f"[INST] <<SYS>>\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n<</SYS>>\n\n{get_cllama_question_template_answer(question, code, result,metadata)}\n[/INST]"
        return prompt
    elif LanguageModelStyle == LMStyle.MagiCoder:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_MAGIC}\n{get_magicoder_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.WizardCoder:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_WIZARD}\n\n{get_wizard_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.Phind:
        prompt = f"### System Prompt\n\n{PromptConstants.SYSTEM_MESSAGE_PHIND}\n\n### User Message\n\n{get_phind_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.DracarysQwen:
        prompt = f"{get_qwen_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.DracarysLlama:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ),
            },
        ]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "abacusai/Dracarys-Llama-3.1-70B-Instruct", padding_side="right", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    if LanguageModelStyle == LMStyle.Eurusx:
        prompt = "[INST] Write Python code to solve the task:\n"
        prompt += f"{get_wizard_question_template_answer(question, code, result,metadata)}"
        prompt += "[/INST]"
        return prompt
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle} not implemented"
        )


def extract_code(model_output: str, lmstyle: LMStyle):
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMa:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])
