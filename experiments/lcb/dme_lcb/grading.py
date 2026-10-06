"""
Unit-test grading for LiveCodeBench candidates. Unlike LCB's default grading, which stops at the first failing test,
every test case is run and scored independently, so a candidate's accuracy is the fraction of tests it passes.

Candidates run in a forked subprocess under LCB's reliability_guard, additionally hardened here with a per-file size
cap and blocked outbound network access. This is not a security sandbox, run untrusted generations in a container.
"""
from __future__ import annotations

import faulthandler
import json
import multiprocessing
import os
import resource
import signal
import socket
import time
from functools import lru_cache

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem, load_code_generation_dataset
from lcb_runner.evaluation import testing_util

_MAX_FILE_SIZE_BYTES = 512 * 1024 ** 2  # 512MB per file

_ORIGINAL_SOCKET_CTOR = socket.socket
_IP_FAMILIES = {socket.AF_INET, socket.AF_INET6}


def _guarded_socket_ctor(family=socket.AF_INET, type=socket.SOCK_STREAM, *args, **kwargs):
    if family in _IP_FAMILIES:
        raise OSError("network access is disabled in the LCB eval sandbox")
    return _ORIGINAL_SOCKET_CTOR(family, type, *args, **kwargs)


def _blocked_create_connection(*_args, **_kwargs):
    raise OSError("network access is disabled in the LCB eval sandbox")


def _harden_reliability_guard() -> None:
    """Patch reliability_guard to also cap per-file size and block outbound sockets."""
    original_guard = testing_util.reliability_guard

    def guarded(maximum_memory_bytes=None):
        # Must run before original_guard(), which blanks out sys.modules["resource"].
        resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_FILE_SIZE_BYTES, _MAX_FILE_SIZE_BYTES))
        original_guard(maximum_memory_bytes=maximum_memory_bytes)
        socket.socket = _guarded_socket_ctor
        socket.create_connection = _blocked_create_connection

    testing_util.reliability_guard = guarded


# applied at import time, so it also takes effect in spawned grader pool workers, which import this module
_harden_reliability_guard()


@lru_cache(maxsize=1)
def load_problems_by_id() -> dict[str, CodeGenerationProblem]:
    problems = load_code_generation_dataset(release_version="release_latest")
    return {p.question_id: p for p in problems}


def grade_call_based_independent(
        code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int,
) -> tuple[list, dict] | None:
    # basically copied from LCB core, modified to go over all tests, not stopping at first failure
    code = testing_util.import_string + "\n\n" + code
    compiled_sol = testing_util.compile_code(code, timeout)
    if compiled_sol is None:
        return None

    method = testing_util.get_function(compiled_sol, fn_name)
    if method is None:
        return None

    all_inputs = [
        [json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs
    ]
    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0.0
    all_results = []
    first_failure_metadata: dict | None = None
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            with testing_util.Capturing():  # capture candidate prints, don't pollute actual stdout
                prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out
            all_results.append(tmp_result)

            if not tmp_result and first_failure_metadata is None:
                first_failure_metadata = {
                    "output": testing_util.truncatefn(prediction),
                    "inputs": testing_util.truncatefn(gt_inp),
                    "expected": testing_util.truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                error_code, error_message = -3, "Time Limit Exceeded"
            else:
                all_results.append(-4)
                error_code, error_message = -4, "Runtime Error"
            if first_failure_metadata is None:
                first_failure_metadata = {
                    "error": repr(e),
                    "error_code": error_code,
                    "error_message": error_message,
                    "inputs": testing_util.truncatefn(gt_inp),
                    "expected": testing_util.truncatefn(gt_out),
                }
        finally:
            signal.alarm(0)
            faulthandler.disable()

    if first_failure_metadata is not None:
        return all_results, first_failure_metadata
    return all_results, {"execution time": total_execution}


def grade_stdio_independent(
        code: str, all_inputs: list, all_outputs: list, timeout: int,
) -> tuple[list, dict] | None:
    code = testing_util.clean_if_name(code)
    code = testing_util.make_function(code)
    compiled_sol = testing_util.compile_code(code, timeout)
    if compiled_sol is None:
        return None

    method = testing_util.get_function(compiled_sol, "wrapped_function")
    if method is None:
        return None

    all_results = []
    total_execution_time = 0.0
    first_failure_metadata: dict | None = None
    for gt_inp, gt_out in zip(all_inputs, all_outputs):
        signal.alarm(timeout)
        faulthandler.enable()
        with testing_util.Capturing() as captured_output:
            try:
                start = time.time()
                testing_util.call_method(method, gt_inp)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    error_code, error_message = -3, "Time Limit Exceeded"
                else:
                    all_results.append(-4)
                    error_code, error_message = -4, "Runtime Error"
                if first_failure_metadata is None:
                    first_failure_metadata = {
                        "error": repr(e),
                        "error_code": error_code,
                        "error_message": error_message,
                        "inputs": testing_util.truncatefn(gt_inp),
                        "expected": testing_util.truncatefn(gt_out),
                    }
                continue
            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]
        stripped_prediction_lines = testing_util.get_stripped_lines(prediction)
        stripped_gt_out_lines = testing_util.get_stripped_lines(gt_out)

        wa_args = {
            "output": testing_util.truncatefn(prediction),
            "inputs": testing_util.truncatefn(gt_inp),
            "expected": testing_util.truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            wa_args["error_message"] = "Wrong answer: mismatched output length"
            if first_failure_metadata is None:
                first_failure_metadata = wa_args
            continue

        line_mismatch = False
        for output_line_idx, (stripped_prediction_line, stripped_gt_out_line) in enumerate(
                zip(stripped_prediction_lines, stripped_gt_out_lines)
        ):
            wa_args["error_message"] = (
                f"Wrong answer at {output_line_idx=}: "
                f"{testing_util.truncatefn(stripped_prediction_line)} != {testing_util.truncatefn(stripped_gt_out_line)}"
            )

            if stripped_prediction_line == stripped_gt_out_line:
                continue

            success, decimal_prediction_line = testing_util.convert_line_to_decimals(stripped_prediction_line)
            if not success:
                line_mismatch = True
                break
            success, decimal_gtout_line = testing_util.convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                line_mismatch = True
                break
            if decimal_prediction_line == decimal_gtout_line:
                continue

            line_mismatch = True
            break

        if line_mismatch:
            all_results.append(-2)
            if first_failure_metadata is None:
                first_failure_metadata = wa_args
            continue

        all_results.append(True)

    if first_failure_metadata is not None:
        return all_results, first_failure_metadata
    return all_results, {"execution time": total_execution_time}


def run_test_independent(sample: dict, test: str, timeout: int = 6) -> tuple[list, dict]:
    """Like testing_util.run_test, but every test case in `sample` runs and is
    scored independently instead of stopping at the first failure."""
    signal.signal(signal.SIGALRM, testing_util.timeout_handler)
    testing_util.reliability_guard()

    try:
        in_outs = json.loads(sample["input_output"])
    except MemoryError:
        return [-6], {"error_code": -6, "error_message": "MemoryError"}
    if in_outs.get("fn_name") is None:
        which_type = testing_util.CODE_TYPE.standard_input
        method_name = None
    else:
        which_type = testing_util.CODE_TYPE.call_based
        method_name = in_outs["fn_name"]

    signal.alarm(timeout)
    try:
        if which_type == testing_util.CODE_TYPE.call_based:
            graded = grade_call_based_independent(
                code=test, all_inputs=in_outs["inputs"], all_outputs=in_outs["outputs"],
                fn_name=method_name, timeout=timeout,
            )
        else:
            graded = grade_stdio_independent(
                code=test, all_inputs=in_outs["inputs"], all_outputs=in_outs["outputs"], timeout=timeout,
            )
    except MemoryError:
        return [-6], {"error_code": -6, "error_message": "MemoryError"}
    except Exception as e:
        return [-4], {"error_code": -4, "error_message": f"Error during testing: {e}"}
    finally:
        signal.alarm(0)

    if graded is None:
        return [-4], {
            "error_code": -4,
            "error_message": "Error during testing: cannot unpack non-iterable NoneType object",
        }
    return graded


def _temp_run_independent(sample: dict, generation: str, result: list, metadata_list: list, timeout: int) -> None:
    # Redirect this process's stdout (fd 1) to /dev/null
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)

    try:
        res, metadata = run_test_independent(sample, test=generation, timeout=timeout)
    except MemoryError:
        res, metadata = [-6], {"error_code": -6, "error_message": "MemoryError"}
    except Exception as e:
        res, metadata = [-7], {"error_code": -7, "error_message": f"{type(e).__name__}: {e}"}
    result.append(res)
    metadata_list.append(metadata)


def _temp_run_fast(sample: dict, generation: str, result: list, metadata_list: list, timeout: int) -> None:
    # Same fd-redirect as _temp_run_independent, but grades with LCB's stock early-exit-on-first-failure run_test
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)

    try:
        res, metadata = testing_util.run_test(sample, test=generation, timeout=timeout)
    except MemoryError:
        res, metadata = [-6], {"error_code": -6, "error_message": "MemoryError"}
    except Exception as e:
        res, metadata = [-7], {"error_code": -7, "error_message": f"{type(e).__name__}: {e}"}
    result.append(res)
    metadata_list.append(metadata)


def check_correctness_independent(
        sample: dict, generation: str, timeout: int, fast: bool = False,
) -> tuple[list, dict]:
    """Same subprocess/timeout wrapping as LCB's compute_code_generation_metrics.check_correctness,
    just calling run_test_independent instead of run_test. fast=True stops at the first failing test case
    (LCB's default grading behavior) instead of scoring every test case.

    Explicitly forked (not the ambient default -- vLLM's VLLM_WORKER_MULTIPROC_METHOD=spawn env var
    flips the process-wide multiprocessing default to spawn): a spawned child re-imports the entire
    dependency stack from scratch, which under cluster load on an NFS-mounted venv can block for minutes,
    which blows the wall-clock join below -- forking from the (lightweight, no vLLM loaded) grader worker
    process is instant."""
    fork_ctx = multiprocessing.get_context("fork")
    manager = fork_ctx.Manager()
    try:
        result = manager.list()
        metadata_list = manager.list()
        target = _temp_run_fast if fast else _temp_run_independent
        p = fork_ctx.Process(
            target=target,
            args=(sample, generation, result, metadata_list, timeout),
        )
        p.start()
        p.join(
            timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5
        )
        if p.is_alive():
            p.kill()
        if not result or not metadata_list:
            in_outs = json.loads(sample["input_output"])
            result = [[-1 for _ in range(len(in_outs["inputs"]))]]
            metadata_list = [{"error_code": -1, "error_message": "global timeout"}]
        return result[0], metadata_list[0]
    finally:
        # manager.list() spawns its own subprocess, need to shut it down explicitly so
        # long-lived pool workers (e.g. AsyncGrader) don't leak one per candidate
        manager.shutdown()


def grade_one(sample: dict, code: str, total_tests: int, timeout: int, fast: bool = False) -> tuple[float, dict]:
    """(accuracy, metadata) of a single candidate, accuracy being the fraction of unit tests passed."""
    result, meta = check_correctness_independent(sample, code, timeout=timeout, fast=fast)
    acc = sum(1 for x in result if x is True) / total_tests
    return acc, meta
