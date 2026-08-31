#!/usr/bin/env python3
"""Run the supplied 14-shape hackathon benchmark in isolated processes."""

from __future__ import annotations

import argparse
import math
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import torch

from torch_transformer_benchmark import (
    OFFICIAL_BENCHMARK_SHAPES,
    TransformerConfig,
    print_official_benchmark_shapes,
)


GIB = 1024**3


@dataclass(frozen=True)
class CaseResult:
    shape_id: int
    status: str
    speedup: float | None = None
    reason: str = ""


def parse_case_ids(specification: str) -> list[int]:
    """Parse values such as ``all``, ``1,3,7`` or ``1-5,12``."""
    if specification.strip().lower() == "all":
        return list(OFFICIAL_BENCHMARK_SHAPES)

    selected: set[int] = set()
    for part in specification.split(","):
        token = part.strip()
        if not token:
            raise ValueError("empty value in --cases")
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"invalid descending case range: {token}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))

    unknown = sorted(selected.difference(OFFICIAL_BENCHMARK_SHAPES))
    if unknown:
        raise ValueError(f"unknown benchmark shape IDs: {unknown}")
    return sorted(selected)


def estimated_reference_peak_bytes(
    config: TransformerConfig,
    batch_chunk_size: int | None = None,
) -> int:
    """Conservative peak estimate for this script's explicit FP32 baseline."""
    batch = config.batch_size
    sequence = config.seq_len
    width = config.d_model
    heads = config.num_heads

    activation = batch * sequence * width * 4
    working_batch = min(batch, batch_chunk_size or batch)
    chunk_activation = working_batch * sequence * width * 4
    # scores and fp32 softmax probabilities coexist in the baseline path.
    attention = 2 * working_batch * heads * sequence * sequence * 4
    parameters_per_layer = (
        4 * width * width
        + 2 * width * config.ffn_dim
        + 4 * width
        + config.ffn_dim
    )
    # Both the reference and optimized model are resident simultaneously.
    model_parameters = 2 * config.num_layers * parameters_per_layer * 4
    if working_batch < batch:
        # Full input/reference/candidate outputs remain resident, while all
        # intermediate tensors are bounded by the chunk size.
        resident_full_tensors = 3 * activation
        other_activations = 8 * chunk_activation
        return (
            attention
            + model_parameters
            + resident_full_tensors
            + other_activations
        )
    # Input, QKV, context, residuals, reference output, and candidate output.
    return attention + model_parameters + 8 * activation


def available_device_bytes(device_name: str) -> int | None:
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(device).total_memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="all",
        help="shape IDs to run, for example all, 1,3,7, or 1-5,12",
    )
    parser.add_argument("--list", action="store_true", help="list shapes and exit")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use one accuracy trial and fewer timing samples",
    )
    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.8,
        help="maximum fraction of device memory allowed by the preflight estimate",
    )
    parser.add_argument(
        "--force-oversized",
        action="store_true",
        help="run cases even when the explicit reference is estimated not to fit",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return a failure status if any case is skipped, proxied, or fails",
    )
    parser.add_argument(
        "--run-full-shape14",
        action="store_true",
        help=(
            "opt in to the real 100000-token streamed shape-14 run; without "
            "this flag the suite runs a bounded correctness/performance proxy"
        ),
    )
    parser.add_argument(
        "--shape14-samples",
        type=int,
        default=32,
        help="streamed samples for the full shape-14 run (official batch is 32)",
    )
    parser.add_argument(
        "--shape14-proxy-seq-len",
        type=int,
        default=2048,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("--memory-fraction must be in (0, 1]")
    if args.accuracy_trials <= 0:
        raise ValueError("--accuracy-trials must be positive")
    if args.warmup < 0 or args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("warmup must be non-negative; repeats/rounds must be positive")
    if not 1 <= args.shape14_samples <= 32:
        raise ValueError("--shape14-samples must be between 1 and 32")
    if not 1 <= args.shape14_proxy_seq_len < 100000:
        raise ValueError("--shape14-proxy-seq-len must be in [1, 100000)")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.list:
        print_official_benchmark_shapes()
        return 0

    shape_ids = parse_case_ids(args.cases)
    device_bytes = available_device_bytes(args.device)
    benchmark = Path(__file__).with_name("torch_transformer_benchmark.py")
    trials = 1 if args.quick else args.accuracy_trials
    warmup = 3 if args.quick else args.warmup
    repeats = 10 if args.quick else args.repeats
    rounds = 1 if args.quick else args.benchmark_rounds
    results: list[CaseResult] = []

    for position, shape_id in enumerate(shape_ids, start=1):
        config = OFFICIAL_BENCHMARK_SHAPES[shape_id]
        if shape_id == 14:
            print(
                f"\n=== [{position}/{len(shape_ids)}] official shape 14 "
                "(streamed) ===",
                flush=True,
            )
            streamed_runner = Path(__file__).with_name("streamed_shape14.py")
            command = [
                sys.executable,
                "-u",
                str(streamed_runner),
                "--proxy-seq-len",
                str(args.shape14_proxy_seq_len),
                "--memory-fraction",
                str(args.memory_fraction),
                "--warmup",
                "0" if args.quick else "1",
                "--repeats",
                "1" if args.quick or args.run_full_shape14 else "3",
            ]
            if args.run_full_shape14:
                command.extend(
                    [
                        "--full",
                        "--samples",
                        str(args.shape14_samples),
                    ]
                )
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, file=sys.stderr, end="")
            status_match = re.search(
                r"shape14_streamed_status=(FULL|PROXY)", completed.stdout
            )
            if completed.returncode == 0 and status_match:
                status = "PASS" if status_match.group(1) == "FULL" else "PROXY"
                reason = (
                    "full streamed batch completed"
                    if status == "PASS"
                    else (
                        f"real sequence completed for {args.shape14_samples}/32 samples"
                        if args.run_full_shape14
                        else "bounded proxy; use --run-full-shape14 for 100000 tokens"
                    )
                )
                results.append(CaseResult(shape_id, status, reason=reason))
            else:
                results.append(
                    CaseResult(
                        shape_id,
                        "FAIL",
                        reason=f"streamed runner exited with status {completed.returncode}",
                    )
                )
            continue

        batch_chunk_size = 256 if shape_id == 6 else None
        estimate = estimated_reference_peak_bytes(config, batch_chunk_size)
        print(
            f"\n=== [{position}/{len(shape_ids)}] official shape {shape_id} ===",
            flush=True,
        )
        print(
            f"config={config} | estimated_reference_peak={estimate / GIB:.2f} GiB",
            flush=True,
        )
        if (
            device_bytes is not None
            and estimate > device_bytes * args.memory_fraction
            and not args.force_oversized
        ):
            reason = (
                f"estimated peak exceeds {args.memory_fraction:.0%} of "
                f"{device_bytes / GIB:.2f} GiB device memory"
            )
            print(f"SKIP: {reason}", flush=True)
            results.append(CaseResult(shape_id, "SKIP", reason=reason))
            continue

        command = [
            sys.executable,
            "-u",
            str(benchmark),
            "--shape-id",
            str(shape_id),
            "--device",
            args.device,
            "--accuracy-trials",
            str(trials),
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
            "--benchmark-rounds",
            str(rounds),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, end="")
        match = re.search(r"speedup\s+: ([0-9.]+)x", completed.stdout)
        if completed.returncode == 0 and match:
            results.append(CaseResult(shape_id, "PASS", float(match.group(1))))
        else:
            reason = f"benchmark exited with status {completed.returncode}"
            results.append(CaseResult(shape_id, "FAIL", reason=reason))

    print("\n=== Official shape-suite summary ===")
    for result in results:
        detail = (
            f"{result.speedup:.3f}x" if result.speedup is not None else result.reason
        )
        print(f"shape {result.shape_id:02d}: {result.status:4s} | {detail}")

    speedups = [result.speedup for result in results if result.speedup is not None]
    if speedups:
        geometric_mean = math.exp(statistics.fmean(math.log(x) for x in speedups))
        print(f"geometric-mean speedup across completed cases: {geometric_mean:.3f}x")
    failures = [result for result in results if result.status == "FAIL"]
    incomplete = [
        result for result in results if result.status in ("SKIP", "PROXY")
    ]
    if failures or (args.strict and incomplete):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
