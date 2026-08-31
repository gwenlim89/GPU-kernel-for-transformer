#!/usr/bin/env python3
"""Run isolated correctness, fallback, and latency regression scenarios."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Scenario:
    name: str
    arguments: tuple[str, ...]
    measures_target: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use fewer trials and timing samples for a fast smoke test",
    )
    parser.add_argument(
        "--target-speedup",
        type=float,
        default=4.0,
        help="speedup goal reported for the default CUDA scenario",
    )
    parser.add_argument(
        "--require-target",
        action="store_true",
        help="return a failure status when the speedup target is not reached",
    )
    return parser.parse_args()


def scenarios(quick: bool) -> Sequence[Scenario]:
    trials = "3" if quick else "20"
    repeats = "20" if quick else "200"
    rounds = "1" if quick else "5"
    common = (
        "--accuracy-trials",
        trials,
        "--warmup",
        "10" if quick else "50",
        "--repeats",
        repeats,
        "--benchmark-rounds",
        rounds,
    )
    return (
        Scenario("default CUDA path", common, measures_target=True),
        Scenario("causal attention", (*common, "--causal")),
        Scenario(
            "fixed padding-mask CUDA Graph",
            (*common, "--padding-ratio", "0.25"),
        ),
        Scenario(
            "native CUDA fallbacks",
            (
                *common,
                "--no-triton-fused-norm",
                "--no-triton-fused-ffn",
                "--no-attention-fp16-accumulation",
                "--no-cuda-graphs",
            ),
        ),
        Scenario(
            "portable CPU fallback",
            (
                "--device",
                "cpu",
                "--batch-size",
                "2",
                "--seq-len",
                "16",
                "--d-model",
                "128",
                "--heads",
                "4",
                "--ffn-dim",
                "512",
                "--layers",
                "2",
                "--mixed-precision",
                "none",
                "--accuracy-trials",
                "2" if quick else "5",
                "--warmup",
                "2",
                "--repeats",
                "3" if quick else "10",
                "--benchmark-rounds",
                "1" if quick else "3",
                "--no-cuda-graphs",
            ),
        ),
    )


def main() -> int:
    args = parse_args()
    if args.target_speedup <= 0:
        raise ValueError("target-speedup must be positive")

    benchmark = Path(__file__).with_name("torch_transformer_benchmark.py")
    failures: list[str] = []
    measured_speedup: float | None = None

    for index, scenario in enumerate(scenarios(args.quick), start=1):
        print(f"\n=== [{index}/5] {scenario.name} ===", flush=True)
        result = subprocess.run(
            [sys.executable, "-u", str(benchmark), *scenario.arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        if result.returncode != 0:
            failures.append(
                f"{scenario.name} exited with status {result.returncode}"
            )
        if scenario.measures_target:
            match = re.search(r"speedup\s+: ([0-9.]+)x", result.stdout)
            if match:
                measured_speedup = float(match.group(1))
            else:
                failures.append("default CUDA speedup was not reported")

    print("\n=== Regression summary ===")
    if measured_speedup is not None:
        reached = measured_speedup >= args.target_speedup
        print(
            f"speedup target: {args.target_speedup:.3f}x | "
            f"measured: {measured_speedup:.3f}x | "
            f"{'REACHED' if reached else 'NOT REACHED'}"
        )
        if args.require_target and not reached:
            failures.append(
                f"speedup {measured_speedup:.3f}x is below "
                f"{args.target_speedup:.3f}x"
            )

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: all correctness and fallback scenarios completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
