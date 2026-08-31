#!/usr/bin/env python3
"""Safely validate and stream benchmark shape 14 on a 6 GiB GPU."""

from __future__ import annotations

import argparse
import statistics
import sys

import torch

from torch_transformer_benchmark import (
    OFFICIAL_BENCHMARK_SHAPES,
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
)


GIB = 1024**3
SHAPE_ID = 14
DEFAULT_PROXY_SEQUENCE = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="use the real sequence length of 100000 instead of a safe proxy",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="streamed batch samples to execute (full official batch is 32)",
    )
    parser.add_argument(
        "--proxy-seq-len",
        type=int,
        default=DEFAULT_PROXY_SEQUENCE,
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.75,
        help="maximum GPU-memory fraction allowed by the streaming estimate",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.samples is not None and not 1 <= args.samples <= 32:
        raise ValueError("--samples must be between 1 and 32")
    if not 1 <= args.proxy_seq_len < 100000:
        raise ValueError("--proxy-seq-len must be in [1, 100000)")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("--memory-fraction must be in (0, 1]")


def streaming_peak_estimate(sequence: int, width: int, layers: int) -> int:
    """Conservative O(SD), not O(S^2), mixed-precision working estimate."""
    fp32_sample = sequence * width * 4
    parameter_elements_per_layer = 6 * width * width + 6 * width
    parameters = layers * parameter_elements_per_layer * 4
    # Input/output/residual plus fp16 QKV, context, norms, and FFN workspace.
    return 7 * fp32_sample + parameters


def make_models(config: TransformerConfig, seed: int):
    torch.manual_seed(seed)
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    copy_model_weights(baseline, optimized)
    return baseline, optimized


def proxy_accuracy(args: argparse.Namespace, device: torch.device) -> None:
    config = TransformerConfig(
        batch_size=1,
        seq_len=args.proxy_seq_len,
        d_model=1024,
        num_heads=16,
        ffn_dim=1024,
        num_layers=2,
        causal=True,
    )
    baseline, optimized = make_models(config, args.seed)
    baseline = baseline.to(device).eval()
    optimized = optimized.to(device).eval()
    optimized.enable_mixed_precision(torch.float16)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(
        1,
        config.seq_len,
        config.d_model,
        device=device,
        generator=generator,
    )
    mask = torch.ones(1, config.seq_len, device=device, dtype=torch.bool)
    with torch.inference_mode():
        reference = baseline(x, mask)
        candidate = optimized(x, mask)
    result = compare_outputs(reference, candidate, args.rtol, args.atol)
    print(
        "proxy_accuracy="
        f"{'PASS' if result.passed else 'FAIL'} "
        f"sequence={config.seq_len} failed={result.failed_elements}/"
        f"{result.total_elements} max_abs={result.max_abs_error:.6g}"
    )
    if not result.passed:
        raise RuntimeError("shape-14 proxy accuracy failed")
    del baseline, optimized, x, mask, reference, candidate
    torch.cuda.empty_cache()


def streamed_performance(
    args: argparse.Namespace,
    device: torch.device,
    sequence: int,
    samples: int,
) -> None:
    torch.cuda.reset_peak_memory_stats(device)
    official = OFFICIAL_BENCHMARK_SHAPES[SHAPE_ID]
    config = TransformerConfig(
        batch_size=official.batch_size,
        seq_len=sequence,
        d_model=official.d_model,
        num_heads=official.num_heads,
        ffn_dim=official.ffn_dim,
        num_layers=official.num_layers,
        causal=True,
    )
    _, optimized = make_models(config, args.seed)
    del _
    optimized = optimized.to(device).eval()
    optimized.enable_mixed_precision(torch.float16)

    generator = torch.Generator(device=device).manual_seed(args.seed + 100000)
    static_sample = torch.randn(
        1,
        sequence,
        config.d_model,
        device=device,
        generator=generator,
    )
    static_mask = torch.ones(1, sequence, device=device, dtype=torch.bool)

    with torch.inference_mode():
        for _ in range(args.warmup):
            optimized(static_sample, static_mask)
        torch.cuda.synchronize(device)

        timings: list[float] = []
        checksum = 0.0
        for _ in range(args.repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _sample_index in range(samples):
                output = optimized(static_sample, static_mask)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
            checksum += float(output[0, -1, 0].item())

        output_is_finite = bool(torch.isfinite(output).all().item())
        if not output_is_finite:
            raise RuntimeError("shape-14 streamed output contains NaN or Inf")

    median_ms = statistics.median(timings)
    per_sample_ms = median_ms / samples
    extrapolated_ms = per_sample_ms * official.batch_size
    peak_allocated = torch.cuda.max_memory_allocated(device) / GIB
    peak_reserved = torch.cuda.max_memory_reserved(device) / GIB
    print(
        f"streamed_sequence={sequence} samples={samples} "
        f"median={median_ms:.3f} ms per_sample={per_sample_ms:.3f} ms "
        f"official_batch32_estimate={extrapolated_ms:.3f} ms "
        f"peak_allocated={peak_allocated:.2f} GiB "
        f"peak_reserved={peak_reserved:.2f} GiB finite={output_is_finite} "
        f"checksum={checksum:.6g}"
    )
    mode = "FULL" if sequence == official.seq_len and samples == 32 else "PROXY"
    print(f"shape14_streamed_status={mode}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("shape-14 streaming requires a CUDA GPU")
    device = torch.device("cuda")
    official = OFFICIAL_BENCHMARK_SHAPES[SHAPE_ID]
    sequence = official.seq_len if args.full else args.proxy_seq_len
    samples = args.samples if args.samples is not None else (32 if args.full else 1)
    estimate = streaming_peak_estimate(
        sequence, official.d_model, official.num_layers
    )
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    print(
        f"shape14_mode={'full' if args.full else 'proxy'} "
        f"sequence={sequence} streamed_samples={samples} "
        f"estimated_peak={estimate / GIB:.2f} GiB "
        f"gpu_free={free_memory / GIB:.2f} GiB "
        f"gpu_total={total_memory / GIB:.2f} GiB"
    )
    memory_budget = min(free_memory, total_memory * args.memory_fraction)
    if estimate > memory_budget:
        print(
            "SAFE_SKIP: streaming estimate exceeds configured GPU-memory "
            "budget; no model or input was allocated"
        )
        return 3

    try:
        proxy_accuracy(args, device)
        streamed_performance(args, device, sequence, samples)
    except torch.cuda.OutOfMemoryError as error:
        torch.cuda.empty_cache()
        print(f"SAFE_ABORT: CUDA out of memory: {error}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
