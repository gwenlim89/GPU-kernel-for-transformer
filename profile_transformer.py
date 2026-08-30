#!/usr/bin/env python3
"""Profile the optimized Transformer and print its hottest CUDA operations."""

from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from torch_transformer_benchmark import (
    TransformerConfig,
    UserOptimizedTransformer,
    generate_random_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rows", type=int, default=25)
    parser.add_argument("--trace", help="optional Chrome trace output path")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("profile_transformer.py requires a CUDA GPU")
    if args.iterations <= 0 or args.warmup < 0 or args.rows <= 0:
        raise ValueError("iterations/rows must be positive and warmup non-negative")

    device = torch.device("cuda")
    config = TransformerConfig(8, 128, 512, 8, 2048, 6, args.causal)
    model = UserOptimizedTransformer(config).to(device).eval()
    if torch.cuda.get_device_capability(device) >= (7, 0):
        model.enable_mixed_precision(torch.float16)
    x, mask = generate_random_case(
        config, device, torch.float32, 1234, args.padding_ratio, 1.0
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x, mask)
        torch.cuda.synchronize(device)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            for _ in range(args.iterations):
                model(x, mask)
        torch.cuda.synchronize(device)

    print(
        profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=args.rows
        )
    )
    if args.trace:
        profiler.export_chrome_trace(args.trace)
        print(f"Chrome trace written to {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
