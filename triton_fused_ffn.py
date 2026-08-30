"""Optional shape-tuned Triton GEMM + GELU for Transformer inference."""

from __future__ import annotations

import os
from pathlib import Path

import torch

os.environ.setdefault(
    "TRITON_CACHE_DIR",
    str(Path(__file__).resolve().parent / ".kernel_cache" / "triton"),
)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    TRITON_FFN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on non-Triton systems.
    triton = None
    tl = None
    libdevice = None
    TRITON_FFN_AVAILABLE = False


if TRITON_FFN_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config(
                {"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BM": 64, "BN": 256, "BK": 32, "GROUP_M": 8},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BM": 128, "BN": 64, "BK": 32, "GROUP_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BM": 128, "BN": 256, "BK": 32, "GROUP_M": 8},
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 8},
                num_stages=4,
                num_warps=4,
            ),
        ],
        key=["M", "N", "K"],
        cache_results=True,
    )
    @triton.jit
    def _ffn_gelu_kernel(
        input_ptr,
        weight_kn_ptr,
        bias_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        programs_m = tl.cdiv(M, BM)
        programs_n = tl.cdiv(N, BN)

        # Group neighboring M tiles so they reuse the same weight tiles while
        # those tiles are still resident in L2 cache.
        programs_per_group = GROUP_M * programs_n
        group_id = program_id // programs_per_group
        first_program_m = group_id * GROUP_M
        group_size_m = min(programs_m - first_program_m, GROUP_M)
        program_m = first_program_m + (
            (program_id % programs_per_group) % group_size_m
        )
        program_n = (program_id % programs_per_group) // group_size_m

        offsets_m = program_m * BM + tl.arange(0, BM)
        offsets_n = program_n * BN + tl.arange(0, BN)
        offsets_k = tl.arange(0, BK)
        input_ptrs = (
            input_ptr + offsets_m[:, None] * K + offsets_k[None, :]
        )
        weight_ptrs = (
            weight_kn_ptr + offsets_k[:, None] * N + offsets_n[None, :]
        )

        accumulator = tl.zeros((BM, BN), dtype=tl.float32)
        for k_block in range(0, tl.cdiv(K, BK)):
            k_mask = k_block * BK + offsets_k < K
            inputs = tl.load(
                input_ptrs,
                mask=(offsets_m[:, None] < M) & k_mask[None, :],
                other=0.0,
            )
            weights = tl.load(
                weight_ptrs,
                mask=k_mask[:, None] & (offsets_n[None, :] < N),
                other=0.0,
            )
            accumulator += tl.dot(inputs, weights)
            input_ptrs += BK
            weight_ptrs += BK * N

        bias = tl.load(
            bias_ptr + offsets_n,
            mask=offsets_n < N,
            other=0.0,
        )
        values = accumulator + bias[None, :]

        # Fuse the tanh GELU epilogue so the [M, N] activation is written only
        # once. Computation remains fp32 until the final fp16 store.
        cube = values * values * values
        values = 0.5 * values * (
            1.0
            + libdevice.tanh(
                0.7978845608 * (values + 0.044715 * cube)
            )
        )
        output_ptrs = (
            output_ptr + offsets_m[:, None] * N + offsets_n[None, :]
        )
        tl.store(
            output_ptrs,
            values,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )


def triton_ffn_gelu(
    inputs: torch.Tensor,
    weight_kn: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Compute GELU(inputs @ weight_kn + bias) with fp32 accumulation."""
    if not TRITON_FFN_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if inputs.device.type != "cuda":
        raise ValueError("the Triton FFN kernel requires CUDA tensors")
    if weight_kn.ndim != 2 or bias.ndim != 1:
        raise ValueError("weight must be 2-D and bias must be 1-D")

    leading_shape = inputs.shape[:-1]
    m = inputs.numel() // inputs.shape[-1]
    k = inputs.shape[-1]
    n = bias.numel()
    if weight_kn.shape != (k, n):
        raise ValueError("transposed FFN weight does not match the input")

    inputs_2d = inputs.reshape(m, k).to(torch.float16)
    output = torch.empty((m, n), device=inputs.device, dtype=torch.float16)

    def grid(meta):
        return (
            triton.cdiv(m, meta["BM"]) * triton.cdiv(n, meta["BN"]),
        )

    _ffn_gelu_kernel[grid](
        inputs_2d,
        weight_kn,
        bias,
        output,
        M=m,
        N=n,
        K=k,
    )
    return output.view(*leading_shape, n)
