"""Optional shape-tuned Triton FFN kernels for Transformer inference."""

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

    TRITON_FFN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on non-Triton systems.
    triton = None
    tl = None
    TRITON_FFN_AVAILABLE = False


if TRITON_FFN_AVAILABLE:

    @triton.jit
    def _grouped_accumulation_kernel(
        input_ptr,
        weight_kn_ptr,
        bias_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        GROUP_K: tl.constexpr,
        FUSE_GELU: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        """Tensor-core GEMM with short fp16 partials promoted to fp32."""
        program_id = tl.program_id(0)
        programs_n = tl.cdiv(N, BN)
        program_m = program_id // programs_n
        program_n = program_id % programs_n
        offsets_m = program_m * BM + tl.arange(0, BM)
        offsets_n = program_n * BN + tl.arange(0, BN)
        offsets_k = tl.arange(0, BK)

        # Reduced-precision accumulation is considerably faster on Ampere, but
        # one fp16 reduction across the complete K dimension is not accurate
        # enough for this benchmark. Accumulate short tensor-core fragments in
        # fp16 and promote each fragment before adding it to the fp32 result.
        accumulator = tl.zeros((BM, BN), dtype=tl.float32)
        for group_start in range(0, K, GROUP_K):
            partial = tl.zeros((BM, BN), dtype=tl.float16)
            for inner in range(0, GROUP_K, BK):
                current_k = group_start + inner + offsets_k
                inputs = tl.load(
                    input_ptr
                    + offsets_m[:, None] * K
                    + current_k[None, :],
                    mask=(offsets_m[:, None] < M)
                    & (current_k[None, :] < K),
                    other=0.0,
                )
                weights = tl.load(
                    weight_kn_ptr
                    + current_k[:, None] * N
                    + offsets_n[None, :],
                    mask=(current_k[:, None] < K)
                    & (offsets_n[None, :] < N),
                    other=0.0,
                )
                partial = tl.dot(
                    inputs,
                    weights,
                    acc=partial,
                    out_dtype=tl.float16,
                )
            accumulator += partial.to(tl.float32)

        bias = tl.load(
            bias_ptr + offsets_n,
            mask=offsets_n < N,
            other=0.0,
        )
        values = accumulator + bias[None, :]
        if FUSE_GELU:
            cube = values * values * values
            gelu_argument = 0.7978845608 * (values + 0.044715 * cube)
            tanh_value = 2.0 * tl.sigmoid(2.0 * gelu_argument) - 1.0
            values = 0.5 * values * (1.0 + tanh_value)

        output_offsets = (
            output_ptr + offsets_m[:, None] * N + offsets_n[None, :]
        )
        tl.store(
            output_offsets,
            values,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )

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
        gelu_argument = 0.7978845608 * (values + 0.044715 * cube)
        # tanh(x) == 2 * sigmoid(2x) - 1. Triton's sigmoid implementation is
        # measurably faster than the libdevice tanh call on Ampere, without
        # changing the tanh-GELU approximation used by the reference path.
        tanh_value = 2.0 * tl.sigmoid(2.0 * gelu_argument) - 1.0
        values = 0.5 * values * (1.0 + tanh_value)
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


def _grouped_accumulation_linear(
    inputs: torch.Tensor,
    weight_kn: torch.Tensor,
    bias: torch.Tensor,
    *,
    group_k: int,
    block_m: int,
    block_n: int,
    fuse_gelu: bool,
) -> torch.Tensor:
    """Run one of the shape-tuned accuracy-preserving Ampere kernels."""
    if not TRITON_FFN_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if inputs.device.type != "cuda":
        raise ValueError("the grouped FFN kernel requires CUDA tensors")
    if weight_kn.ndim != 2 or bias.ndim != 1:
        raise ValueError("weight must be 2-D and bias must be 1-D")

    leading_shape = inputs.shape[:-1]
    m = inputs.numel() // inputs.shape[-1]
    k = inputs.shape[-1]
    n = bias.numel()
    if weight_kn.shape != (k, n):
        raise ValueError("transposed FFN weight does not match the input")
    if group_k <= 0 or group_k % 32 != 0:
        raise ValueError("group_k must be a positive multiple of 32")

    inputs_2d = inputs.reshape(m, k).to(torch.float16)
    output = torch.empty((m, n), device=inputs.device, dtype=torch.float16)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _grouped_accumulation_kernel[grid](
        inputs_2d,
        weight_kn,
        bias,
        output,
        M=m,
        N=n,
        K=k,
        GROUP_K=group_k,
        FUSE_GELU=fuse_gelu,
        BM=block_m,
        BN=block_n,
        BK=32,
        num_warps=4,
        num_stages=3,
    )
    return output.view(*leading_shape, n)


def triton_grouped_ffn_gelu(
    inputs: torch.Tensor,
    weight_kn: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Shape-tuned 512-to-2048 FFN projection with fused tanh GELU."""
    if inputs.shape[-1] != 512 or weight_kn.shape != (512, 2048):
        raise ValueError("grouped FFN+GELU expects a 512-to-2048 projection")
    return _grouped_accumulation_linear(
        inputs,
        weight_kn,
        bias,
        group_k=32,
        block_m=64,
        block_n=128,
        fuse_gelu=True,
    )


def triton_grouped_ffn_out(
    inputs: torch.Tensor,
    weight_kn: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Shape-tuned 2048-to-512 FFN output projection."""
    if inputs.shape[-1] != 2048 or weight_kn.shape != (2048, 512):
        raise ValueError("grouped FFN output expects a 2048-to-512 projection")
    return _grouped_accumulation_linear(
        inputs,
        weight_kn,
        bias,
        group_k=96,
        block_m=128,
        block_n=64,
        fuse_gelu=False,
    )
