"""Optional Triton kernels for inference-only Transformer normalization.

The fused operation computes ``residual + update`` and the following
LayerNorm in one pass over GPU memory.  Importing this module is safe when
Triton is not installed; callers can check ``TRITON_AVAILABLE`` and retain a
native PyTorch fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch

# Keep JIT artifacts inside the project by default. This also avoids permission
# failures when an IDE or sandbox cannot write to the user's home directory.
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    str(Path(__file__).resolve().parent / ".kernel_cache" / "triton"),
)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on non-Triton systems.
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _residual_layer_norm_kernel(
        residual_ptr,
        update_ptr,
        weight_ptr,
        bias_ptr,
        residual_out_ptr,
        norm_out_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        STORE_RESIDUAL: tl.constexpr,
        APPLY_AFFINE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row_offsets = row * n_cols + offsets

        residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        update = tl.load(update_ptr + row_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        summed = residual + update
        if STORE_RESIDUAL:
            tl.store(residual_out_ptr + row_offsets, summed, mask=mask)

        mean = tl.sum(summed, axis=0) / n_cols
        centered = tl.where(mask, summed - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / n_cols
        normalized = centered * tl.rsqrt(variance + eps)

        if APPLY_AFFINE:
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            output = normalized * weight + bias
        else:
            output = normalized
        tl.store(norm_out_ptr + row_offsets, output, mask=mask)


    @triton.jit
    def _layer_norm_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        APPLY_AFFINE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row_offsets = row * n_cols + offsets

        values = tl.load(input_ptr + row_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        mean = tl.sum(values, axis=0) / n_cols
        centered = tl.where(mask, values - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / n_cols
        normalized = centered * tl.rsqrt(variance + eps)
        if APPLY_AFFINE:
            weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            output = normalized * weight + bias
        else:
            output = normalized
        tl.store(
            output_ptr + row_offsets,
            output,
            mask=mask,
        )


def _launch_parameters(n_cols: int) -> Tuple[int, int]:
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError("feature dimension is too large for this Triton kernel")
    # Narrow Transformer rows expose enough independent programs that one warp
    # per row gives this Ampere workload better occupancy than the tutorial's
    # more conservative four-warp default.
    if block_size <= 512:
        num_warps = 1
    elif block_size < 2048:
        num_warps = 4
    else:
        num_warps = 8
    return block_size, num_warps


def layer_norm(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    output_dtype: torch.dtype,
    apply_affine: bool = True,
) -> torch.Tensor:
    """LayerNorm whose output is stored directly in the following GEMM dtype."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if inputs.device.type != "cuda":
        raise ValueError("the fused LayerNorm kernel requires CUDA tensors")
    n_cols = inputs.shape[-1]
    if n_cols != weight.numel() or weight.shape != bias.shape:
        raise ValueError("LayerNorm parameters do not match the input width")
    block_size, num_warps = _launch_parameters(n_cols)
    output = torch.empty_like(inputs, dtype=output_dtype)
    rows = inputs.numel() // n_cols
    _layer_norm_kernel[(rows,)](
        inputs,
        weight,
        bias,
        output,
        n_cols=n_cols,
        eps=eps,
        APPLY_AFFINE=apply_affine,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output


def fused_residual_layer_norm(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    norm_dtype: torch.dtype,
    store_residual: bool = True,
    apply_affine: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the fp32 residual sum and normalized output in ``norm_dtype``."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if residual.device.type != "cuda":
        raise ValueError("the fused LayerNorm kernel requires CUDA tensors")
    if residual.shape != update.shape:
        raise ValueError("residual and update must have identical shapes")
    if residual.shape[-1] != weight.numel() or weight.shape != bias.shape:
        raise ValueError("LayerNorm parameters do not match the input width")

    n_cols = residual.shape[-1]
    block_size, num_warps = _launch_parameters(n_cols)

    # When the final residual is immediately normalized and discarded, pass an
    # existing pointer and compile out the otherwise unnecessary global write.
    residual_out = torch.empty_like(residual) if store_residual else residual
    norm_out = torch.empty_like(residual, dtype=norm_dtype)
    rows = residual.numel() // n_cols
    _residual_layer_norm_kernel[(rows,)](
        residual,
        update,
        weight,
        bias,
        residual_out,
        norm_out,
        n_cols=n_cols,
        eps=eps,
        STORE_RESIDUAL=store_residual,
        APPLY_AFFINE=apply_affine,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return residual_out, norm_out
