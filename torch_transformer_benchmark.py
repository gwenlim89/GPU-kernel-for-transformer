#!/usr/bin/env python3
"""
Compare numerical accuracy and inference latency between a baseline Transformer
and a user-optimized implementation.

Correctness rule for every output element:
    abs(user - ref) <= atol
    OR
    abs(user - ref) <= rtol * abs(ref)

The default thresholds are atol=0.002 and rtol=0.02 (2%).
"""

from __future__ import annotations

import argparse
import copy
import math
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from batch_chunk_runner import BatchChunkedTransformer
from cuda_graph_runner import maybe_cuda_graph
from triton_fused_norm import (
    TRITON_AVAILABLE,
    fused_residual_layer_norm,
    layer_norm as triton_layer_norm,
)
from triton_fused_ffn import (
    TRITON_FFN_AVAILABLE,
    triton_ffn_gelu,
    triton_grouped_ffn_gelu,
    triton_grouped_ffn_out,
    triton_grouped_qkv_1024,
    triton_grouped_square_1024_gelu,
    triton_grouped_square_1024_out,
)


@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


# Official benchmark combinations supplied for the hackathon. ``d_model`` is
# the table's QKV dimension. Keep this mapping in one place so the single-case
# benchmark and the multi-shape suite cannot silently drift apart.
OFFICIAL_BENCHMARK_SHAPES: Dict[int, TransformerConfig] = {
    1: TransformerConfig(64, 128, 128, 4, 128, 4, True),
    2: TransformerConfig(1, 128, 128, 4, 128, 4, True),
    3: TransformerConfig(4, 128, 128, 4, 128, 4, True),
    4: TransformerConfig(16, 128, 128, 4, 128, 4, True),
    5: TransformerConfig(128, 128, 128, 4, 128, 4, True),
    6: TransformerConfig(10000, 128, 128, 4, 128, 4, True),
    7: TransformerConfig(64, 128, 32, 4, 32, 4, True),
    8: TransformerConfig(64, 128, 1024, 4, 1024, 4, True),
    9: TransformerConfig(64, 128, 128, 1, 128, 4, True),
    10: TransformerConfig(64, 128, 128, 2, 128, 4, True),
    11: TransformerConfig(64, 128, 128, 16, 128, 4, True),
    12: TransformerConfig(64, 32, 128, 4, 128, 4, True),
    13: TransformerConfig(64, 1024, 128, 4, 128, 4, True),
    14: TransformerConfig(32, 100000, 1024, 16, 1024, 2, True),
}


@dataclass(frozen=True)
class ShapeExecutionPlan:
    """Measured kernel choices for one exact official benchmark shape."""

    official_shape_id: Optional[int]
    ffn_strategy: str = "generic_triton"
    attention_strategy: str = "automatic"
    projection_strategy: str = "native"
    batch_chunk_size: int = 0

    def describe(self) -> str:
        shape = (
            f"official-{self.official_shape_id}"
            if self.official_shape_id is not None
            else "general"
        )
        chunk = (
            f", batch_chunk={self.batch_chunk_size}"
            if self.batch_chunk_size
            else ""
        )
        return (
            f"{shape}: ffn={self.ffn_strategy}, "
            f"attention={self.attention_strategy}, "
            f"projections={self.projection_strategy}{chunk}"
        )


_OFFICIAL_SHAPE_IDS_BY_CONFIG = {
    config: shape_id for shape_id, config in OFFICIAL_BENCHMARK_SHAPES.items()
}


def resolve_shape_execution_plan(config: TransformerConfig) -> ShapeExecutionPlan:
    """Select only hardware-validated specializations; retain safe fallbacks."""
    shape_id = _OFFICIAL_SHAPE_IDS_BY_CONFIG.get(config)
    return ShapeExecutionPlan(
        official_shape_id=shape_id,
        # The square-1024 grouped kernel wins for both the 8,192-row shape 8
        # and the streamed 100,000-row shape 14. Other widths retain the
        # autotuned fp32-accumulation Triton implementation.
        ffn_strategy=(
            "grouped_square_1024" if shape_id in (8, 14) else "generic_triton"
        ),
        # The Windows build has no Flash Attention. Efficient attention is a
        # measured win for shape 11's unusual 16 heads x head_dim 8 geometry;
        # automatic dispatch is already optimal for the other ordinary cases.
        attention_strategy=(
            "efficient" if shape_id == 11 else "automatic"
        ),
        # Shape 8's packed QKV and attention output projections use the same
        # conservative short-accumulation approach as its square FFN. Shape 14
        # remains native because attention projection tuning moved its full
        # 100,000-token latency by less than one percent.
        projection_strategy=(
            "grouped_1024" if shape_id == 8 else "native"
        ),
        # Shape 6 is exact batch-independent computation in bounded slices.
        batch_chunk_size=1024 if shape_id == 6 else 0,
    )


def print_official_benchmark_shapes() -> None:
    """Print the canonical benchmark table in command-line friendly form."""
    headings = (
        "ID",
        "Batch",
        "QKV Dim",
        "Heads",
        "Seq Len",
        "Layers",
        "Causal",
        "FFN Dim",
    )
    rows = [
        (
            shape_id,
            config.batch_size,
            config.d_model,
            config.num_heads,
            config.seq_len,
            config.num_layers,
            str(config.causal),
            config.ffn_dim,
        )
        for shape_id, config in OFFICIAL_BENCHMARK_SHAPES.items()
    ]
    widths = [
        max(len(str(heading)), *(len(str(row[index])) for row in rows))
        for index, heading in enumerate(headings)
    ]
    print(
        "  ".join(
            str(heading).rjust(widths[index])
            for index, heading in enumerate(headings)
        )
    )
    for row in rows:
        print(
            "  ".join(
                str(value).rjust(widths[index])
                for index, value in enumerate(row)
            )
        )


class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            # Mask invalid key positions. Shape: [B, 1, 1, S].
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        # Computing softmax in fp32 provides a stable reference for fp16/bf16 tests.
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )
        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)
        x = self.final_norm(x)
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class UserOptimizedTransformer(BaselineTransformer):
    """CUDA-optimized Transformer with portable eager-mode fallbacks."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self._shape_execution_plan = resolve_shape_execution_plan(config)
        # Caching is opt-in: ordinary forward calls retain full-sequence
        # semantics, while cached calls return outputs for only the new tokens.
        self._kv_cache_enabled = False
        self._kv_cache: dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._kv_cache_length = 0
        self._mixed_precision_dtype: Optional[torch.dtype] = None
        self._triton_fused_norm_enabled = TRITON_AVAILABLE
        self._triton_fused_ffn_enabled = TRITON_FFN_AVAILABLE
        self._attention_fp16_accumulation_enabled = (
            not config.causal
            and hasattr(
                torch.backends.cuda.matmul, "allow_fp16_accumulation"
            )
        )
        # Shape 6 cannot safely materialize full-batch attention on a 6 GiB
        # GPU. Batch elements are independent, so this is an exact execution
        # transformation rather than an approximation.
        self._large_batch_chunk_size = (
            self._shape_execution_plan.batch_chunk_size
        )

        # Padding plus causal attention needs their intersection. Build the
        # causal part once and let Module.to() move it with the model.
        # A dense causal mask is useful only when it must be intersected with
        # padding. Never allocate it for extreme sequences: shape 14 would
        # otherwise create a 100000^2 boolean tensor (~9.3 GiB) during model
        # construction. The all-valid path uses SDPA's implicit causal mask.
        causal_mask = (
            torch.ones(
                config.seq_len, config.seq_len, dtype=torch.bool
            ).tril()
            if config.causal and config.seq_len <= 4096
            else None
        )
        self.register_buffer(
            "_optimized_causal_mask", causal_mask, persistent=False
        )

    @property
    def shape_execution_plan(self) -> ShapeExecutionPlan:
        """Expose the immutable dispatch decision for diagnostics and tests."""
        return self._shape_execution_plan

    def enable_mixed_precision(
        self, dtype: Optional[torch.dtype] = torch.float16
    ) -> None:
        """Use CUDA autocast for GEMMs and attention; pass None to disable."""
        if dtype not in (None, torch.float16, torch.bfloat16):
            raise ValueError("mixed precision dtype must be float16 or bfloat16")
        if self._kv_cache_length:
            raise RuntimeError("reset the KV cache before changing precision")
        self._mixed_precision_dtype = dtype

    def enable_triton_fused_norm(self, enabled: bool = True) -> None:
        """Fuse residual additions and LayerNorm when Triton is available."""
        if enabled and not TRITON_AVAILABLE:
            raise RuntimeError(
                "Triton is not installed; disable this option or install Triton"
            )
        self._triton_fused_norm_enabled = enabled

    def enable_triton_fused_ffn(self, enabled: bool = True) -> None:
        """Use a shape-tuned Triton GEMM with a fused GELU epilogue."""
        if enabled and not TRITON_FFN_AVAILABLE:
            raise RuntimeError(
                "Triton is not installed; disable this option or install Triton"
            )
        self._triton_fused_ffn_enabled = enabled

    def enable_attention_fp16_accumulation(self, enabled: bool = True) -> None:
        """Use faster reduced-precision accumulation for attention GEMMs."""
        if enabled and self.config.causal:
            raise ValueError(
                "FP16 attention accumulation is disabled for causal models "
                "because it exceeds the accuracy tolerance"
            )
        if enabled and not hasattr(
            torch.backends.cuda.matmul, "allow_fp16_accumulation"
        ):
            raise RuntimeError("this PyTorch build lacks FP16 accumulation control")
        self._attention_fp16_accumulation_enabled = enabled

    def enable_kv_cache(self, enabled: bool = True) -> None:
        """Enable incremental decoding, clearing any previous sequence."""
        if enabled and not self.config.causal:
            raise ValueError("KV caching requires a causal Transformer")
        if enabled and self.training:
            raise RuntimeError("call eval() before enabling the KV cache")
        self._kv_cache_enabled = enabled
        self.reset_kv_cache()

    def reset_kv_cache(self) -> None:
        """Clear cached keys and values before starting a new sequence."""
        self._kv_cache.clear()
        self._kv_cache_length = 0

    @property
    def kv_cache_length(self) -> int:
        return self._kv_cache_length

    def _append_kv_cache(
        self,
        attention: BaselineSelfAttention,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        batch, heads, chunk_length, head_dim = k.shape
        start = self._kv_cache_length
        end = start + chunk_length
        if end > self.config.seq_len:
            raise ValueError(
                f"KV cache capacity exceeded ({end} > {self.config.seq_len})"
            )

        expected_shape = (batch, heads, self.config.seq_len, head_dim)
        cache_key = id(attention)
        cached = self._kv_cache.get(cache_key)
        if cached is None:
            cached = (
                torch.empty(expected_shape, device=k.device, dtype=k.dtype),
                torch.empty(expected_shape, device=v.device, dtype=v.dtype),
            )
            self._kv_cache[cache_key] = cached
        elif (
            cached[0].shape != expected_shape
            or cached[0].device != k.device
            or cached[0].dtype != k.dtype
        ):
            raise RuntimeError(
                "KV cache does not match this input; reset it before changing "
                "batch size, device, or dtype"
            )

        cached_k, cached_v = cached
        cached_k[:, :, start:end].copy_(k)
        cached_v[:, :, start:end].copy_(v)
        return cached_k[:, :, :end], cached_v[:, :, :end], start

    def _prepare_attention_args(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], bool, Optional[torch.Tensor]]:
        """Inspect and cache the SDPA and output masks once per input mask."""
        if valid_token_mask is None:
            return None, self.config.causal, None

        try:
            version: Optional[int] = valid_token_mask._version
        except RuntimeError:
            # Inference tensors deliberately do not expose a version counter.
            version = None
        seq_len = x.shape[1]

        # Retain the tensor object to prevent stale hits if CUDA reuses a data
        # pointer. The version catches ordinary in-place mask updates as well.
        cached = getattr(self, "_optimized_attention_args_cache", None)
        if (
            version is not None
            and cached is not None
            and cached[0] is valid_token_mask
            and cached[1] == version
            and cached[2] == seq_len
        ):
            return cached[3], cached[4], cached[5]

        if bool(valid_token_mask.all().item()):
            result: Tuple[
                Optional[torch.Tensor], bool, Optional[torch.Tensor]
            ] = (None, self.config.causal, None)
        else:
            key_mask = valid_token_mask[:, None, None, :]
            if self.config.causal:
                if self._optimized_causal_mask is None:
                    raise ValueError(
                        "padded causal attention is disabled for sequences "
                        "longer than 4096 because a dense combined mask would "
                        "exceed the bounded-memory execution plan"
                    )
                causal_mask = self._optimized_causal_mask[:seq_len, :seq_len]
                attn_mask = key_mask & causal_mask[None, None, :, :]
            else:
                attn_mask = key_mask

            # Invalid tokens cannot influence valid tokens because they remain
            # masked as keys in every layer. Clear them only at model output.
            invalid_output_mask = ~valid_token_mask[..., None]
            result = (attn_mask, False, invalid_output_mask)

        if version is not None:
            self._optimized_attention_args_cache = (
                valid_token_mask,
                version,
                seq_len,
                *result,
            )
        return result

    @staticmethod
    def _linear_cache_key(
        linear: nn.Linear,
        compute_dtype: torch.dtype,
    ) -> Tuple[object, ...]:
        bias = linear.bias
        return (
            linear.weight.data_ptr(),
            linear.weight._version,
            bias.data_ptr() if bias is not None else None,
            bias._version if bias is not None else None,
            linear.weight.device,
            linear.weight.dtype,
            compute_dtype,
        )

    def _cached_linear_parameters(
        self,
        linear: nn.Linear,
        compute_dtype: torch.dtype,
        *,
        transpose_weight: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return version-checked inference parameters in the compute dtype."""
        cache_name = (
            "_optimized_transposed_linear_cache"
            if transpose_weight
            else "_optimized_linear_cache"
        )
        cache_key = self._linear_cache_key(linear, compute_dtype)
        cached = getattr(linear, cache_name, None)
        if cached is None or cached[0] != cache_key:
            weight = linear.weight.to(dtype=compute_dtype)
            if transpose_weight:
                weight = weight.t().contiguous()
            bias = (
                linear.bias.to(dtype=compute_dtype)
                if linear.bias is not None
                else None
            )
            cached = (cache_key, weight, bias)
            setattr(linear, cache_name, cached)
        return cached[1], cached[2]

    @staticmethod
    def _layer_norm_uses_affine(layer_norm: nn.LayerNorm) -> bool:
        """Cache whether a LayerNorm's affine transform is non-identity."""
        weight = layer_norm.weight
        bias = layer_norm.bias
        if weight is None or bias is None:
            return False
        try:
            cache_key = (
                weight.data_ptr(),
                weight._version,
                bias.data_ptr(),
                bias._version,
            )
        except RuntimeError:
            # Inference tensors do not expose version counters. Keep the
            # general affine path rather than cache a result that could become
            # stale after an unobservable in-place parameter update.
            return True
        cached = getattr(layer_norm, "_optimized_affine_cache", None)
        if cached is None or cached[0] != cache_key:
            # This synchronizes only when parameters change, before CUDA Graph
            # capture. Default LayerNorm parameters are exactly one and zero.
            is_identity = bool(torch.all(weight == 1).item()) and bool(
                torch.all(bias == 0).item()
            )
            cached = (cache_key, not is_identity)
            layer_norm._optimized_affine_cache = cached
        return cached[1]

    def _packed_qkv(
        self,
        attention: BaselineSelfAttention,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # The baseline launches three separate Linear ops for Q, K, and V. Stack
        # those weights once so inference can compute QKV with one larger GEMM.
        q_proj = attention.q_proj
        k_proj = attention.k_proj
        v_proj = attention.v_proj

        cache_key = (
            q_proj.weight.data_ptr(),
            q_proj.weight._version,
            k_proj.weight.data_ptr(),
            k_proj.weight._version,
            v_proj.weight.data_ptr(),
            v_proj.weight._version,
            q_proj.bias.data_ptr(),
            q_proj.bias._version,
            k_proj.bias.data_ptr(),
            k_proj.bias._version,
            v_proj.bias.data_ptr(),
            v_proj.bias._version,
            q_proj.weight.dtype,
            q_proj.weight.device.type,
            q_proj.weight.device.index,
            self._mixed_precision_dtype,
        )
        cached_qkv = getattr(attention, "_optimized_qkv_cache", None)
        if cached_qkv is None or cached_qkv[0] != cache_key:
            # These cached tensors are derived from the existing parameters, so
            # parameter names stay compatible with the benchmark weight copier.
            weight = torch.cat(
                (q_proj.weight, k_proj.weight, v_proj.weight), dim=0
            ).contiguous()
            bias = torch.cat(
                (q_proj.bias, k_proj.bias, v_proj.bias), dim=0
            ).contiguous()
            if self._mixed_precision_dtype is not None:
                weight = weight.to(dtype=self._mixed_precision_dtype)
                bias = bias.to(dtype=self._mixed_precision_dtype)
            cached_qkv = (cache_key, weight, bias)
            attention._optimized_qkv_cache = cached_qkv
        return cached_qkv[1], cached_qkv[2]

    @staticmethod
    def _transposed_packed_qkv(
        attention: BaselineSelfAttention,
        packed_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Cache packed QKV in the KxN layout consumed by the Triton kernel."""
        cached = getattr(attention, "_optimized_qkv_kn_cache", None)
        # Retaining the source tensor prevents a stale hit if CUDA later reuses
        # its data pointer after source parameters are updated and repacked.
        if cached is None or cached[0] is not packed_weight:
            cached = (packed_weight, packed_weight.t().contiguous())
            attention._optimized_qkv_kn_cache = cached
        return cached[1]

    def _uses_grouped_attention_projections(
        self,
        attention: BaselineSelfAttention,
        x: torch.Tensor,
    ) -> bool:
        """Gate the measured shape-8 QKV/output kernels and their fallback."""
        return (
            self._shape_execution_plan.projection_strategy == "grouped_1024"
            and self._triton_fused_ffn_enabled
            and self._mixed_precision_dtype == torch.float16
            and x.device.type == "cuda"
            and not self._kv_cache_enabled
            and x.numel() // x.shape[-1] >= 8192
            and attention.d_model == 1024
            and attention.out_proj.bias is not None
            and self._supports_shape_tuned_kernels(x)
        )

    def _fast_linear(
        self,
        linear: nn.Linear,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Linear using immutable cached mixed-precision inference weights."""
        compute_dtype = self._mixed_precision_dtype
        if compute_dtype is None:
            return linear(x)
        weight, bias = self._cached_linear_parameters(linear, compute_dtype)
        return F.linear(x, weight, bias)

    def _fast_attention(
        self,
        attention: BaselineSelfAttention,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        # On Ampere the attention projections gain a few percent from FP16
        # accumulation and stay within the benchmark's error budget.  Applying
        # it to the much wider FFN reductions does not, so scope the global
        # backend switch tightly and restore it even if an operation fails.
        use_fp16_accumulation = (
            self._attention_fp16_accumulation_enabled
            and self._mixed_precision_dtype == torch.float16
            and not self.config.causal
            and not self._kv_cache_enabled
        )
        if not use_fp16_accumulation:
            return self._fast_attention_impl(
                attention, x, attn_mask, is_causal
            )

        previous = torch.backends.cuda.matmul.allow_fp16_accumulation
        try:
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            return self._fast_attention_impl(
                attention, x, attn_mask, is_causal
            )
        finally:
            torch.backends.cuda.matmul.allow_fp16_accumulation = previous

    def _fast_attention_impl(
        self,
        attention: BaselineSelfAttention,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv_weight, qkv_bias = self._packed_qkv(attention)
        use_grouped_projections = self._uses_grouped_attention_projections(
            attention, x
        )

        # One packed projection cuts launch overhead and memory traffic compared
        # with separate q_proj/k_proj/v_proj calls. Shape 8 additionally uses a
        # measured short-accumulation kernel for this unusually wide GEMM.
        if use_grouped_projections:
            qkv = triton_grouped_qkv_1024(
                x,
                self._transposed_packed_qkv(attention, qkv_weight),
                qkv_bias,
            )
        else:
            qkv = F.linear(x, qkv_weight, qkv_bias)
        qkv = qkv.view(
            batch, seq_len, 3, attention.num_heads, attention.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        if self._kv_cache_enabled:
            k, v, query_start = self._append_kv_cache(attention, k, v)
            key_length = k.shape[-2]
            if query_start == 0:
                # Prompt prefill begins at position zero, matching SDPA's
                # built-in causal-mask alignment.
                is_causal = True
            elif seq_len == 1:
                # Every cached position is valid for the newest single token.
                is_causal = False
            else:
                query_positions = query_start + torch.arange(
                    seq_len, device=x.device
                )
                key_positions = torch.arange(key_length, device=x.device)
                attn_mask = (
                    key_positions[None, :] <= query_positions[:, None]
                )[None, None, :, :]
                is_causal = False

        # On NVIDIA GPUs this can route to fused/memory-efficient attention
        # kernels instead of materializing the full [B, H, S, S] scores/probs.
        if max(q.shape[-2], k.shape[-2]) >= 8192:
            # This path must never fall back to the math implementation, which
            # materializes an O(S^2) attention tensor. The current Windows
            # build provides the CUTLASS memory-efficient backend; cuDNN is an
            # additional fused fallback on builds that support it.
            with sdpa_kernel(
                backends=[
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.CUDNN_ATTENTION,
                ]
            ):
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
        elif self._planned_efficient_attention_available(
            q, k, v, attn_mask, is_causal
        ):
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                context = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )
        else:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            )
        context = context.transpose(1, 2).reshape(batch, seq_len, attention.d_model)
        if use_grouped_projections:
            weight, bias = self._cached_linear_parameters(
                attention.out_proj,
                torch.float16,
                transpose_weight=True,
            )
            assert bias is not None
            return triton_grouped_square_1024_out(context, weight, bias)
        return self._fast_linear(attention.out_proj, context)

    def _fast_block(
        self,
        layer: BaselineTransformerBlock,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> torch.Tensor:
        # Preserve the reference block formula while swapping in the faster
        # attention implementation: LN -> attention -> residual, then FFN.
        x = x + self._fast_attention(
            layer.attention, layer.norm1(x), attn_mask, is_causal
        )
        hidden = self._fast_ffn_in_gelu(layer, layer.norm2(x))
        x = x + self._fast_ffn_out(layer, hidden)
        return x

    def _fast_ffn_in_gelu(
        self,
        layer: BaselineTransformerBlock,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse the mixed-precision FFN input GEMM, bias, and GELU."""
        compute_dtype = self._mixed_precision_dtype
        linear = layer.ffn_in
        if (
            self._triton_fused_ffn_enabled
            and compute_dtype == torch.float16
            and x.device.type == "cuda"
            and linear.bias is not None
        ):
            weight, bias = self._cached_linear_parameters(
                linear,
                compute_dtype,
                transpose_weight=True,
            )
            assert bias is not None
            rows = x.numel() // x.shape[-1]
            if (
                self._shape_execution_plan.ffn_strategy
                == "grouped_square_1024"
                and self._supports_shape_tuned_kernels(x)
                and rows >= 2048
                and linear.in_features == 1024
                and linear.out_features == 1024
            ):
                return triton_grouped_square_1024_gelu(x, weight, bias)
            if (
                rows == 1024
                and linear.in_features == 512
                and linear.out_features == 2048
                and self._supports_shape_tuned_kernels(x)
            ):
                return triton_grouped_ffn_gelu(x, weight, bias)
            return triton_ffn_gelu(x, weight, bias)

        if (
            compute_dtype is None
            or linear.bias is None
            or not hasattr(torch, "_addmm_activation")
        ):
            return F.gelu(linear(x), approximate="none")

        # The fused primitive is internal, so explicitly supply the same cached
        # inference parameters used by the other optimized linear operations.
        weight, bias = self._cached_linear_parameters(
            linear,
            compute_dtype,
        )
        assert bias is not None
        x_2d = x.reshape(-1, x.shape[-1]).to(dtype=compute_dtype)
        hidden = torch._addmm_activation(
            bias,
            x_2d,
            weight.t(),
            use_gelu=True,
        )
        return hidden.view(*x.shape[:-1], linear.out_features)

    def _supports_shape_tuned_kernels(self, x: torch.Tensor) -> bool:
        """Use measured shape specializations only on the target SM 8.6 GPU."""
        device_key = (x.device.type, x.device.index)
        cached = getattr(self, "_optimized_shape_kernel_device_cache", None)
        if cached is None or cached[0] != device_key:
            capability = torch.cuda.get_device_capability(x.device)
            cached = (device_key, capability == (8, 6))
            self._optimized_shape_kernel_device_cache = cached
        return cached[1]

    def _planned_efficient_attention_available(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
    ) -> bool:
        """Validate and cache shape 11's backend choice before graph capture."""
        if (
            self._shape_execution_plan.attention_strategy != "efficient"
            or self._kv_cache_enabled
            or q.shape[0] != self.config.batch_size
            or q.shape[-2] != self.config.seq_len
            or not self._supports_shape_tuned_kernels(q)
        ):
            return False

        cache_key = (
            tuple(q.shape),
            tuple(k.shape),
            tuple(v.shape),
            q.dtype,
            q.device,
            attn_mask is not None,
            is_causal,
        )
        cache = getattr(self, "_optimized_efficient_attention_cache", None)
        if cache is None:
            cache = {}
            self._optimized_efficient_attention_cache = cache
        if cache_key not in cache:
            params_type = getattr(torch.backends.cuda, "SDPAParams", None)
            checker = getattr(
                torch.backends.cuda, "can_use_efficient_attention", None
            )
            if params_type is None or checker is None:
                cache[cache_key] = False
            else:
                try:
                    params = params_type(
                        q, k, v, attn_mask, 0.0, is_causal, False
                    )
                    cache[cache_key] = bool(checker(params))
                except (RuntimeError, TypeError):
                    cache[cache_key] = False
        return bool(cache[cache_key])

    def _fast_ffn_out(
        self,
        layer: BaselineTransformerBlock,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Use the grouped-accumulation kernel for the tuned FFN output."""
        linear = layer.ffn_out
        if (
            self._triton_fused_ffn_enabled
            and self._mixed_precision_dtype == torch.float16
            and x.device.type == "cuda"
            and linear.bias is not None
        ):
            is_square_1024 = (
                self._shape_execution_plan.ffn_strategy
                == "grouped_square_1024"
                and x.numel() // x.shape[-1] >= 2048
                and linear.in_features == 1024
                and linear.out_features == 1024
            )
            is_legacy_tuned_shape = (
                x.numel() // x.shape[-1] == 1024
                and linear.in_features == 2048
                and linear.out_features == 512
            )
            if (
                (is_square_1024 or is_legacy_tuned_shape)
                and self._supports_shape_tuned_kernels(x)
            ):
                weight, bias = self._cached_linear_parameters(
                    linear,
                    torch.float16,
                    transpose_weight=True,
                )
                assert bias is not None
                if is_square_1024:
                    return triton_grouped_square_1024_out(x, weight, bias)
                return triton_grouped_ffn_out(x, weight, bias)
        return self._fast_linear(linear, x)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Keep a baseline fallback for Mac/CPU runs; use the faster SDPA path
        # on CUDA, where the RTX 3060 Laptop GPU can benefit from fused kernels.
        if x.device.type != "cuda":
            if self._kv_cache_enabled:
                raise RuntimeError("KV caching is implemented for CUDA inference")
            return super().forward(x, valid_token_mask)

        forward_cuda = (
            self._forward_cuda_batch_chunked
            if self._large_batch_chunk_size
            and x.shape[0] > self._large_batch_chunk_size
            and not self._kv_cache_enabled
            else self._forward_cuda
        )
        if self._mixed_precision_dtype is not None:
            with torch.autocast(
                device_type="cuda", dtype=self._mixed_precision_dtype
            ):
                return forward_cuda(x, valid_token_mask)
        return forward_cuda(x, valid_token_mask)

    def _forward_cuda_batch_chunked(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run the exact shape-6 batch in bounded-memory independent slices."""
        inner_mask = valid_token_mask
        if valid_token_mask is not None:
            # Reuse the existing version-checked mask cache. For the official
            # all-valid case this synchronizes once during warmup, after which
            # each chunk can use the cheaper no-padding path.
            attn_mask, _, invalid_output_mask = self._prepare_attention_args(
                x, valid_token_mask
            )
            if attn_mask is None and invalid_output_mask is None:
                inner_mask = None

        output = torch.empty_like(x)
        chunk_size = self._large_batch_chunk_size
        for start in range(0, x.shape[0], chunk_size):
            end = min(start + chunk_size, x.shape[0])
            mask_slice = (
                None if inner_mask is None else inner_mask[start:end]
            )
            output[start:end].copy_(
                self._forward_cuda(x[start:end], mask_slice)
            )
        return output

    def _forward_cuda(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CUDA implementation, optionally entered under an autocast context."""

        if (
            self._kv_cache_enabled
            and valid_token_mask is not None
            and not bool(valid_token_mask.all().item())
        ):
            raise ValueError("KV-cached decoding does not support padding")

        attn_mask, is_causal, invalid_output_mask = self._prepare_attention_args(
            x, valid_token_mask
        )

        # Each pre-norm block otherwise reads/writes the complete activation
        # once for the residual addition and again for LayerNorm.  For the
        # default fp16 CUDA path, combine those operations in one Triton kernel
        # and emit the normalized tensor directly in the next GEMM's dtype.
        if (
            self._triton_fused_norm_enabled
            and self._mixed_precision_dtype == torch.float16
            and x.dtype == torch.float32
            and not self._kv_cache_enabled
        ):
            return self._forward_cuda_fused_norm(
                x, attn_mask, is_causal, invalid_output_mask
            )

        for layer in self.layers:
            x = self._fast_block(layer, x, attn_mask, is_causal)
        if self._kv_cache_enabled:
            self._kv_cache_length += x.shape[1]
        x = self.final_norm(x)
        if invalid_output_mask is not None:
            x = x.masked_fill(invalid_output_mask, 0)
        return x

    def _forward_cuda_fused_norm(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        is_causal: bool,
        invalid_output_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mixed-precision path with fused residual-add/LayerNorm kernels."""
        # The first normalization has no preceding update to fuse.  Every
        # subsequent normalization is paired with the residual that produces
        # its input, including final_norm after the last FFN.
        first_norm = self.layers[0].norm1
        norm1 = triton_layer_norm(
            x,
            first_norm.weight,
            first_norm.bias,
            first_norm.eps,
            torch.float16,
            apply_affine=self._layer_norm_uses_affine(first_norm),
        )
        last_index = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            attention_update = self._fast_attention(
                layer.attention, norm1, attn_mask, is_causal
            )
            x, norm2 = fused_residual_layer_norm(
                x,
                attention_update,
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
                torch.float16,
                apply_affine=self._layer_norm_uses_affine(layer.norm2),
            )

            hidden = self._fast_ffn_in_gelu(layer, norm2)
            ffn_update = self._fast_ffn_out(layer, hidden)
            following_norm = (
                self.layers[index + 1].norm1
                if index < last_index
                else self.final_norm
            )
            norm_dtype = torch.float16 if index < last_index else torch.float32
            x, norm1 = fused_residual_layer_norm(
                x,
                ffn_update,
                following_norm.weight,
                following_norm.bias,
                following_norm.eps,
                norm_dtype,
                store_residual=index < last_index,
                apply_affine=self._layer_norm_uses_affine(following_norm),
            )

        if invalid_output_mask is not None:
            norm1 = norm1.masked_fill(invalid_output_mask, 0)
        return norm1


def copy_model_weights(
    baseline: nn.Module, optimized: nn.Module, strict: bool = True
) -> None:
    """Copy identical weights into both implementations for a fair comparison."""
    state_dict = copy.deepcopy(baseline.state_dict())
    incompatible = optimized.load_state_dict(state_dict, strict=strict)
    if not strict:
        if incompatible.missing_keys:
            print(f"[warning] missing optimized keys: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            print(f"[warning] unexpected optimized keys: {incompatible.unexpected_keys}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_name]


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size, config.seq_len, device=device, dtype=torch.bool
        )
        return x, valid_token_mask

    min_valid = max(1, int(round(config.seq_len * (1.0 - padding_ratio))))
    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )
    positions = torch.arange(config.seq_len, device=device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


@dataclass
class AccuracyResult:
    passed: bool
    total_elements: int
    failed_elements: int
    max_abs_error: float
    max_relative_error: float
    mean_abs_error: float
    failed_feature_dims: List[int]
    worst_index: Tuple[int, ...]
    reference_at_worst: float
    optimized_at_worst: float


def compare_outputs(
    reference: torch.Tensor,
    optimized: torch.Tensor,
    rtol: float,
    atol: float,
) -> AccuracyResult:
    if reference.shape != optimized.shape:
        raise AssertionError(
            f"shape mismatch: baseline={tuple(reference.shape)}, "
            f"optimized={tuple(optimized.shape)}"
        )
    if reference.dtype != optimized.dtype:
        print(
            f"[warning] dtype mismatch: baseline={reference.dtype}, "
            f"optimized={optimized.dtype}"
        )

    ref = reference.detach().float()
    opt = optimized.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    # Exact interpretation of the requested OR condition. torch.isclose uses
    # atol + rtol * abs(ref), which is slightly more permissive and is not used.
    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed_mask = ~passed_mask
    failed_elements = int(failed_mask.sum().item())
    total_elements = reference.numel()

    flat_worst = int(abs_error.reshape(-1).argmax().item())
    worst_index_list = []
    remaining = flat_worst
    for size in reversed(reference.shape):
        worst_index_list.append(remaining % size)
        remaining //= size
    worst_index = tuple(reversed(worst_index_list))

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    # Summarize failures by the last/output-feature dimension.
    if reference.ndim == 0:
        failed_feature_dims = [0] if failed_elements else []
    elif reference.ndim == 1:
        failed_feature_dims = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
    else:
        reduce_dims = tuple(range(reference.ndim - 1))
        failed_by_feature = failed_mask.any(dim=reduce_dims)
        failed_feature_dims = (
            torch.nonzero(failed_by_feature, as_tuple=False).flatten().tolist()
        )

    return AccuracyResult(
        passed=failed_elements == 0,
        total_elements=total_elements,
        failed_elements=failed_elements,
        max_abs_error=float(abs_error.max().item()),
        max_relative_error=float(relative_error.max().item()),
        mean_abs_error=float(abs_error.mean().item()),
        failed_feature_dims=failed_feature_dims,
        worst_index=worst_index,
        reference_at_worst=float(ref[worst_index].item()),
        optimized_at_worst=float(opt[worst_index].item()),
    )


def run_accuracy_tests(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> bool:
    print("\n=== Accuracy check ===")
    print(f"criterion: abs_error <= {atol:g} OR relative_error <= {rtol:.2%}")

    all_passed = True
    global_max_abs = 0.0
    global_max_rel = 0.0
    total_failed = 0
    total_elements = 0

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )
            reference = baseline(x, valid_mask)
            candidate = optimized(x, valid_mask)
            result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)

            all_passed &= result.passed
            global_max_abs = max(global_max_abs, result.max_abs_error)
            global_max_rel = max(global_max_rel, result.max_relative_error)
            total_failed += result.failed_elements
            total_elements += result.total_elements

            status = "PASS" if result.passed else "FAIL"
            print(
                f"trial {trial + 1:02d}/{trials}: {status} | "
                f"max_abs={result.max_abs_error:.6g} | "
                f"max_rel={result.max_relative_error:.6g} | "
                f"failed={result.failed_elements}/{result.total_elements}"
            )

            if not result.passed:
                preview = result.failed_feature_dims[:16]
                suffix = "..." if len(result.failed_feature_dims) > len(preview) else ""
                print(
                    f"  worst_index={result.worst_index}, "
                    f"baseline={result.reference_at_worst:.8g}, "
                    f"optimized={result.optimized_at_worst:.8g}"
                )
                print(f"  failed output feature dims={preview}{suffix}")

    print(
        f"summary: {'PASS' if all_passed else 'FAIL'} | "
        f"max_abs={global_max_abs:.6g} | max_rel={global_max_rel:.6g} | "
        f"failed={total_failed}/{total_elements}"
    )
    return all_passed


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]

            torch.cuda.synchronize(device)
            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()
            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def benchmark_models(
    baseline: nn.Module,
    optimized: nn.Module,
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
    cuda_graphs: bool,
) -> None:
    print("\n=== Performance benchmark ===")
    print("timing excludes random-data generation and uses a fixed input")
    if device.type == "cuda":
        print("CUDA latency is measured with torch.cuda.Event on the current stream")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    # Warm up both models before collecting any timing data.
    warmup_model(baseline, x, valid_mask, warmup, device)
    warmup_model(optimized, x, valid_mask, warmup, device)

    timed_optimized, graph_enabled, graph_reason = maybe_cuda_graph(
        optimized, x, valid_mask, enabled=cuda_graphs
    )
    print(
        f"CUDA Graphs: {'enabled' if graph_enabled else 'not enabled'} "
        f"({graph_reason})"
    )
    if graph_enabled:
        # Exercise graph replay before collecting samples. Capture and setup
        # time remain outside the reported steady-state inference latency.
        warmup_model(timed_optimized, x, valid_mask, 3, device)

    baseline_samples: List[float] = []
    optimized_samples: List[float] = []

    # Alternate measurement order to reduce thermal/clock-order bias.
    for round_index in range(rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )
            optimized_samples.extend(
                benchmark_once(timed_optimized, x, valid_mask, repeats, device)
            )
        else:
            optimized_samples.extend(
                benchmark_once(timed_optimized, x, valid_mask, repeats, device)
            )
            baseline_samples.extend(
                benchmark_once(baseline, x, valid_mask, repeats, device)
            )

    baseline_result = TimingResult(baseline_samples)
    optimized_result = TimingResult(optimized_samples)
    speedup = baseline_result.median_ms / optimized_result.median_ms
    throughput_gain = (speedup - 1.0) * 100.0
    latency_reduction = (
        1.0 - optimized_result.median_ms / baseline_result.median_ms
    ) * 100.0
    tokens_per_call = config.batch_size * config.seq_len
    baseline_tokens_per_second = tokens_per_call * 1000.0 / baseline_result.median_ms
    optimized_tokens_per_second = tokens_per_call * 1000.0 / optimized_result.median_ms

    print(
        f"baseline : median={baseline_result.median_ms:.4f} ms | "
        f"mean={baseline_result.mean_ms:.4f} ms | "
        f"p90={baseline_result.p90_ms:.4f} ms | "
        f"min={baseline_result.min_ms:.4f} ms | "
        f"throughput={baseline_tokens_per_second:.2f} token/s"
    )
    print(
        f"optimized: median={optimized_result.median_ms:.4f} ms | "
        f"mean={optimized_result.mean_ms:.4f} ms | "
        f"p90={optimized_result.p90_ms:.4f} ms | "
        f"min={optimized_result.min_ms:.4f} ms | "
        f"throughput={optimized_tokens_per_second:.2f} token/s"
    )
    print(f"speedup  : {speedup:.3f}x based on median latency")
    print(
        f"rate gain: {throughput_gain:+.1f}% throughput | "
        f"{latency_reduction:.1f}% lower median latency"
    )


def maybe_compile(model: nn.Module, enabled: bool, mode: str) -> nn.Module:
    if not enabled:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a baseline and optimized PyTorch Transformer"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--shape-id",
        type=int,
        choices=tuple(OFFICIAL_BENCHMARK_SHAPES),
        help=(
            "use one official benchmark shape; this overrides batch size, "
            "sequence length, dimensions, heads, layers, and causal mode"
        ),
    )
    parser.add_argument(
        "--list-shapes",
        action="store_true",
        help="print the official benchmark shape table and exit",
    )
    parser.add_argument(
        "--batch-chunk-size",
        type=int,
        default=None,
        help=(
            "maximum batch processed at once; shape 6 defaults to a safe 1024, "
            "zero disables chunking"
        ),
    )

    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, cuda:0, ..."
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("auto", "none", "float16", "bfloat16"),
        default="auto",
        help=(
            "optimized CUDA compute dtype; auto uses float16 when the GPU, "
            "input scale, and accuracy tolerances support it"
        ),
    )
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--input-scale", type=float, default=1.0)

    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--benchmark-on-failure", action="store_true")

    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--cuda-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="capture fixed, all-valid optimized CUDA inference (default: enabled)",
    )
    parser.add_argument(
        "--triton-fused-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "fuse residual-add and LayerNorm when Triton is installed "
            "(default: enabled)"
        ),
    )
    parser.add_argument(
        "--triton-fused-ffn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use shape-tuned Triton FFN kernels (default: enabled)",
    )
    parser.add_argument(
        "--attention-fp16-accumulation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use faster FP16 accumulation only for non-causal attention "
            "projections (default: enabled)"
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument("--non-strict-weight-copy", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable/disable TF32 on CUDA for both implementations",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("padding_ratio must be in [0, 1)")
    if args.input_scale <= 0:
        raise ValueError("input_scale must be positive")
    if args.accuracy_trials <= 0:
        raise ValueError("accuracy_trials must be positive")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.repeats <= 0 or args.benchmark_rounds <= 0:
        raise ValueError("repeats and benchmark_rounds must be positive")
    if args.batch_chunk_size is not None and args.batch_chunk_size < 0:
        raise ValueError("--batch-chunk-size must be non-negative")
    if device.type == "cpu" and dtype == torch.float16:
        print("[warning] float16 CPU kernels may be unsupported or slow")
    if args.mixed_precision not in ("auto", "none") and device.type != "cuda":
        raise ValueError("--mixed-precision requires a CUDA device")


def main() -> int:
    args = parse_args()
    if args.list_shapes:
        print_official_benchmark_shapes()
        return 0
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    if args.shape_id is not None:
        config = OFFICIAL_BENCHMARK_SHAPES[args.shape_id]
    else:
        config = TransformerConfig(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            d_model=args.d_model,
            num_heads=args.heads,
            ffn_dim=args.ffn_dim,
            num_layers=args.layers,
            causal=args.causal,
        )
    config.validate()
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    shape_execution_plan = optimized.shape_execution_plan
    batch_chunk_size = args.batch_chunk_size
    if batch_chunk_size is None:
        batch_chunk_size = shape_execution_plan.batch_chunk_size
    # Keep direct calls and the outer benchmark wrapper on the same chunk
    # geometry. This also makes --batch-chunk-size 0 genuinely disable the
    # automatic shape-6 plan instead of leaving an inner split active.
    optimized._large_batch_chunk_size = batch_chunk_size
    copy_model_weights(
        baseline,
        optimized,
        strict=not args.non_strict_weight_copy,
    )

    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    mixed_precision_name = args.mixed_precision
    if mixed_precision_name == "auto":
        supports_fast_fp16 = (
            device.type == "cuda"
            and torch.cuda.get_device_capability(device) >= (7, 0)
        )
        accuracy_budget_allows_fp16 = (
            args.atol >= 0.002
            and args.rtol >= 0.02
            and args.input_scale >= 1.0
        )
        mixed_precision_name = (
            "float16"
            if (
                supports_fast_fp16
                and dtype == torch.float32
                and accuracy_budget_allows_fp16
            )
            else "none"
        )
    if mixed_precision_name != "none":
        optimized.enable_mixed_precision(resolve_dtype(mixed_precision_name))
    if not args.triton_fused_norm:
        optimized.enable_triton_fused_norm(False)
    if not args.triton_fused_ffn:
        optimized.enable_triton_fused_ffn(False)
    if not args.attention_fp16_accumulation:
        optimized.enable_attention_fp16_accumulation(False)

    # Compile only after model construction, weight copy, device transfer, and eval().
    baseline = maybe_compile(baseline, args.compile_baseline, args.compile_mode)
    optimized = maybe_compile(optimized, args.compile_user, args.compile_mode)

    print("=== Configuration ===")
    if args.shape_id is not None:
        print(f"official_benchmark_shape={args.shape_id}")
    print(config)
    print(f"device={device}, dtype={dtype}, torch={torch.__version__}")
    print(
        f"optimized_mixed_precision={mixed_precision_name} "
        f"(requested={args.mixed_precision})"
    )
    triton_norm_active = (
        optimized._triton_fused_norm_enabled
        and device.type == "cuda"
        and dtype == torch.float32
        and mixed_precision_name == "float16"
    )
    triton_ffn_active = (
        optimized._triton_fused_ffn_enabled
        and device.type == "cuda"
        and mixed_precision_name == "float16"
    )
    attention_fp16_accumulation_active = (
        optimized._attention_fp16_accumulation_enabled
        and device.type == "cuda"
        and mixed_precision_name == "float16"
        and not config.causal
    )
    print(
        "optimized_triton_fused_norm="
        f"{triton_norm_active} "
        f"(available={TRITON_AVAILABLE})"
    )
    print(
        "optimized_triton_fused_ffn="
        f"{triton_ffn_active} "
        f"(available={TRITON_FFN_AVAILABLE})"
    )
    print(
        "optimized_attention_fp16_accumulation="
        f"{attention_fp16_accumulation_active}"
    )
    print(f"optimized_shape_plan={shape_execution_plan.describe()}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    if batch_chunk_size and batch_chunk_size < config.batch_size:
        all_tokens_valid = args.padding_ratio == 0.0
        baseline = BatchChunkedTransformer(
            baseline,
            batch_chunk_size,
            all_tokens_valid=all_tokens_valid,
            use_cuda_graphs=False,
        )
        optimized = BatchChunkedTransformer(
            optimized,
            batch_chunk_size,
            all_tokens_valid=all_tokens_valid,
            use_cuda_graphs=args.cuda_graphs,
        )
        print(
            "execution_plan=batch_chunked "
            f"(chunk_size={batch_chunk_size}, chunks="
            f"{math.ceil(config.batch_size / batch_chunk_size)}, "
            f"optimized_chunk_graphs={optimized.use_cuda_graphs})"
        )

    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
    )

    if not accuracy_passed and not args.benchmark_on_failure:
        print("\nPerformance benchmark skipped because accuracy validation failed.")
        print("Use --benchmark-on-failure to benchmark an incorrect implementation anyway.")
        return 2

    benchmark_models(
        baseline=baseline,
        optimized=optimized,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
        cuda_graphs=args.cuda_graphs,
    )
    return 0 if accuracy_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
