"""CUDA Graph support for fixed-input Transformer inference.

CUDA Graphs replay an already-captured sequence of GPU kernels, reducing the
CPU and driver launch overhead between small Transformer operations.  This
wrapper intentionally targets a fixed tensor object and an all-valid mask.  If
either input object changes, it transparently falls back to the eager model.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _tensor_version(tensor: torch.Tensor) -> Optional[int]:
    try:
        return tensor._version
    except RuntimeError:
        return None


class FixedInputCUDAGraph(nn.Module):
    """Replay one eval-mode CUDA model for the exact captured input objects."""

    def __init__(
        self,
        model: nn.Module,
        example_x: torch.Tensor,
        example_mask: torch.Tensor,
        warmup_iterations: int = 3,
    ) -> None:
        super().__init__()
        if not example_x.is_cuda or not example_mask.is_cuda:
            raise ValueError("CUDA Graph inputs must be CUDA tensors")
        if model.training:
            raise ValueError("CUDA Graph inference requires model.eval()")
        if not bool(example_mask.all().item()):
            raise ValueError("fixed-input CUDA Graph path requires an all-valid mask")

        self.model = model
        self._captured_x = example_x
        self._captured_mask = example_mask
        self._captured_x_version = _tensor_version(example_x)
        self._captured_mask_version = _tensor_version(example_mask)
        device = example_x.device

        # Warm up allocator, autocast, cuBLAS, and SDPA state on a side stream.
        current_stream = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.inference_mode(), torch.cuda.stream(capture_stream):
            for _ in range(warmup_iterations):
                model(example_x, None)
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(device)

        self._graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self._graph):
            # An all-True mask is semantically identical to no mask and avoids
            # capturing mask inspection or host synchronization.
            self._captured_output = model(example_x, None)
        torch.cuda.synchronize(device)

    def _matches_capture(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> bool:
        return (
            x is self._captured_x
            and valid_token_mask is self._captured_mask
            and _tensor_version(x) == self._captured_x_version
            and _tensor_version(valid_token_mask) == self._captured_mask_version
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self._matches_capture(x, valid_token_mask):
            return self.model(x, valid_token_mask)

        self._graph.replay()
        # CUDA Graph outputs use persistent storage. Clone so callers receive
        # normal independent output tensors on repeated calls.
        return self._captured_output.clone()


def maybe_cuda_graph(
    model: nn.Module,
    x: torch.Tensor,
    valid_token_mask: torch.Tensor,
    enabled: bool,
) -> Tuple[nn.Module, bool, str]:
    """Return a graphed model when supported, otherwise an eager fallback."""
    if not enabled:
        return model, False, "disabled by --no-cuda-graphs"
    if not x.is_cuda:
        return model, False, "CUDA Graphs require a CUDA input"
    if not hasattr(torch.cuda, "CUDAGraph"):
        return model, False, "this PyTorch build has no CUDA Graph support"
    if not bool(valid_token_mask.all().item()):
        return model, False, "padding masks use the eager optimized path"
    if bool(getattr(model, "_kv_cache_enabled", False)):
        return model, False, "stateful KV-cache decoding uses the eager path"

    try:
        with torch.inference_mode():
            eager_output = model(x, None)
        wrapped = FixedInputCUDAGraph(model, x, valid_token_mask)
        with torch.inference_mode():
            graph_output = wrapped(x, valid_token_mask)
    except (RuntimeError, ValueError) as error:
        return model, False, f"capture failed: {error}"

    if not torch.equal(eager_output, graph_output):
        max_error = float(
            (eager_output.float() - graph_output.float()).abs().max().item()
        )
        return model, False, f"replay validation failed (max_abs={max_error:.3g})"
    return wrapped, True, "fixed-input CUDA Graph replay"
