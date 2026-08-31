"""Bounded-memory batch chunking for very large independent batches."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _ChunkCUDAGraph:
    """Replay a model for one fixed chunk shape and copy out before reuse."""

    def __init__(
        self,
        model: nn.Module,
        example: torch.Tensor,
        warmup_iterations: int = 3,
    ) -> None:
        if not example.is_cuda:
            raise ValueError("chunk CUDA Graphs require CUDA inputs")
        if model.training:
            raise ValueError("chunk CUDA Graphs require model.eval()")

        self.static_input = torch.empty_like(example)
        self.static_input.copy_(example)
        device = example.device
        current_stream = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.inference_mode(), torch.cuda.stream(capture_stream):
            for _ in range(warmup_iterations):
                model(self.static_input, None)
        current_stream.wait_stream(capture_stream)
        torch.cuda.synchronize(device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            self.static_output = model(self.static_input, None)
        torch.cuda.synchronize(device)

    def run_into(self, inputs: torch.Tensor, destination: torch.Tensor) -> None:
        self.static_input.copy_(inputs)
        self.graph.replay()
        # Copy before the next replay overwrites the graph's persistent output.
        destination.copy_(self.static_output)


class BatchChunkedTransformer(nn.Module):
    """Execute independent batch slices while bounding attention memory.

    The wrapper is inference-only. When ``all_tokens_valid`` is true it omits
    the redundant all-True padding mask, allowing one CUDA Graph to be reused
    for every full-sized chunk and another for the final remainder.
    """

    def __init__(
        self,
        model: nn.Module,
        chunk_size: int,
        *,
        all_tokens_valid: bool,
        use_cuda_graphs: bool,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if model.training:
            raise ValueError("batch chunking requires model.eval()")
        self.model = model
        self.chunk_size = chunk_size
        self.all_tokens_valid = all_tokens_valid
        self.use_cuda_graphs = (
            use_cuda_graphs
            and all_tokens_valid
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "CUDAGraph")
        )
        # Signals maybe_cuda_graph() that graph replay is already managed at
        # the reusable chunk level; capturing the enormous outer call would
        # retain an unnecessary full-batch graph output.
        self._uses_internal_cuda_graphs = self.use_cuda_graphs
        self._chunk_graphs: dict[tuple[object, ...], Optional[_ChunkCUDAGraph]] = {}
        self._chunk_graph_failures: dict[tuple[object, ...], str] = {}
        self.training = False

    @property
    def graph_status(self) -> str:
        if not self.use_cuda_graphs:
            return "disabled"
        if self._chunk_graph_failures:
            reasons = sorted(set(self._chunk_graph_failures.values()))
            return "partial fallback: " + "; ".join(reasons)
        return "reusable per-chunk CUDA Graph replay"

    @staticmethod
    def _graph_key(chunk: torch.Tensor) -> tuple[object, ...]:
        return (
            tuple(chunk.shape),
            chunk.dtype,
            chunk.device.type,
            chunk.device.index,
        )

    def _run_chunk(
        self,
        inputs: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        destination: torch.Tensor,
    ) -> None:
        if not self.use_cuda_graphs:
            destination.copy_(self.model(inputs, valid_mask))
            return

        key = self._graph_key(inputs)
        if key not in self._chunk_graphs:
            try:
                self._chunk_graphs[key] = _ChunkCUDAGraph(self.model, inputs)
            except (RuntimeError, ValueError) as error:
                self._chunk_graphs[key] = None
                self._chunk_graph_failures[key] = str(error)

        graph = self._chunk_graphs[key]
        if graph is None:
            destination.copy_(self.model(inputs, None))
        else:
            graph.run_into(inputs, destination)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("batch-chunked Transformer input must be 3-D")
        if self.all_tokens_valid:
            chunk_mask = None
        elif valid_token_mask is None:
            chunk_mask = None
        else:
            chunk_mask = valid_token_mask

        output = torch.empty_like(x)
        for start in range(0, x.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, x.shape[0])
            mask_slice = (
                None if chunk_mask is None else chunk_mask[start:end]
            )
            self._run_chunk(
                x[start:end],
                mask_slice,
                output[start:end],
            )
        return output
