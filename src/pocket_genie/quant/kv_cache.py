from __future__ import annotations

import dataclasses
import hashlib
from typing import Any, Sequence

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclasses.dataclass(slots=True)
class KVCompressionStats:
    seq_len: int
    sink_tokens: int
    quantized_middle_tokens: int
    recent_window_tokens: int
    fp16_mb: float
    compressed_mb: float
    key_mse: float
    value_mse: float

    @property
    def saved_mb(self) -> float:
        return max(0.0, round(self.fp16_mb - self.compressed_mb, 4))


def quantize_k_per_channel_int4(matrix: Sequence[Sequence[float]] | Any) -> tuple[list[list[float]], float]:
    """Per-channel asymmetric INT4 quantization across sequence dimension for Keys (K)."""
    if torch is not None and isinstance(matrix, torch.Tensor):
        if matrix.numel() == 0:
            return [], 0.0
        mat_fp = matrix.detach().float()
        cmin = mat_fp.amin(dim=0, keepdim=True)
        cmax = mat_fp.amax(dim=0, keepdim=True)
        scale = (cmax - cmin).clamp(min=1e-6) / 15.0
        q = torch.round((mat_fp - cmin) / scale).clamp(0, 15)
        deq = cmin + q * scale
        mse = float((mat_fp - deq).pow(2).mean().item())
        return deq.tolist(), round(mse, 6)

    if not matrix or not matrix[0]:
        return [], 0.0
    seq_len = len(matrix)
    dim = len(matrix[0])
    reconstructed = [[0.0] * dim for _ in range(seq_len)]
    sq_err = 0.0

    for c in range(dim):
        col = [float(matrix[t][c]) for t in range(seq_len)]
        cmin = min(col)
        cmax = max(col)
        span = max(cmax - cmin, 1e-6)
        scale = span / 15.0
        for t in range(seq_len):
            q = max(0, min(15, round((col[t] - cmin) / scale)))
            deq = cmin + q * scale
            reconstructed[t][c] = deq
            diff = col[t] - deq
            sq_err += diff * diff

    mse = sq_err / float(seq_len * dim)
    return reconstructed, round(mse, 6)


def quantize_v_per_token_int4(matrix: Sequence[Sequence[float]] | Any) -> tuple[list[list[float]], float]:
    """Per-token asymmetric INT4 quantization across head dimension for Values (V)."""
    if torch is not None and isinstance(matrix, torch.Tensor):
        if matrix.numel() == 0:
            return [], 0.0
        mat_fp = matrix.detach().float()
        rmin = mat_fp.amin(dim=-1, keepdim=True)
        rmax = mat_fp.amax(dim=-1, keepdim=True)
        scale = (rmax - rmin).clamp(min=1e-6) / 15.0
        q = torch.round((mat_fp - rmin) / scale).clamp(0, 15)
        deq = rmin + q * scale
        mse = float((mat_fp - deq).pow(2).mean().item())
        return deq.tolist(), round(mse, 6)

    if not matrix or not matrix[0]:
        return [], 0.0
    seq_len = len(matrix)
    dim = len(matrix[0])
    reconstructed: list[list[float]] = []
    sq_err = 0.0

    for t in range(seq_len):
        row = [float(x) for x in matrix[t]]
        rmin = min(row)
        rmax = max(row)
        span = max(rmax - rmin, 1e-6)
        scale = span / 15.0
        deq_row: list[float] = []
        for val in row:
            q = max(0, min(15, round((val - rmin) / scale)))
            deq = rmin + q * scale
            deq_row.append(deq)
            diff = val - deq
            sq_err += diff * diff
        reconstructed.append(deq_row)

    mse = sq_err / float(seq_len * dim)
    return reconstructed, round(mse, 6)


class BOSSinkINT4KVCache:
    """Tri-region KV cache manager with static prefix pinning and BOS-sink-protected INT4 middle compression."""

    def __init__(
        self,
        sink_tokens: int = 4,
        recent_window: int = 64,
        num_layers: int = 24,
        num_kv_heads: int = 2,
        head_dim: int = 64,
    ) -> None:
        self.sink_tokens = sink_tokens
        self.recent_window = recent_window
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._pinned_prefix_hash: str | None = None
        self._pinned_prefix_tokens: int = 0

    def pin_prefix(self, system_prompt: str, tool_schemas_json: str = "") -> bool:
        digest = hashlib.sha256(f"{system_prompt}||{tool_schemas_json}".encode("utf-8")).hexdigest()
        hit = self._pinned_prefix_hash == digest
        self._pinned_prefix_hash = digest
        self._pinned_prefix_tokens = max(1, (len(system_prompt) + len(tool_schemas_json)) // 4)
        return hit

    @property
    def pinned_prefix_tokens(self) -> int:
        return self._pinned_prefix_tokens

    def estimate_memory_mb(self, seq_len: int) -> tuple[float, float, int, int, int]:
        sink_n = min(seq_len, self.sink_tokens)
        rem = max(0, seq_len - sink_n)
        recent_n = min(rem, self.recent_window)
        mid_n = max(0, rem - recent_n)

        bytes_per_fp16_token = 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2
        bytes_per_int4_token = int(bytes_per_fp16_token * 0.28)

        fp16_bytes = seq_len * bytes_per_fp16_token
        comp_bytes = (sink_n + recent_n) * bytes_per_fp16_token + mid_n * bytes_per_int4_token
        return (
            round(fp16_bytes / (1024.0 * 1024.0), 4),
            round(comp_bytes / (1024.0 * 1024.0), 4),
            sink_n,
            mid_n,
            recent_n,
        )

    def compress_kv(
        self,
        keys: Sequence[Sequence[float]],
        values: Sequence[Sequence[float]],
    ) -> tuple[list[list[float]], list[list[float]], KVCompressionStats]:
        seq_len = len(keys)
        fp16_mb, comp_mb, sink_n, mid_n, recent_n = self.estimate_memory_mb(seq_len)

        if mid_n <= 0:
            stats = KVCompressionStats(
                seq_len=seq_len,
                sink_tokens=sink_n,
                quantized_middle_tokens=0,
                recent_window_tokens=recent_n,
                fp16_mb=fp16_mb,
                compressed_mb=comp_mb,
                key_mse=0.0,
                value_mse=0.0,
            )
            return [list(r) for r in keys], [list(r) for r in values], stats

        mid_end = sink_n + mid_n
        k_sink = [list(r) for r in keys[:sink_n]]
        v_sink = [list(r) for r in values[:sink_n]]
        k_mid, k_mse = quantize_k_per_channel_int4(keys[sink_n:mid_end])
        v_mid, v_mse = quantize_v_per_token_int4(values[sink_n:mid_end])
        k_rec = [list(r) for r in keys[mid_end:]]
        v_rec = [list(r) for r in values[mid_end:]]

        stats = KVCompressionStats(
            seq_len=seq_len,
            sink_tokens=sink_n,
            quantized_middle_tokens=mid_n,
            recent_window_tokens=recent_n,
            fp16_mb=fp16_mb,
            compressed_mb=comp_mb,
            key_mse=k_mse,
            value_mse=v_mse,
        )
        return [*k_sink, *k_mid, *k_rec], [*v_sink, *v_mid, *v_rec], stats
