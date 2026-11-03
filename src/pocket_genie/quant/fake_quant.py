from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def fake_quantize_weight_per_channel(weight: Any, bits: int = 4) -> Any:
    """Per-output-channel symmetric integer fake quantization for linear weights."""
    if torch is None:
        raise RuntimeError("PyTorch is required for fake_quantize_weight_per_channel")
    if bits >= 16:
        return weight
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    orig_dtype = weight.dtype
    w_fp = weight.detach().float()
    max_abs = w_fp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = max_abs / float(qmax)
    w_q = torch.round(w_fp / scale).clamp(qmin, qmax) * scale
    return w_q.to(dtype=orig_dtype)


def fake_quantize_activation_asymmetric(
    x: Any,
    bits: int = 8,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> Any:
    """Per-tensor asymmetric UINT8/UINT16 activation fake quantization."""
    if torch is None or bits >= 16 and clip_min is None and clip_max is None:
        return x
    qmax = (1 << bits) - 1
    orig_dtype = x.dtype
    x_fp = x.float()
    xmin = float(x_fp.min().item()) if clip_min is None else float(clip_min)
    xmax = float(x_fp.max().item()) if clip_max is None else float(clip_max)
    xmin = min(xmin, 0.0)
    xmax = max(xmax, 0.0)
    span = max(xmax - xmin, 1e-6)
    scale = span / float(qmax)
    zero_point = round(-xmin / scale)
    zero_point = max(0, min(qmax, zero_point))
    q = torch.round(x_fp / scale + zero_point).clamp(0, qmax)
    deq = (q - zero_point) * scale
    return deq.to(dtype=orig_dtype)


def attach_per_layer_activation_hooks(
    model: Any,
    layer_act_bits: dict[str, int],
    default_bits: int = 8,
    clip_ranges: dict[str, tuple[float, float]] | None = None,
) -> list[Any]:
    """Attach forward pre-hooks with per-layer configurable activation bitwidths (e.g. 8 vs 16 bits)."""
    if torch is None or nn is None:
        return []
    handles: list[Any] = []
    clips = clip_ranges or {}

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        bits = int(layer_act_bits.get(name, default_bits))
        c_range = clips.get(name)

        def _make_hook(b: int, cr: tuple[float, float] | None) -> Any:
            def _hook(_mod: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    return inputs
                if b >= 16:
                    return inputs
                cmin, cmax = (cr[0], cr[1]) if cr is not None else (None, None)
                q_in = fake_quantize_activation_asymmetric(inputs[0], bits=b, clip_min=cmin, clip_max=cmax)
                return (q_in, *inputs[1:])

            return _hook

        handles.append(module.register_forward_pre_hook(_make_hook(bits, c_range)))

    return handles
