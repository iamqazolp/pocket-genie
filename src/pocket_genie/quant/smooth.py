from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

ALPHA_GRID: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85)


def compute_smooth_scale(
    act_max: Any,
    weight_max: Any,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> Any:
    """Compute per-channel SmoothQuant migration scale s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)."""
    if torch is None:
        raise RuntimeError("PyTorch is required for compute_smooth_scale")
    a_clamped = act_max.clamp(min=eps).float()
    w_clamped = weight_max.clamp(min=eps).float()
    scale = (a_clamped.pow(alpha) / w_clamped.pow(1.0 - alpha)).clamp(min=eps)
    return scale


def search_optimal_alpha(
    act_max: Any,
    weight_tensor: Any,
    w_bits: int = 4,
    a_bits: int = 8,
    candidates: tuple[float, ...] = ALPHA_GRID,
) -> tuple[float, Any]:
    """Select alpha* minimizing proxy reconstruction MSE across the boundary."""
    if torch is None:
        raise RuntimeError("PyTorch is required for search_optimal_alpha")
    w_max = weight_tensor.detach().abs().amax(dim=0)
    best_alpha = candidates[0]
    best_loss = float("inf")
    best_scale = compute_smooth_scale(act_max, w_max, alpha=best_alpha)

    qmax_w = (1 << (w_bits - 1)) - 1
    qmax_a = (1 << a_bits) - 1

    for alpha in candidates:
        scale = compute_smooth_scale(act_max, w_max, alpha=alpha)
        smoothed_act = act_max.float() / scale
        smoothed_w = weight_tensor.detach().float() * scale.unsqueeze(0)

        a_step = smoothed_act.max().clamp(min=1e-6) / float(qmax_a)
        a_q = torch.round(smoothed_act / a_step).clamp(0, qmax_a) * a_step
        act_rel_err = ((smoothed_act - a_q).pow(2).mean() / (smoothed_act.pow(2).mean() + 1e-6)).item()

        w_step = smoothed_w.abs().amax(dim=1, keepdim=True).clamp(min=1e-6) / float(qmax_w)
        w_q = torch.round(smoothed_w / w_step).clamp(-qmax_w - 1, qmax_w) * w_step
        w_rel_err = ((smoothed_w - w_q).pow(2).mean() / (smoothed_w.pow(2).mean() + 1e-6)).item()

        loss = act_rel_err + w_rel_err
        if loss < best_loss:
            best_loss = loss
            best_alpha = alpha
            best_scale = scale

    return best_alpha, best_scale


def migrate_swiglu_down_proj_scale(
    up_proj: Any,
    down_proj: Any,
    act_max: Any,
    alpha: float | None = None,
) -> float:
    """Apply Boundary-C SmoothQuant migration between mlp.up_proj output rows and mlp.down_proj input columns."""
    if torch is None:
        raise RuntimeError("PyTorch is required for migrate_swiglu_down_proj_scale")
    with torch.no_grad():
        if alpha is None:
            chosen_alpha, scale = search_optimal_alpha(act_max, down_proj.weight.data)
        else:
            chosen_alpha = float(alpha)
            w_max = down_proj.weight.data.abs().amax(dim=0)
            scale = compute_smooth_scale(act_max, w_max, alpha=chosen_alpha)

        dtype = down_proj.weight.dtype
        s = scale.to(device=down_proj.weight.device, dtype=dtype)
        up_proj.weight.data.div_(s.unsqueeze(1))
        if getattr(up_proj, "bias", None) is not None:
            up_proj.bias.data.div_(s)
        down_proj.weight.data.mul_(s.unsqueeze(0))
        return chosen_alpha
