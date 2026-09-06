"""Normalisation, resizing and NaN/range guards for both modalities.

The recipes come from the pretrained checkpoints, not from us:

* **Sentinel-2** -- SSL4EO-S12 uses ``x / 10000`` with no mean subtraction
  (torchgeo ``_zhu_xlab_transforms``). Reflectance stays roughly in [0, 1].
* **Sentinel-1** -- channel-wise dB standardisation with
  ``mean = [-12.59, -20.26]``, ``std = [5.26, 5.91]`` (torchgeo
  ``_ssl4eo_s12_transforms_s1``).

Deviation from the published recipe, deliberately: torchgeo resizes to 256 and
centre-crops to 224, which would discard the outer ~12% of every patch. Because
the whole study is about *how much of the patch is visible*, we resize
120 -> 224 directly and keep the entire footprint. This is documented in the
README.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def s2_fill_values(scale: float, n_bands: int, fill: str, bright_value: float) -> np.ndarray:
    """Raw-space value written into masked pixels.

    ``mean`` -> 0.0 raw, which is also 0.0 after ``/scale``: the neutral
    "no observation" encoding. ``bright`` -> an opaque bright cloud top.
    """
    if fill == "mean":
        return np.zeros(n_bands, dtype=np.float32)
    if fill == "bright":
        return np.full(n_bands, bright_value * scale, dtype=np.float32)
    raise ValueError(f"unknown fill strategy: {fill}")


def normalize_s2(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x / scale


def normalize_s1(x: torch.Tensor, mean: list[float], std: list[float], clip_db) -> torch.Tensor:
    m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    # GRD dB can carry extreme values in layover/shadow; clip before standardising.
    x = x.clamp(min=clip_db[0], max=clip_db[1])
    return (x - m) / s


def resize(x: torch.Tensor, size: int, mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    return F.interpolate(x, size=(size, size), mode=mode, **kwargs)


def resize_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest-neighbour so cloud edges stay hard after upsampling."""
    m = mask.to(torch.float32).unsqueeze(1)
    m = F.interpolate(m, size=(size, size), mode="nearest")
    return m.squeeze(1) > 0.5


def assert_finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        n_nan = int(torch.isnan(x).sum())
        n_inf = int(torch.isinf(x).sum())
        raise ValueError(f"{name} contains {n_nan} NaN and {n_inf} Inf values")
