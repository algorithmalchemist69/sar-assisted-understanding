"""Simulated cloud / missing-observation masking of Sentinel-2 patches.

Design decisions, all of which are reported in the README:

* **Spatially coherent, not per-pixel.** Clouds occlude contiguous regions. A
  per-pixel dropout of p% leaves almost every local neighbourhood partially
  visible, which is a far easier problem and would understate what SAR buys us.
  We threshold a low-pass-filtered white-noise field, which yields smooth blobs.
* **Exact coverage.** We mask the top ``round(level * H * W)`` pixels of the
  field by rank rather than thresholding at a quantile, so "40%" is 40.00%.
* **Deterministic per (patch_id, level).** The seed is derived from a hash of
  the patch id, the level and the global mask seed. Consequently arm B and arm
  C see *byte-identical* degraded imagery, which is what makes the comparison
  fair, and re-running reproduces the same masks.
* **Fill value.** ``mean`` writes the per-band training-set mean, i.e. exactly
  0.0 after normalisation: the neutral "no information" encoding of a missing
  observation. ``bright`` writes a high reflectance instead, imitating an opaque
  white cloud top. ``mean`` is the default.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _seed_for(patch_id: str, level: float, mask_seed: int) -> int:
    key = f"{patch_id}|{level:.4f}|{mask_seed}".encode()
    return int.from_bytes(hashlib.sha1(key).digest()[:8], "big") % (2**31 - 1)


def _gaussian_lowpass(noise: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur in the frequency domain (circular boundary, no scipy)."""
    h, w = noise.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    kernel = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fy**2 + fx**2))
    return np.fft.irfft2(np.fft.rfft2(noise) * kernel, s=(h, w))


def generate_mask(
    patch_id: str,
    level: float,
    size: int = 120,
    sigma: float = 8.0,
    mask_seed: int = 1234,
    mode: str = "coherent",
) -> np.ndarray:
    """Return a boolean (size, size) array; True marks an *obscured* pixel."""
    if level <= 0.0:
        return np.zeros((size, size), dtype=bool)
    if level >= 1.0:
        return np.ones((size, size), dtype=bool)

    rng = np.random.default_rng(_seed_for(patch_id, level, mask_seed))
    field = rng.standard_normal((size, size))
    if mode == "coherent":
        field = _gaussian_lowpass(field, sigma)
    elif mode != "pixel":
        raise ValueError(f"unknown masking mode: {mode}")

    n_masked = int(round(level * size * size))
    flat = field.ravel()
    # argpartition gives the n_masked largest field values -> one contiguous-ish blob set.
    idx = np.argpartition(flat, -n_masked)[-n_masked:]
    mask = np.zeros(size * size, dtype=bool)
    mask[idx] = True
    return mask.reshape(size, size)


def apply_mask(image: np.ndarray, mask: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """Fill masked pixels of a (C, H, W) image with per-band values.

    ``fill`` is a (C,) array. The input is not modified in place.
    """
    if image.ndim != 3:
        raise ValueError(f"expected (C, H, W), got {image.shape}")
    if mask.shape != image.shape[1:]:
        raise ValueError(f"mask {mask.shape} does not match image {image.shape[1:]}")
    if fill.shape[0] != image.shape[0]:
        raise ValueError(f"fill has {fill.shape[0]} values for {image.shape[0]} bands")
    out = image.copy()
    out[:, mask] = fill[:, None]
    return out


def coverage(mask: np.ndarray) -> float:
    return float(mask.mean())
