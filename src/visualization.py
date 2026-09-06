"""Plot helpers: a shared style, patch rendering, and per-sample scoring."""
from __future__ import annotations

import matplotlib as mpl
import numpy as np

# Categorical slots, validated for colour-vision deficiency against a light
# surface (see the dataviz palette reference). Colour never carries identity on
# its own here: every series is also direct-labelled and given its own marker.
PALETTE = {
    "optical": "#2a78d6",      # slot 1, blue   -- conditions A / B
    "fusion": "#eb6834",       # slot 2, orange -- condition C
    "sar": "#1baf7a",          # slot 3, aqua   -- SAR-only reference
    "fusion_shuf": "#9a9a94",  # neutral grey   -- the shuffled-SAR control
}
MARKERS = {"optical": "o", "fusion": "s", "sar": "^", "fusion_shuf": "D"}
LABELS = {
    "optical": "B: degraded optical",
    "fusion": "C: degraded optical + SAR",
    "sar": "D: SAR only",
    "fusion_shuf": "control: shuffled SAR",
}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e3df"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.size": 10,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def s2_rgb(patch: np.ndarray, band_names: list[str],
           vmax: float = 3000.0, gamma: float = 0.7) -> np.ndarray:
    """True-colour render from B04/B03/B02.

    A *fixed* reflectance stretch, not a per-image percentile one. Percentile
    stretching blows sensor noise up to full range on radiometrically uniform
    patches -- open water in particular renders as RGB static -- and makes
    patches incomparable with one another. Clipping at a fixed reflectance and
    applying a mild gamma keeps water dark, keeps land legible, and means the
    same colour denotes the same reflectance in every panel.
    """
    idx = [band_names.index(b) for b in ("B04", "B03", "B02")]
    rgb = np.clip(patch[idx].astype(np.float32) / vmax, 0, 1) ** gamma
    return np.transpose(rgb, (1, 2, 0))


def s1_gray(patch: np.ndarray, channel: int = 0,
            vmin: float = -25.0, vmax: float = 0.0) -> np.ndarray:
    """VV/VH backscatter on a fixed dB scale, for the same reason as above."""
    band = patch[channel].astype(np.float32)
    return np.clip((band - vmin) / (vmax - vmin), 0, 1)


def sample_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-sample (example-based) F1 for multi-label predictions."""
    tp = (y_true * y_pred).sum(1)
    denom = y_true.sum(1) + y_pred.sum(1)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-9), 1.0)


def label_text(vec: np.ndarray, classes: list[str], max_n: int = 4) -> str:
    names = [classes[i] for i in np.where(vec > 0)[0]]
    if not names:
        return "(none)"
    short = [n if len(n) <= 26 else n[:24] + "..." for n in names]
    if len(short) > max_n:
        return "\n".join(short[:max_n]) + f"\n(+{len(short)-max_n} more)"
    return "\n".join(short)
