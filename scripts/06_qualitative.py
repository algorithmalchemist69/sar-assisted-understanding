#!/usr/bin/env python3
"""Representative successes and failures at a chosen degradation level.

Four cases are selected automatically by per-sample F1, so nothing is
cherry-picked: the selection rule is fixed in advance and the two failure
categories are included by construction.

  1. optical already succeeds, SAR adds little
  2. optical fails, SAR repairs it
  3. both fail
  4. SAR actively hurts
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, results_dir
from src.dataset import PatchReader, build_subset
from src.visualization import INK_SOFT, label_text, s1_gray, s2_rgb, sample_f1, use_style

CASES = [
    ("1. Optical already succeeds,\nSAR adds little",
     lambda fb, fc: (fb > 0.75) & (np.abs(fc - fb) < 0.05)),
    ("2. Optical fails,\nSAR repairs it",
     lambda fb, fc: (fb < 0.45) & (fc - fb > 0.30)),
    ("3. Both fail",
     lambda fb, fc: (fb < 0.35) & (fc < 0.35)),
    ("4. SAR actively hurts",
     lambda fb, fc: (fb - fc > 0.30) & (fb > 0.5)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--level", type=float, default=0.8)
    ap.add_argument("--regime", default="degraded")
    args = ap.parse_args()

    cfg = load_config(args.config)
    use_style()
    thr = float(cfg.eval.threshold)
    m = results_dir(cfg, "metrics")

    z = np.load(m / "test_probs.npz", allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    targets = z["targets"]
    pb = z[f"{args.regime}/optical/{args.level}"]
    pc = z[f"{args.regime}/fusion/{args.level}"]
    yb, yc = (pb >= thr).astype(int), (pc >= thr).astype(int)
    fb, fc = sample_f1(targets, yb), sample_f1(targets, yc)

    ids = np.load(Path("data/features") / "test_patch_ids.npy", allow_pickle=True)
    subset = build_subset(cfg, verbose=False)
    frame = subset.split("test").set_index("patch_id")
    reader = PatchReader(cfg)
    band_names = list(cfg.bands.s2)

    rng = np.random.default_rng(cfg.seed)
    picks = []
    for title, rule in CASES:
        idx = np.where(rule(fb, fc))[0]
        if len(idx) == 0:
            print(f"  no example matched: {title!r}")
            continue
        # Deterministic pick among qualifying patches (seeded), not the extreme,
        # so the example is representative of the category rather than an outlier.
        picks.append((title, int(rng.choice(idx))))

    fig, axes = plt.subplots(len(picks), 4, figsize=(15.0, 3.5 * len(picks)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 2.35]})
    if len(picks) == 1:
        axes = axes[None, :]

    from src.masking import generate_mask
    from src.preprocessing import s2_fill_values

    fill = s2_fill_values(float(cfg.preprocess.s2_scale), len(band_names),
                          cfg.masking.fill, float(cfg.masking.bright_value))
    rows = []
    for r, (title, i) in enumerate(picks):
        pid = str(ids[i])
        s1name = frame.loc[pid, "s1_name"]
        s2 = reader.read_s2(pid)
        s1 = reader.read_s1(s1name)
        mask = generate_mask(pid, args.level, size=int(cfg.preprocess.patch_size),
                             sigma=float(cfg.masking.smoothing_sigma),
                             mask_seed=int(cfg.masking.mask_seed), mode=cfg.masking.mode)
        s2m = s2.copy()
        s2m[:, mask] = fill[:, None]

        axes[r, 0].imshow(s2_rgb(s2, band_names))
        axes[r, 0].set_ylabel(title, fontsize=10, fontweight="bold", labelpad=12)
        axes[r, 1].imshow(s2_rgb(s2m, band_names))
        axes[r, 2].imshow(s1_gray(s1, 0), cmap="gray")
        if r == 0:
            axes[r, 0].set_title("Sentinel-2 (true colour)", fontsize=10)
            axes[r, 1].set_title(f"Sentinel-2, {args.level:.0%} masked", fontsize=10)
            axes[r, 2].set_title("Sentinel-1 (VV, dB)", fontsize=10)
            axes[r, 3].set_title("ground truth vs predictions", fontsize=10)
        for k in range(3):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([]); axes[r, k].grid(False)

        ax = axes[r, 3]
        ax.axis("off")
        conf_b = ", ".join(f"{pb[i, j]:.2f}" for j in np.where(yb[i])[0][:4]) or "-"
        conf_c = ", ".join(f"{pc[i, j]:.2f}" for j in np.where(yc[i])[0][:4]) or "-"
        txt = (
            f"$\\bf{{Ground\\ truth}}$\n{label_text(targets[i], classes)}\n\n"
            f"$\\bf{{B:\\ degraded\\ optical}}$   (F1 = {fb[i]:.2f})\n"
            f"{label_text(yb[i], classes)}\n  confidences: {conf_b}\n\n"
            f"$\\bf{{C:\\ +\\ SAR}}$   (F1 = {fc[i]:.2f},  {fc[i]-fb[i]:+.2f})\n"
            f"{label_text(yc[i], classes)}\n  confidences: {conf_c}"
        )
        ax.text(0, 1, txt, va="top", ha="left", fontsize=8.2, family="DejaVu Sans",
                transform=ax.transAxes, linespacing=1.35)
        rows.append({"case": title.replace("\n", " "), "patch_id": pid,
                     "f1_B": round(float(fb[i]), 3), "f1_C": round(float(fc[i]), 3),
                     "delta": round(float(fc[i] - fb[i]), 3),
                     "truth": "|".join(np.array(classes)[targets[i] > 0]),
                     "pred_B": "|".join(np.array(classes)[yb[i] > 0]),
                     "pred_C": "|".join(np.array(classes)[yc[i] > 0])})

    fig.suptitle(
        f"Representative successes and failures at {args.level:.0%} simulated cloud cover "
        f"({args.regime}-trained models)",
        fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    out = results_dir(cfg, "figures") / f"05_qualitative_L{int(args.level*100):03d}.png"
    fig.savefig(out)
    plt.close(fig)

    pd.DataFrame(rows).to_csv(m / "qualitative_examples.csv", index=False)
    print(f"Wrote {out}")
    print(pd.DataFrame(rows).to_string(index=False, max_colwidth=44))

    # How common is each category across the whole test set?
    print("\nCategory frequency over the full test set:")
    for title, rule in CASES:
        n = int(rule(fb, fc).sum())
        print(f"  {n:5d} / {len(fb)}  ({n/len(fb):5.1%})  {title.replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
