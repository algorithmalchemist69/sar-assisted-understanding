#!/usr/bin/env python3
"""Qualitative comparison of all three models: B, C and ED, on the same patches.

Same selection discipline as scripts/06_qualitative.py -- the rules are fixed
before any result is inspected, and the categories where reconstruction *loses*
are included by construction. What differs is that cases are now chosen by the
disagreement between the two ways of using SAR, which is the question this
figure exists to illustrate.

Writes results/figures/11_qualitative_unified_L080.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, results_dir
from src.dataset import PatchReader, build_subset
from src.masking import generate_mask
from src.visualization import INK, INK_SOFT, s1_gray, s2_rgb, sample_f1, use_style

# Selection rules, fixed in advance. Two of the four are cases where the new
# arm is the worse one.
CASES = [
    ("1. Reconstruction repairs what\nraw fusion could not",
     lambda fb, fc, fe: (fe - fc > 0.25) & (fe - fb > 0.25)),
    ("2. Raw fusion wins,\nreconstruction loses the signal",
     lambda fb, fc, fe: (fc - fe > 0.25) & (fc > 0.5)),
    ("3. Both SAR routes agree\nand both help",
     lambda fb, fc, fe: (fc - fb > 0.20) & (fe - fb > 0.20) & (np.abs(fe - fc) < 0.10)),
    ("4. Both SAR routes hurt",
     lambda fb, fc, fe: (fb - fc > 0.20) & (fb - fe > 0.20)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--level", type=float, default=0.8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    use_style()
    thr = float(cfg.eval.threshold)
    m = results_dir(cfg, "metrics")
    lvl = args.level

    z = np.load(m / "test_probs.npz", allow_pickle=True)
    ze = np.load(m / "ed_test_probs.npz", allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    targets = z["targets"]
    pb = z[f"degraded/optical/{lvl}"]
    pc = z[f"degraded/fusion/{lvl}"]
    pe = ze[f"ed/ed_r2/{lvl}"]
    fb = sample_f1(targets, (pb >= thr).astype(int))
    fc = sample_f1(targets, (pc >= thr).astype(int))
    fe = sample_f1(targets, (pe >= thr).astype(int))

    print(f"At {int(lvl*100)}% masking, over {len(targets)} test patches:")
    print(f"  ED strictly better than C : {int(((fe - fc) > 0.05).sum()):4d} "
          f"({(fe - fc > 0.05).mean()*100:.1f}%)")
    print(f"  C strictly better than ED : {int(((fc - fe) > 0.05).sum()):4d} "
          f"({(fc - fe > 0.05).mean()*100:.1f}%)")
    print(f"  tied within 0.05          : {int((np.abs(fe - fc) <= 0.05).sum()):4d} "
          f"({(np.abs(fe - fc) <= 0.05).mean()*100:.1f}%)\n")

    ids = np.load(Path("data/features") / "test_patch_ids.npy", allow_pickle=True)
    subset = build_subset(cfg, verbose=False)
    frame = subset.split("test").set_index("patch_id")
    reader = PatchReader(cfg)
    bands = list(cfg.bands.s2)
    rng = np.random.default_rng(cfg.seed)

    picks = []
    for title, rule in CASES:
        idx = np.where(rule(fb, fc, fe))[0]
        print(f"  {title.splitlines()[0]:52s} {len(idx):4d} candidates")
        picks.append((title, int(rng.choice(idx)) if len(idx) else None))

    ncols = 4
    fig, axes = plt.subplots(len(picks), ncols, figsize=(15.2, 3.5 * len(picks)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 2.5]})
    for r, (title, i) in enumerate(picks):
        row = axes[r]
        if i is None:
            for ax in row:
                ax.axis("off")
            row[0].text(0.5, 0.5, f"{title}\n(no qualifying patch)", ha="center",
                        va="center", fontsize=9, color=INK_SOFT)
            continue
        pid = str(ids[i])
        s2 = reader.read_s2(pid)
        s1 = reader.read_s1(str(frame.loc[pid].s1_name))
        mask = generate_mask(pid, lvl, size=int(cfg.preprocess.patch_size),
                             sigma=float(cfg.masking.smoothing_sigma),
                             mask_seed=int(cfg.masking.mask_seed), mode=cfg.masking.mode)
        rgb = s2_rgb(s2, bands)
        masked = rgb.copy()
        masked[mask] = 0.0

        for ax, img, ttl, kw in [
            (row[0], rgb, "Sentinel-2 (true colour)", {}),
            (row[1], masked, f"Sentinel-2, {int(lvl*100)}% masked", {}),
            (row[2], s1_gray(s1), "Sentinel-1 (VV, dB)", {"cmap": "gray"}),
        ]:
            ax.imshow(img, **kw)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if r == 0:
                ax.set_title(ttl, fontsize=9.5)
        row[0].set_ylabel(title, fontsize=9, fontweight="bold")

        ax = row[3]
        ax.axis("off")
        truth = [classes[k] for k in np.where(targets[i] == 1)[0]]
        lines = [("Ground truth", "bold"), *[(f"  {t}", None) for t in truth], ("", None)]
        for name, probs, f1 in [("B: degraded optical", pb, fb),
                                ("C: + raw SAR", pc, fc),
                                ("ED: + SAR-reconstructed optical", pe, fe)]:
            pred = np.where(probs[i] >= thr)[0]
            lines.append((f"{name}   (F1 = {f1[i]:.2f})", "bold"))
            if len(pred) == 0:
                lines.append(("  (nothing above threshold)", None))
            for k in pred[:5]:
                mark = "+" if targets[i, k] == 1 else "x"
                lines.append((f"  {mark} {classes[k][:44]}  {probs[i, k]:.2f}", None))
            if len(pred) > 5:
                lines.append((f"  (+{len(pred)-5} more)", None))
            lines.append(("", None))
        y = 1.0
        for text, weight in lines:
            ax.text(0, y, text, fontsize=8, va="top", transform=ax.transAxes,
                    fontweight=weight or "normal",
                    color=INK if weight else INK_SOFT, family="monospace")
            y -= 0.042
        if r == 0:
            ax.set_title("ground truth vs predictions", fontsize=9.5, loc="left")

    fig.suptitle(f"Two ways of using SAR, on the same patches at {int(lvl*100)}% masking\n"
                 f"(+ = correct, x = false positive)",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    out = results_dir(cfg, "figures") / f"11_qualitative_unified_L{int(lvl*100):03d}.png"
    fig.savefig(out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
