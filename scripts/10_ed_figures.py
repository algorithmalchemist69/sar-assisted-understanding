#!/usr/bin/env python3
"""Figures for the SAR -> optical reconstruction experiment (arm ED).

Writes results/figures/06_encoder_decoder.png. Reuses the project's palette so
these sit alongside figures 01-05 rather than looking like a different study.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, results_dir
from src.visualization import GRID, INK, INK_SOFT, PALETTE, use_style

ED_COLOR = "#8256d0"     # purple: the new arm
CTRL_COLOR = "#9a9a94"   # the project's control grey


def main() -> None:
    cfg = load_config()
    out = results_dir(cfg, "metrics")
    fig_dir = results_dir(cfg, "figures")
    use_style()

    matched = pd.read_csv(out / "ed_matched_comparison.csv")
    rec = pd.read_csv(out / "ed_reconstruction.csv")
    edpc = pd.read_csv(out / "ed_per_class.csv")
    pc = pd.read_csv(out / "per_class.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9))

    # ---- (a) classification: B vs C vs ED, 10 paired seeds -----------------
    ax = axes[0]
    x = matched.masking_pct.to_numpy()
    for col, sd, color, label, mk in [
        ("B_mean", "B_std", PALETTE["optical"], "B: degraded optical", "o"),
        ("C_mean", "C_std", PALETTE["fusion"], "C: + raw SAR (fusion)", "s"),
        ("ED_mean", "ED_std", ED_COLOR, "ED: + SAR-reconstructed optical", "P"),
    ]:
        m, s = matched[col].to_numpy(), matched[sd].to_numpy()
        ax.plot(x, m, color=color, marker=mk, markersize=6, linewidth=2, label=label, zorder=3)
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
    ax.axvspan(-2, 62, color=ED_COLOR, alpha=0.06, linewidth=0, zorder=0)
    ax.text(28, 0.27, "ED > C\n10/10 seeds", ha="center", fontsize=8.5,
            color=ED_COLOR, fontweight="bold")
    ax.text(90, 0.30, "ED collapses:\nhead never trained\nat 100%", ha="center",
            fontsize=8, color=INK_SOFT)
    ax.set_xlabel("% of optical image masked")
    ax.set_ylabel("macro F1")
    ax.set_title("(a) Classification, R2 regime\n10 paired seeds", loc="left")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_xticks(x)

    # ---- (b) reconstruction quality, mean-relative ------------------------
    ax = axes[1]
    g = rec[rec.variant == "ed"].groupby("test_level")
    gs = rec[rec.variant == "ed_shuf"].groupby("test_level")
    xl = np.array(sorted(g.r2_hat.mean().index)) * 100
    ax.plot(xl, g.r2_hat.mean().to_numpy(), color=ED_COLOR, marker="P", linewidth=2,
            label=r"$\hat{z}_{clean}$ from SAR", zorder=3)
    ax.plot(xl, g.r2_degraded.mean().to_numpy(), color=PALETTE["optical"], marker="o",
            linewidth=2, label=r"$z_{degraded}$ (what the ViT sees)", zorder=3)
    ax.plot(xl, gs.r2_hat.mean().to_numpy(), color=CTRL_COLOR, marker="D", linewidth=1.8,
            linestyle="--", label="control: decoder on shuffled SAR", zorder=2)
    ax.axhline(0, color=INK, linewidth=1)
    ax.text(101, 0.02, "  no better than\n  predicting the mean", va="bottom",
            fontsize=8, color=INK_SOFT)
    # Where the reconstruction overtakes the degraded feature.
    r2h = g.r2_hat.mean().to_numpy(); r2d = g.r2_degraded.mean().to_numpy()
    # r2h - r2d increases monotonically with masking, so interpolate directly;
    # reversing the arrays here silently pins the crossover to 0%.
    cross = float(np.interp(0.0, r2h - r2d, xl))
    ax.axvline(cross, color=INK_SOFT, linestyle=":", linewidth=1)
    ax.annotate(f"crossover ~{cross:.0f}%: above this, SAR\npredicts the clean feature better\n"
                f"than the masked image itself does",
                xy=(cross, -0.25), xytext=(4, -3.1), fontsize=8, color=INK_SOFT,
                arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=0.9,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_ylim(-4.0, 1.15)
    ax.set_xlabel("% of optical image masked")
    ax.set_ylabel(r"$R^2$ against the clean ViT feature")
    ax.set_title("(b) Does SAR predict the clean feature?\n"
                 r"$R^2=1-\mathrm{MSE}/\mathrm{MSE}_{mean}$", loc="left")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_xticks(xl)

    # ---- (c) per-class, ED - C at 80% -------------------------------------
    ax = axes[2]
    e = edpc[(edpc.variant == "ed") & (edpc.regime == "ed_r2")
             & (edpc.test_level == 0.8)][["class", "f1"]]
    c = pc[(pc.regime == "degraded") & (pc.arm == "fusion")
           & (pc.test_level == 0.8)][["class", "f1"]]
    m = e.merge(c, on="class", suffixes=("_ED", "_C"))
    m["d"] = m.f1_ED - m.f1_C
    m = m.sort_values("d")
    names = [n if len(n) <= 30 else n[:28] + "..." for n in m["class"]]
    colors = [ED_COLOR if v > 0 else PALETTE["fusion"] for v in m.d]
    ax.barh(range(len(m)), m.d.to_numpy(), color=colors, height=0.68)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("per-class F1:  ED $-$ C  (80% masking)")
    ax.set_title("(c) What the bottleneck costs\nradar-distinctive classes", loc="left")
    ax.text(0.02, 0.5, "ED better", transform=ax.transAxes, fontsize=8.5,
            color=ED_COLOR, fontweight="bold")
    ax.text(0.62, 0.06, "raw SAR better\n(structural signatures)", transform=ax.transAxes,
            fontsize=8, color=PALETTE["fusion"])
    ax.grid(axis="y", visible=False)

    fig.suptitle("Additional experiment: can SAR reconstruct the clean optical representation?",
                 fontsize=12.5, fontweight="bold", x=0.5, y=1.03)
    fig.tight_layout()
    fig.savefig(fig_dir / "06_encoder_decoder.png")
    print(f"Wrote {fig_dir/'06_encoder_decoder.png'}")


if __name__ == "__main__":
    main()
