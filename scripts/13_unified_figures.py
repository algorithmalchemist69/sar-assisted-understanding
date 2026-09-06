#!/usr/bin/env python3
"""Figures for the unified comparison: original arms and encoder-decoder together.

Writes results/figures/07-10. Figures 01-05 are the original B/C study and are
left in place unchanged; these are the all-arm versions.
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

ED_COLOR = "#8256d0"
EDRES_COLOR = "#c77fd6"
STYLE = {
    "optical":     (PALETTE["optical"], "o", "-",  "B: degraded optical"),
    "fusion":      (PALETTE["fusion"], "s", "-",   "C: + raw SAR (fusion)"),
    "ed":          (ED_COLOR, "P", "-",            "ED: + SAR-reconstructed optical"),
    "ed_residual": (EDRES_COLOR, "v", "-.",        "ED-res: reconstructed residual"),
    "sar":         (PALETTE["sar"], "^", "--",     "D: SAR only"),
    "fusion_shuf": (PALETTE["fusion_shuf"], "D", ":", "control: shuffled SAR"),
}
REGIME_TITLE = {
    "clean": "R1 / ED-R1: trained on clean optical\n(the assignment's minimum baseline)",
    "degraded": "R2 / ED-R2: trained with masking augmentation\n(the like-for-like test)",
}


def fig_unified_curves(df, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    for ax, regime in zip(axes, ["clean", "degraded"]):
        sub = df[df.regime == regime]
        for arm in ["optical", "fusion", "ed", "ed_residual", "sar", "fusion_shuf"]:
            d = sub[sub.arm == arm]
            if d.empty:
                continue
            color, mk, ls, label = STYLE[arm]
            g = d.groupby("test_level").macro_f1
            x = np.array(sorted(g.mean().index)) * 100
            m, s = g.mean().to_numpy(), g.std().to_numpy()
            faint = arm in ("fusion_shuf", "sar", "ed_residual")
            ax.plot(x, m, color=color, marker=mk, markersize=5.5, linewidth=2 if not faint else 1.6,
                    linestyle=ls, label=label, alpha=0.6 if faint else 1.0, zorder=3)
            if not faint:
                ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel("% of optical image masked")
        ax.set_title(REGIME_TITLE[regime], loc="left", fontsize=10.5)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_ylabel("macro F1")
    axes[1].annotate("fusion\nbest here", xy=(100, 0.464), xytext=(86, 0.30), fontsize=8.5,
                     color=PALETTE["fusion"], fontweight="bold", ha="center",
                     arrowprops=dict(arrowstyle="->", color=PALETTE["fusion"], lw=1.1))

    # In R2 every arm sits inside a 0.09 band, so the differences that matter are
    # invisible at the shared scale. Inset zooms the operational range.
    ins = axes[1].inset_axes([0.09, 0.10, 0.50, 0.40])
    sub = df[df.regime == "degraded"]
    for arm in ["optical", "fusion", "ed", "ed_residual"]:
        d = sub[sub.arm == arm]
        color, mk, ls, _ = STYLE[arm]
        g = d.groupby("test_level").macro_f1
        xx = np.array(sorted(g.mean().index)) * 100
        mm, ss = g.mean().to_numpy(), g.std().to_numpy()
        keep = xx <= 80
        ins.plot(xx[keep], mm[keep], color=color, marker=mk, markersize=4.2,
                 linewidth=1.6, linestyle=ls)
        ins.fill_between(xx[keep], (mm - ss)[keep], (mm + ss)[keep], color=color,
                         alpha=0.13, linewidth=0)
    ins.set_xlim(-4, 84)
    ins.set_ylim(0.585, 0.712)
    ins.set_xticks([0, 20, 40, 60, 80])
    ins.tick_params(labelsize=7)
    ins.set_title("zoom: the operational range", fontsize=8, loc="left", pad=3)
    for sp in ins.spines.values():
        sp.set_edgecolor(INK_SOFT); sp.set_linewidth(0.8)
    ins.set_facecolor("#fbfbfa")
    # One legend for both panels; the R2 axis is the only one carrying every arm.
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=8.4, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("All arms, both regimes: reconstruction vs concatenation",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "07_unified_macro_f1.png")
    print("  07_unified_macro_f1.png")


def fig_unified_gains(table, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    series = [("C_minus_B", PALETTE["fusion"], "C - B  (raw SAR fusion)"),
              ("ED_minus_B", ED_COLOR, "ED - B  (SAR reconstruction)"),
              ("ED_minus_C", "#3f3f3c", "ED - C  (which SAR use is better)")]
    w = 0.26
    for ax, regime in zip(axes, ["clean", "degraded"]):
        t = table[table.regime == regime]
        for i, (col, color, label) in enumerate(series):
            off = (i - 1) * w
            vals = t[col].to_numpy(dtype=float)
            errs = t[f"{col}_seed_std"].to_numpy(dtype=float)
            ax.bar(t.masking_pct + off * 20, vals, width=w * 20, color=color,
                   label=label, zorder=3, edgecolor="white", linewidth=0.6)
            ax.errorbar(t.masking_pct + off * 20, vals, yerr=errs, fmt="none",
                        ecolor=INK, elinewidth=0.9, capsize=2.5, zorder=4)
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_xlabel("% of optical image masked")
        ax.set_title(REGIME_TITLE[regime], loc="left", fontsize=10.5)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_ylabel("macro F1 difference")
    axes[0].legend(loc="upper left", fontsize=8.5)

    # In R2 the 100% bars are ~10x everything else, so the operational range is
    # unreadable at the shared scale. Inset zooms 0-80%.
    ins = axes[1].inset_axes([0.06, 0.06, 0.62, 0.42])
    t = table[(table.regime == "degraded") & (table.masking_pct <= 80)]
    for i, (col, color, _) in enumerate(series):
        off = (i - 1) * w
        ins.bar(t.masking_pct + off * 20, t[col].to_numpy(dtype=float), width=w * 20,
                color=color, zorder=3, edgecolor="white", linewidth=0.5)
        ins.errorbar(t.masking_pct + off * 20, t[col].to_numpy(dtype=float),
                     yerr=t[f"{col}_seed_std"].to_numpy(dtype=float), fmt="none",
                     ecolor=INK, elinewidth=0.7, capsize=2, zorder=4)
    ins.axhline(0, color=INK, linewidth=0.9)
    ins.set_xticks([0, 20, 40, 60, 80])
    ins.tick_params(labelsize=7)
    ins.set_title("zoom: 0-80% masking", fontsize=8, loc="left", pad=3)
    for sp in ins.spines.values():
        sp.set_edgecolor(INK_SOFT); sp.set_linewidth(0.8)
    ins.set_facecolor("#fbfbfa")
    fig.suptitle("Where each way of using SAR pays off", fontsize=12.5,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "08_unified_gains.png")
    print("  08_unified_gains.png")


def fig_unified_per_class(pc, fig_dir):
    d = pc[pc.regime == "degraded"]
    piv = {}
    for arm in ("optical", "fusion", "ed"):
        piv[arm] = d[d.arm == arm].pivot_table(index="class", columns="masking_pct", values="f1")
    ed_b = (piv["ed"] - piv["optical"])
    ed_c = (piv["ed"] - piv["fusion"])
    order = ed_c[80].sort_values(ascending=False).index

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), sharey=True)
    for ax, mat, title in [
        (axes[0], ed_b.loc[order], "ED $-$ B\n(does reconstruction beat plain optical?)"),
        (axes[1], ed_c.loc[order], "ED $-$ C\n(does reconstruction beat raw fusion?)"),
    ]:
        # Scale the palette on the operational columns only. At 100% masking both
        # arms are out of distribution and degenerate, so letting that column set
        # vmax washes out every difference that actually matters.
        op = mat[[c for c in mat.columns if c <= 80]].to_numpy()
        v = float(np.nanmax(np.abs(op)))
        im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([f"{c}%" for c in mat.columns], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xlabel("% masked")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.to_numpy()[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(val) > v * 0.55 else INK)
        # Mark the clipped column so a saturated cell is not misread as extreme.
        ax.axvline(len(mat.columns) - 1.5, color=INK, linewidth=1.2, linestyle="--")
        ax.text(len(mat.columns) - 1, -0.85, "clipped\n(out of distribution)", ha="center",
                fontsize=6.8, color=INK_SOFT)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([n if len(n) <= 34 else n[:32] + "..." for n in order], fontsize=8)
    fig.suptitle("Per-class effect of reconstructing optical features from SAR (R2 regime)",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(fig_dir / "09_unified_per_class.png")
    print("  09_unified_per_class.png")


def fig_unified_error_matrices(cfg, fig_dir, level=0.8, regime="degraded"):
    m = results_dir(cfg, "metrics")
    z = np.load(m / "unified_error_matrices.npz", allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    short = [c if len(c) <= 22 else c[:20] + "..." for c in classes]
    mats = {a: z[f"{regime}/{a}/{level}"] for a in ("optical", "fusion", "ed")}

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4),
                             gridspec_kw={"width_ratios": [1, 1, 1]})
    for ax, (a, title) in zip(axes, [
        ("optical", "B: degraded optical"),
        ("fusion", "C: + raw SAR"),
        ("ed", "ED: + SAR-reconstructed optical"),
    ]):
        im = ax.imshow(mats[a], cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(short, rotation=90, fontsize=7)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels([])          # only the leftmost panel is labelled
        ax.grid(False)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = mats[a][i, j]
                if np.isfinite(v) and v > 0.01:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                            color="white" if v < 0.55 else "#111")
    axes[0].set_yticks(range(len(classes)))
    axes[0].set_yticklabels(short, fontsize=7)
    axes[0].set_ylabel("true class")
    fig.colorbar(im, ax=axes, shrink=0.75, pad=0.015,
                 label="P(predict column | true row)")
    fig.suptitle(f"Error structure at {int(level*100)}% masking — the diagonal is recall",
                 fontsize=12.5, fontweight="bold", y=1.0)
    fig.savefig(fig_dir / "10_unified_error_matrices.png")
    print("  10_unified_error_matrices.png")


def main() -> None:
    cfg = load_config()
    use_style()
    m = results_dir(cfg, "metrics")
    fig_dir = results_dir(cfg, "figures")
    df = pd.read_csv(m / "unified_results.csv")
    table = pd.read_csv(m / "unified_gain_table.csv")
    pc = pd.read_csv(m / "unified_per_class.csv")

    fig_unified_curves(df, fig_dir)
    fig_unified_gains(table, fig_dir)
    fig_unified_per_class(pc, fig_dir)
    fig_unified_error_matrices(cfg, fig_dir)
    print(f"\nWrote figures 07-10 to {fig_dir}")


if __name__ == "__main__":
    main()
