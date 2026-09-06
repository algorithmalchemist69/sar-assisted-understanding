#!/usr/bin/env python3
"""Three-model comparison figures: B vs C vs ED, at every masking level.

Every figure here answers the same question a different way: given a
cloud-damaged optical image, is it better to use no radar (B), to concatenate
raw radar (C), or to let radar reconstruct the clean optical feature (ED)?

Design rules, applied consistently:
  * exactly three models per panel, always the same three colours;
  * no insets -- a panel that needs a zoom gets its own panel;
  * difference panels share one scale so they are directly comparable;
  * the 100% masking column is always shaded, because every masking-trained
    head is out of distribution there and it is not an operating point.

Writes results/figures/07-10. Figures 01-05 are the original two-arm study and
are left unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, results_dir
from src.visualization import INK, INK_SOFT, PALETTE, use_style

B_COLOR = PALETTE["optical"]     # blue
C_COLOR = PALETTE["fusion"]      # orange
E_COLOR = "#8256D0"              # purple
D_COLOR = PALETTE["sar"]         # teal, reference line only

MODELS = [
    ("optical", "B", B_COLOR, "o", "B: optical only (no radar)"),
    ("fusion", "C", C_COLOR, "s", "C: + raw radar (concatenation)"),
    ("ed", "ED", E_COLOR, "P", "ED: + reconstructed optical"),
]
REGIMES = [("clean", "R1  ·  trained on clean optical only"),
           ("degraded", "R2  ·  trained with masking augmentation")]


def _ood(ax, lo=90.0):
    """Shade the 100% column -- out of distribution for every trained head."""
    ax.axvspan(lo, 106, color=INK_SOFT, alpha=0.07, linewidth=0, zorder=0)


# ----------------------------------------------------------------- figure 07
def fig_curves(df, fig_dir):
    """Full range on top, operational range below. Separate panels, no insets."""
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8))

    for col, (regime, title) in enumerate(REGIMES):
        sub = df[df.regime == regime]
        for row, (lo, hi, tag) in enumerate([(0, 100, "full range"),
                                             (0, 80, "operational range, rescaled")]):
            ax = axes[row][col]
            for arm, _, color, mk, label in MODELS:
                g = sub[sub.arm == arm].groupby("test_level").macro_f1
                x = np.array(sorted(g.mean().index)) * 100
                m, s = g.mean().to_numpy(), g.std().to_numpy()
                k = (x >= lo) & (x <= hi)
                ax.plot(x[k], m[k], color=color, marker=mk, markersize=6.5,
                        linewidth=2.2, label=label, zorder=3)
                ax.fill_between(x[k], (m - s)[k], (m + s)[k], color=color,
                                alpha=0.15, linewidth=0)
            d = sub[sub.arm == "sar"].macro_f1.mean()
            ax.axhline(d, color=D_COLOR, linestyle="--", linewidth=1.4, alpha=0.8, zorder=1)
            ax.text(lo + 1.5, d + 0.008, "D: radar only", fontsize=7.5, color=D_COLOR)
            if hi == 100:
                _ood(ax)
                ax.text(97.5, ax.get_ylim()[0] + 0.04, "out of\ndistribution",
                        ha="center", fontsize=7, color=INK_SOFT)
            ax.set_xticks([t for t in [0, 20, 40, 60, 80, 100] if lo <= t <= hi])
            ax.set_xlim(lo - 5, hi + 6)
            ax.set_title(f"{title}\n{tag}" if row == 0 else tag, loc="left",
                         fontsize=10.5 if row == 0 else 9.5,
                         color=INK if row == 0 else INK_SOFT)
            if row == 1:
                ax.set_xlabel("% of optical image masked")
            if col == 0:
                ax.set_ylabel("macro F1")

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Three ways to classify a cloud-damaged image",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "07_three_model_curves.png")
    print("  07_three_model_curves.png")


# ----------------------------------------------------------------- figure 08
def fig_deltas(table, matched, fig_dir):
    """Differences as lines, not 18 bars: three comparisons, two regimes."""
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), sharey=True)
    series = [
        ("C_minus_B", "C $-$ B", "does raw radar help?", C_COLOR),
        ("ED_minus_B", "ED $-$ B", "does reconstruction help?", E_COLOR),
        ("ED_minus_C", "ED $-$ C", "which way of using radar wins?", "#3f3f3c"),
    ]
    for ax, (col, name, sub, color) in zip(axes, series):
        for regime, ls, mk, alpha in [("clean", ":", "o", 0.5),
                                      ("degraded", "-", "P", 1.0)]:
            t = table[table.regime == regime].sort_values("masking_pct")
            ax.plot(t.masking_pct, t[col], color=color, linestyle=ls, marker=mk,
                    markersize=6, linewidth=2.2, alpha=alpha, zorder=3,
                    label="R1 (cloud-naive)" if regime == "clean" else "R2 (cloud-aware)")
        ax.axhline(0, color=INK, linewidth=1.1, zorder=2)
        _ood(ax)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("% of optical image masked")
        ax.set_title(f"{name}\n{sub}", loc="left", fontsize=10.5)
    axes[0].set_ylabel("macro F1 difference")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].text(0.5, 0.06, "above the line = first model wins",
                 transform=axes[0].transAxes, fontsize=8, color=INK_SOFT,
                 ha="center", style="italic")

    r = matched[matched.masking_pct == 80]
    if len(r):
        r = r.iloc[0]
        axes[2].annotate(
            f"80% is a tie\n({r.ED_minus_C:+.4f}, {int(r.ED_beats_C_n_seeds)}/10 seeds\nin the 10-seed matched run)",
            xy=(80, float(r.ED_minus_C)), xytext=(30, 0.17), fontsize=8.2,
            color=INK_SOFT,
            arrowprops=dict(arrowstyle="->", color=INK_SOFT, lw=0.9,
                            connectionstyle="arc3,rad=0.22"))
    fig.suptitle("Where each way of using radar pays off",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "08_three_model_deltas.png")
    print("  08_three_model_deltas.png")


# ----------------------------------------------------------------- figure 09
def fig_per_class(pc, fig_dir, level=80):
    """Grouped bars, not two heatmaps on different colour scales.

    The earlier heatmap gave each panel its own vmax, so identical colours meant
    different magnitudes -- easy to misread. Bars remove the ambiguity, and the
    two difference series here share one axis.
    """
    d = pc[(pc.regime == "degraded") & (pc.masking_pct == level)]
    piv = d.pivot_table(index="class", columns="arm", values="f1")[
        ["optical", "fusion", "ed"]]
    piv = piv.loc[(piv["ed"] - piv["fusion"]).sort_values().index]
    names = [n if len(n) <= 32 else n[:30] + "..." for n in piv.index]
    y = np.arange(len(piv))
    h = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.4),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    for i, (arm, short, color, _, _) in enumerate(MODELS):
        ax.barh(y + (1 - i) * h, piv[arm].to_numpy(), height=h, color=color,
                label=short, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel(f"per-class F1 at {level}% masking")
    ax.set_title("(a) Absolute performance", loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1.02)

    ax = axes[1]
    dc = (piv["fusion"] - piv["optical"]).to_numpy()
    de = (piv["ed"] - piv["optical"]).to_numpy()
    ax.barh(y + 0.5 * h, dc, height=h, color=C_COLOR, label="C $-$ B", zorder=3)
    ax.barh(y - 0.5 * h, de, height=h, color=E_COLOR, label="ED $-$ B", zorder=3)
    ax.axvline(0, color=INK, linewidth=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xlabel("gain over optical-only (B)")
    ax.set_title("(b) What each radar route adds\nboth series on one scale",
                 loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", visible=False)

    # Label the two classes that carry the story, set beside their own bars so
    # no leader line has to cross the other rows.
    idx = list(piv.index)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.16)
    for label, arr, txt in [
            ("Urban fabric", dc, "raw radar keeps double-bounce"),
            ("Inland waters", dc, "both routes recover water")]:
        if label in idx:
            k = idx.index(label)
            ax.text(arr[k] + 0.006, y[k] + (0.5 * h if label == "Urban fabric" else 0.5 * h),
                    "  " + txt, fontsize=7.2, color=INK_SOFT, va="center", style="italic")

    fig.suptitle(f"Per-class comparison at {level}% masking  (R2 regime)",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "09_three_model_per_class.png")
    print("  09_three_model_per_class.png")


# ----------------------------------------------------------------- figure 10
def fig_error_matrices(cfg, fig_dir, level=0.8, regime="degraded"):
    """B in absolute terms, then what each radar route changes -- shared scale."""
    m = results_dir(cfg, "metrics")
    z = np.load(m / "unified_error_matrices.npz", allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    short = [c if len(c) <= 20 else c[:18] + "..." for c in classes]
    mb = z[f"{regime}/optical/{level}"]
    dc = z[f"{regime}/fusion/{level}"] - mb
    de = z[f"{regime}/ed/{level}"] - mb
    v = float(np.nanmax(np.abs(np.concatenate([dc, de]))))

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 6.0))
    im0 = axes[0].imshow(mb, cmap="magma", vmin=0, vmax=1)
    axes[0].set_title("B: optical only\nP(predict column | true row)",
                      loc="left", fontsize=10.5)
    im1 = None
    for ax, mat, title in [(axes[1], dc, "C $-$ B\nwhat raw radar changes"),
                           (axes[2], de, "ED $-$ B\nwhat reconstruction changes")]:
        im1 = ax.imshow(mat, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(title, loc="left", fontsize=10.5)
    for k, ax in enumerate(axes):
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(short, rotation=90, fontsize=7)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(short if k == 0 else [], fontsize=7)
        ax.grid(False)
    axes[0].set_ylabel("true class")
    fig.colorbar(im0, ax=axes[0], shrink=0.7, pad=0.02)
    fig.colorbar(im1, ax=[axes[1], axes[2]], shrink=0.7, pad=0.02,
                 label="change in P(predict | true)")
    fig.suptitle(f"Error structure at {int(level*100)}% masking — "
                 f"the diagonal is per-class recall",
                 fontsize=13.5, fontweight="bold", y=1.01)
    fig.savefig(fig_dir / "10_three_model_error_matrices.png")
    print("  10_three_model_error_matrices.png")


def main() -> None:
    cfg = load_config()
    use_style()
    m = results_dir(cfg, "metrics")
    fig_dir = results_dir(cfg, "figures")

    df = pd.read_csv(m / "unified_results.csv")
    table = pd.read_csv(m / "unified_gain_table.csv")
    matched = pd.read_csv(m / "ed_matched_comparison.csv")
    pc = pd.read_csv(m / "unified_per_class.csv")

    fig_curves(df, fig_dir)
    fig_deltas(table, matched, fig_dir)
    fig_per_class(pc, fig_dir)
    fig_error_matrices(cfg, fig_dir)

    # Drop the superseded congested versions so the directory tells one story.
    for old in ["07_unified_macro_f1.png", "08_unified_gains.png",
                "09_unified_per_class.png", "10_unified_error_matrices.png"]:
        p = fig_dir / old
        if p.exists():
            p.unlink()
            print(f"  removed superseded {old}")
    print(f"\nWrote three-model figures 07-10 to {fig_dir}")


if __name__ == "__main__":
    main()
