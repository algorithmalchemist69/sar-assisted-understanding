#!/usr/bin/env python3
"""Generate every figure in results/figures/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, results_dir
from src.visualization import INK_SOFT, LABELS, MARKERS, PALETTE, use_style

REGIME_TITLE = {
    "clean": "Regime R1: trained on clean optical\n(the assignment's minimum baseline)",
    "degraded": "Regime R2: both arms trained with masking\n(the like-for-like SAR test)",
}


def fig_main(df, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, regime in zip(axes, ["clean", "degraded"]):
        sub = df[df.regime == regime]
        for arm in ["optical", "fusion", "sar", "fusion_shuf"]:
            g = sub[sub.arm == arm].groupby("test_level").macro_f1
            x = np.array(sorted(g.mean().index)) * 100
            m, s = g.mean().to_numpy(), g.std().to_numpy()
            ax.plot(x, m, color=PALETTE[arm], marker=MARKERS[arm], markersize=6,
                    linewidth=2, label=LABELS[arm],
                    linestyle="--" if arm == "fusion_shuf" else "-",
                    alpha=0.55 if arm == "fusion_shuf" else 1.0, zorder=3)
            ax.fill_between(x, m - s, m + s, color=PALETTE[arm], alpha=0.15, linewidth=0)
        a_ref = sub[(sub.arm == "optical") & (sub.test_level == 0.0)].macro_f1.mean()
        ax.axhline(a_ref, color=INK_SOFT, linewidth=1, linestyle=":", zorder=1)
        ax.text(101, a_ref, "  A: optical only,\n  no degradation", va="center",
                fontsize=8, color=INK_SOFT)
        ax.set_title(REGIME_TITLE[regime])
        ax.set_xlabel("simulated cloud cover (% of patch masked)")
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlim(-4, 118)
    axes[0].set_ylabel("macro F1 (test)")
    axes[0].set_ylim(0, 0.80)
    axes[0].legend(loc="lower left")
    fig.suptitle("Does SAR recover what optical degradation destroys?", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.savefig(fig_dir / "01_macro_f1_vs_masking.png")
    plt.close(fig)


def fig_gain(table, fig_dir):
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    w = 7
    for i, (regime, off, color) in enumerate(
        [("clean", -w / 2, "#2a78d6"), ("degraded", w / 2, "#eb6834")]
    ):
        t = table[table.regime == regime].sort_values("masking_pct")
        err = np.abs(np.vstack([t.sar_gain_seed0 - t.sar_gain_boot_lo,
                                t.sar_gain_boot_hi - t.sar_gain_seed0]))
        ax.bar(t.masking_pct + off, t.sar_gain, width=w, color=color,
               label=f"R{i+1}: {'clean-trained' if regime=='clean' else 'degradation-aware'}",
               zorder=3)
        ax.errorbar(t.masking_pct + off, t.sar_gain_seed0, yerr=err, fmt="none",
                    ecolor="#0b0b0b", elinewidth=1, capsize=3, zorder=4)
        # Anchor each label outside the whisker, not the bar, so the two never collide.
        for _, r in t.iterrows():
            if r.sar_gain >= 0:
                y, va = max(r.sar_gain, r.sar_gain_boot_hi) + 0.008, "bottom"
            else:
                y, va = min(r.sar_gain, r.sar_gain_boot_lo) - 0.008, "top"
            ax.text(r.masking_pct + off, y, f"{r.sar_gain:+.3f}", ha="center",
                    va=va, fontsize=7.5, color="#0b0b0b")
    ax.axhline(0, color="#0b0b0b", linewidth=1)
    ax.set_ylim(-0.075, 0.52)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("simulated cloud cover (% of patch masked)")
    ax.set_ylabel("SAR gain:  macro F1(C) - macro F1(B)")
    ax.set_title("SAR gain by degradation level\nbars = mean over 3 seeds; whiskers = 95% paired bootstrap CI (seed 0)")
    ax.legend(loc="upper left")
    fig.savefig(fig_dir / "02_sar_gain.png")
    plt.close(fig)


def fig_per_class(pcg, fig_dir):
    piv = pcg[pcg.regime == "degraded"].pivot_table(
        index="class", columns="masking_pct", values="sar_gain")
    piv = piv.sort_values(80, ascending=True)
    # The 100% column is an order of magnitude larger than the rest (optical is
    # gone, so SAR supplies everything). Sharing one colour scale with it would
    # flatten the whole operational 0-80% range to white, so it gets its own
    # panel and its own scale.
    main, extreme = piv[[c for c in piv.columns if c < 100]], piv[[100]]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2), sharey=True,
                             gridspec_kw={"width_ratios": [5, 1.15], "wspace": 0.06})
    for ax, data, cmap, title in [
        (axes[0], main, "RdBu_r", "Operational range: 0-80% masked"),
        (axes[1], extreme, "RdBu_r", "100%\n(optical fully gone)"),
    ]:
        v = float(np.nanmax(np.abs(data.to_numpy())))
        im = ax.imshow(data.to_numpy(), cmap=cmap, vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(len(data.columns)), [f"{c}%" for c in data.columns])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.iloc[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > v * 0.55 else "#0b0b0b")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("simulated cloud cover")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
    axes[0].set_yticks(range(len(piv.index)),
                       [c if len(c) <= 46 else c[:44] + "..." for c in piv.index])
    fig.suptitle("Per-class SAR gain, F1(C) - F1(B), degradation-aware regime\n"
                 "red = SAR helps that class, blue = SAR hurts it",
                 fontsize=12, fontweight="bold", x=0.02, ha="left", y=1.06)
    fig.savefig(fig_dir / "03_per_class_sar_gain.png")
    plt.close(fig)


def fig_error_matrices(cfg, fig_dir, level=0.8, regime="degraded"):
    d = np.load(results_dir(cfg, "metrics") / "error_matrices.npz", allow_pickle=True)
    classes = [str(c) for c in d["classes"]]
    short = [c if len(c) <= 30 else c[:28] + "..." for c in classes]
    b, c = d[f"{regime}/optical/{level}"], d[f"{regime}/fusion/{level}"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6),
                             gridspec_kw={"wspace": 0.14})
    for ax, m, t in [(axes[0], b, f"B: degraded optical @ {level:.0%}"),
                     (axes[1], c, f"C: degraded optical + SAR @ {level:.0%}")]:
        im0 = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(t, fontsize=10)
    diff = c - b
    v = float(np.nanmax(np.abs(diff)))
    im1 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[2].set_title("C - B   (red = SAR raises this rate)", fontsize=10)

    for i, ax in enumerate(axes):
        ax.set_xticks(range(len(classes)), short, rotation=90, fontsize=7.5)
        ax.set_yticks(range(len(classes)), short if i == 0 else [""] * len(classes),
                      fontsize=7.5)
        ax.set_xlabel("predicted class", fontsize=9)
        ax.grid(False)
    axes[0].set_ylabel("true class (patch contains)", fontsize=9)
    # One shared bar for the two probability panels, one for the difference,
    # both placed outside the axes so they cannot clip neighbouring labels.
    fig.colorbar(im0, ax=axes[:2], shrink=0.8, pad=0.015, location="right",
                 label="P(predict j | true i)")
    fig.colorbar(im1, ax=axes[2], shrink=0.8, pad=0.03, location="right",
                 label="difference")
    fig.suptitle(
        "Error matrices at 80% masking:  M[i,j] = P(model predicts j | patch truly contains i); "
        "diagonal = per-class recall",
        fontsize=11, fontweight="bold", y=0.99)
    fig.savefig(fig_dir / "04_error_matrices.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    use_style()
    m = results_dir(cfg, "metrics")
    fig_dir = results_dir(cfg, "figures")

    df = pd.read_csv(m / "results.csv")
    table = pd.read_csv(m / "sar_gain_table.csv")
    pcg = pd.read_csv(m / "per_class_sar_gain.csv")

    fig_main(df, fig_dir)
    fig_gain(table, fig_dir)
    fig_per_class(pcg, fig_dir)
    fig_error_matrices(cfg, fig_dir)
    print("Wrote:")
    for p in sorted(fig_dir.glob("0*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
