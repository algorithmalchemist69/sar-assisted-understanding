#!/usr/bin/env python3
"""Headline tables: SAR gain per degradation level, with bootstrap CIs.

Two sources of uncertainty are reported:
  * across-seed spread (3 training seeds) -- variability of the fitted head;
  * a paired bootstrap over the 2151 test patches -- sampling variability of the
    test set, computed on matched predictions so the pairing is preserved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, results_dir
from src.evaluate import compute_metrics

N_BOOT = 1000


def paired_bootstrap(targets, probs_a, probs_b, threshold, seed=0, n_boot=N_BOOT):
    """Bootstrap the difference macroF1(b) - macroF1(a) over test patches."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        t = targets[idx]
        keep = t.sum(0) > 0            # classes with no positives are undefined
        a = compute_metrics(t[:, keep], probs_a[idx][:, keep], threshold)["macro_f1"]
        b = compute_metrics(t[:, keep], probs_b[idx][:, keep], threshold)["macro_f1"]
        diffs[i] = b - a
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = results_dir(cfg, "metrics")
    thr = float(cfg.eval.threshold)

    df = pd.read_csv(out / "results.csv")
    probs = np.load(out / "test_probs.npz", allow_pickle=True)
    targets = probs["targets"]

    levels = sorted(df.test_level.unique())
    records = []
    for regime in sorted(df.regime.unique()):
        for level in levels:
            sel = df[(df.regime == regime) & (df.test_level == level)]
            g = sel.groupby("arm").macro_f1
            mean, std = g.mean(), g.std()
            # Paired over seeds: same seed -> same init/order for both arms.
            per_seed = sel.pivot_table(index="seed", columns="arm", values="macro_f1")
            gain = per_seed["fusion"] - per_seed["optical"]
            # The bootstrap resamples test patches using the seed-0 predictions,
            # so its centre is the seed-0 gain, not the across-seed mean. Report
            # both, explicitly labelled, rather than letting the CI appear to
            # bracket a point estimate it was not computed from.
            seed0 = int(sel.seed.min())
            gain_seed0 = float(per_seed.loc[seed0, "fusion"] - per_seed.loc[seed0, "optical"])
            lo, hi = paired_bootstrap(
                targets,
                probs[f"{regime}/optical/{level}"],
                probs[f"{regime}/fusion/{level}"],
                thr,
            )
            records.append({
                "regime": regime,
                "masking_pct": int(round(level * 100)),
                "A_optical_clean_ref": round(float(df[(df.regime == regime) & (df.test_level == 0.0) & (df.arm == "optical")].macro_f1.mean()), 4),
                "B_degraded_optical": round(float(mean["optical"]), 4),
                "B_std": round(float(std["optical"]), 4),
                "C_degraded_plus_sar": round(float(mean["fusion"]), 4),
                "C_std": round(float(std["fusion"]), 4),
                "sar_gain": round(float(gain.mean()), 4),
                "sar_gain_seed_std": round(float(gain.std()), 4),
                "sar_gain_seed0": round(gain_seed0, 4),
                "sar_gain_boot_lo": round(lo, 4),
                "sar_gain_boot_hi": round(hi, 4),
                "significant": bool(lo > 0 or hi < 0),
                "control_fusion_shuffled_sar": round(float(mean["fusion_shuf"]), 4),
                "D_sar_only": round(float(mean["sar"]), 4),
            })

    table = pd.DataFrame(records)
    table.to_csv(out / "sar_gain_table.csv", index=False)

    for regime in table.regime.unique():
        t = table[table.regime == regime]
        print(f"\n{'='*100}\nREGIME: {regime}   (macro F1, mean over 3 seeds)\n{'='*100}")
        print(f"{'Mask':>5} | {'B: degraded opt':>17} | {'C: degraded+SAR':>17} | "
              f"{'gain (3 seeds)':>14} | {'gain (seed0)':>12} | {'95% CI on seed0':>20} | "
              f"{'shuf ctrl':>9} | {'SAR only':>8}")
        print("-" * 130)
        for _, r in t.iterrows():
            star = " *" if r.significant else "  "
            print(f"{r.masking_pct:4d}% | {r.B_degraded_optical:8.4f} +/-{r.B_std:.4f} | "
                  f"{r.C_degraded_plus_sar:8.4f} +/-{r.C_std:.4f} | "
                  f"{r.sar_gain:+9.4f}+/-{r.sar_gain_seed_std:.3f} | {r.sar_gain_seed0:+11.4f}{star}| "
                  f"[{r.sar_gain_boot_lo:+.4f}, {r.sar_gain_boot_hi:+.4f}] | "
                  f"{r.control_fusion_shuffled_sar:8.4f}  | {r.D_sar_only:7.4f}")
        print("  Across-seed spread and the bootstrap measure different things: the first is")
        print("  variability of the fitted head, the second sampling variability of the test set.")
        print("  * = bootstrap 95% CI on the seed-0 gain excludes zero")

    # Per-class SAR gain, at the level where it matters most in each regime.
    pc = pd.read_csv(out / "per_class.csv")
    rows = []
    for regime in sorted(pc.regime.unique()):
        for level in levels:
            b = pc[(pc.regime == regime) & (pc.arm == "optical") & (pc.test_level == level)]
            c = pc[(pc.regime == regime) & (pc.arm == "fusion") & (pc.test_level == level)]
            m = b.merge(c, on="class", suffixes=("_B", "_C"))
            for _, r in m.iterrows():
                rows.append({"regime": regime, "masking_pct": int(round(level * 100)),
                             "class": r["class"], "support": r.support_B,
                             "f1_B": r.f1_B, "f1_C": r.f1_C,
                             "sar_gain": round(r.f1_C - r.f1_B, 4)})
    pcg = pd.DataFrame(rows)
    pcg.to_csv(out / "per_class_sar_gain.csv", index=False)

    print(f"\n\n{'='*100}\nPER-CLASS SAR GAIN (macro F1 delta, degradation-aware regime)\n{'='*100}")
    piv = pcg[pcg.regime == "degraded"].pivot_table(
        index="class", columns="masking_pct", values="sar_gain")
    sup = pcg[(pcg.regime == "degraded") & (pcg.masking_pct == 0)].set_index("class").support
    piv.insert(0, "support", sup)
    print(piv.sort_values(80, ascending=False).to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\nWrote {out/'sar_gain_table.csv'} and {out/'per_class_sar_gain.csv'}")


if __name__ == "__main__":
    main()
