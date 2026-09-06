#!/usr/bin/env python3
"""Unified comparison of every arm, original and encoder-decoder together.

Merges results.csv (arms B / C / D / shuffled control) with ed_results.csv
(arms ED and ED-residual) into one table per regime, so all five arms can be
read side by side at every masking level. Both source files are read only and
left unchanged.

The two experiments used identical splits, identical masks, identical seeds
(0/1/2) and identical head recipe, so the merge is a like-for-like join rather
than a rescaling of one onto the other.

Writes to results/metrics/:
  unified_results.csv        every arm x regime x seed x level, long form
  unified_gain_table.csv     per regime x level: all arms + gains vs B, with
                             paired bootstrap CIs for C-B, ED-B and ED-C
  unified_per_class.csv      per-class F1 for every arm, seed 0
  unified_error_matrices.npz P(predict j | true i) for B, C and ED
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

from src.config import load_config, results_dir
from src.evaluate import error_matrix

paired_bootstrap = import_module("04_analyse").paired_bootstrap

# ED regimes are the same two regimes under different names.
REGIME_OF_ED = {"ed_r1": "clean", "ed_r2": "degraded"}
# Display order and labels for the unified arm axis.
ARM_ORDER = ["optical", "fusion", "ed", "ed_residual", "fusion_shuf", "sar"]
ARM_LABEL = {
    "optical": "B: degraded optical",
    "fusion": "C: + raw SAR (fusion)",
    "ed": "ED: + SAR-reconstructed optical",
    "ed_residual": "ED-res: + reconstructed residual",
    "fusion_shuf": "control: shuffled SAR",
    "sar": "D: SAR only",
}


def unify(cfg) -> tuple[pd.DataFrame, dict]:
    m = results_dir(cfg, "metrics")
    base = pd.read_csv(m / "results.csv")
    ed = pd.read_csv(m / "ed_results.csv")

    ed = ed.rename(columns={"variant": "arm"})
    ed["regime"] = ed.regime.map(REGIME_OF_ED)
    keep = [c for c in base.columns if c in ed.columns]
    df = pd.concat([base[keep], ed[keep]], ignore_index=True)

    probs = {}
    z = np.load(m / "test_probs.npz", allow_pickle=True)
    for k in z.files:
        if k not in ("targets", "classes"):
            probs[k] = z[k]
    ze = np.load(m / "ed_test_probs.npz", allow_pickle=True)
    for k in ze.files:
        if k in ("targets", "classes"):
            continue
        variant, ed_regime, level = k.split("/")
        probs[f"{REGIME_OF_ED[ed_regime]}/{variant}/{level}"] = ze[k]
    probs["_targets"] = z["targets"]
    probs["_classes"] = [str(c) for c in z["classes"]]
    return df, probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = results_dir(cfg, "metrics")
    thr = float(cfg.eval.threshold)

    df, probs = unify(cfg)
    targets, classes = probs["_targets"], probs["_classes"]
    levels = sorted(df.test_level.unique())
    df.to_csv(out / "unified_results.csv", index=False)

    arms_present = [a for a in ARM_ORDER if a in set(df.arm)]
    print(f"arms: {arms_present}")
    print(f"regimes: {sorted(df.regime.unique())}   levels: {levels}")
    print(f"rows: {len(df)}\n")

    # ------------------------------------------------------------ gain table
    records = []
    for regime in sorted(df.regime.unique()):
        for level in levels:
            sel = df[(df.regime == regime) & (df.test_level == level)]
            g = sel.groupby("arm").macro_f1
            mean, std = g.mean(), g.std()
            per_seed = sel.pivot_table(index="seed", columns="arm", values="macro_f1")
            row = {"regime": regime, "masking_pct": int(round(level * 100))}
            for a in arms_present:
                if a in mean.index:
                    row[f"{a}_mean"] = round(float(mean[a]), 4)
                    row[f"{a}_std"] = round(float(std[a]), 4)

            def gain(a, b):
                """Seed-paired mean difference a - b, or None if either is absent."""
                if a not in per_seed.columns or b not in per_seed.columns:
                    return None, None
                d = per_seed[a] - per_seed[b]
                return round(float(d.mean()), 4), round(float(d.std()), 4)

            for name, a, b in [("C_minus_B", "fusion", "optical"),
                               ("ED_minus_B", "ed", "optical"),
                               ("ED_minus_C", "ed", "fusion"),
                               ("EDres_minus_C", "ed_residual", "fusion")]:
                mu, sd = gain(a, b)
                row[name], row[f"{name}_seed_std"] = mu, sd

            # Paired bootstrap over test patches, on seed-0 predictions.
            for name, a, b in [("C_minus_B", "fusion", "optical"),
                               ("ED_minus_B", "ed", "optical"),
                               ("ED_minus_C", "ed", "fusion")]:
                ka, kb = f"{regime}/{a}/{level}", f"{regime}/{b}/{level}"
                if ka in probs and kb in probs:
                    lo, hi = paired_bootstrap(targets, probs[kb], probs[ka], thr)
                    row[f"{name}_boot_lo"] = round(lo, 4)
                    row[f"{name}_boot_hi"] = round(hi, 4)
                    row[f"{name}_significant"] = bool(lo > 0 or hi < 0)
            records.append(row)

    table = pd.DataFrame(records)
    table.to_csv(out / "unified_gain_table.csv", index=False)

    # ------------------------------------------------------------- per class
    pc_rows = []
    from src.evaluate import per_class_metrics
    for regime in sorted(df.regime.unique()):
        for level in levels:
            for a in arms_present:
                k = f"{regime}/{a}/{level}"
                if k not in probs:
                    continue
                for r in per_class_metrics(targets, probs[k], classes, thr):
                    pc_rows.append({"regime": regime, "arm": a,
                                    "masking_pct": int(round(level * 100)), **r})
    pd.DataFrame(pc_rows).to_csv(out / "unified_per_class.csv", index=False)

    # -------------------------------------------------------- error matrices
    mats = {}
    for regime in sorted(df.regime.unique()):
        for level in levels:
            for a in ("optical", "fusion", "ed"):
                k = f"{regime}/{a}/{level}"
                if k in probs:
                    mats[k] = error_matrix(targets, probs[k], thr)
    np.savez_compressed(out / "unified_error_matrices.npz",
                        classes=np.array(classes), **mats)

    # ----------------------------------------------------------- print it out
    for regime in sorted(table.regime.unique()):
        t = table[table.regime == regime]
        title = ("R1 / ED-R1: trained on clean optical"
                 if regime == "clean" else
                 "R2 / ED-R2: trained with masking augmentation")
        print(f"\n{'='*118}\n{title}   (macro F1, mean +/- std over 3 seeds)\n{'='*118}")
        cols = [a for a in arms_present if f"{a}_mean" in t.columns
                and t[f"{a}_mean"].notna().any()]
        hdr = f"{'Mask':>5} |" + "".join(f" {a[:14]:>16} |" for a in cols)
        print(hdr)
        print("-" * len(hdr))
        for _, r in t.iterrows():
            line = f"{r.masking_pct:4d}% |"
            for a in cols:
                v, s = r.get(f"{a}_mean"), r.get(f"{a}_std")
                line += f" {v:8.4f}+-{s:.4f} |" if pd.notna(v) else f" {'--':>16} |"
            print(line)

        print(f"\n  {'Mask':>5} | {'C - B':>18} | {'ED - B':>18} | {'ED - C':>18}")
        print("  " + "-" * 70)
        for _, r in t.iterrows():
            def cell(n):
                mu = r.get(n)
                if pd.isna(mu):
                    return f"{'--':>18}"
                sig = "*" if r.get(f"{n}_significant") else " "
                return f"{mu:+9.4f}+-{r.get(f'{n}_seed_std', 0):.3f}{sig}"
            print(f"  {r.masking_pct:4d}% | {cell('C_minus_B')} | {cell('ED_minus_B')} | "
                  f"{cell('ED_minus_C')}")
        print("  * = paired bootstrap 95% CI on the seed-0 difference excludes zero")

    print(f"\nWrote unified_results.csv, unified_gain_table.csv, unified_per_class.csv,")
    print(f"      unified_error_matrices.npz to {out}")


if __name__ == "__main__":
    main()
