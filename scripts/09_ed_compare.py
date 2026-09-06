#!/usr/bin/env python3
"""Matched B vs C vs ED comparison with enough seeds to resolve the difference.

Why this script exists
----------------------
The first ED run suggested ED beat the fusion arm at 80% masking by +0.007.
Repeating the identical configuration flipped the sign to -0.006: MPS kernels
are non-deterministic, and the ED-vs-C gap is smaller than the run-to-run
spread at 3 seeds. Any claim about which arm is better therefore has to be made
with more seeds and with the arms *paired* on seed, or it is noise.

So this script retrains B (degraded optical), C (fusion) and ED on the SAME
seed list in the same process, and reports the per-seed paired difference plus a
paired bootstrap over test patches. It writes to its own file and never touches
results.csv or ed_results.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_device, load_config, results_dir
from src.dataset import SPLITS, build_subset
from src.encoder_decoder import predict_ed, train_decoder, train_ed_head
from src.evaluate import compute_metrics
from src.features import load_features
from src.train import FeatureBank, predict, train_head

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_an = import_module("04_analyse")
paired_bootstrap = _an.paired_bootstrap


def load_bank(cfg, split, levels) -> FeatureBank:
    return FeatureBank(
        optical={l: load_features(cfg, split, "optical", l) for l in levels},
        sar=load_features(cfg, split, "sar", None),
        targets=np.load(Path("data/features") / f"{split}_targets.npy"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--variant", default="ed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    subset = build_subset(cfg, verbose=False)
    classes = subset.classes
    levels = [float(v) for v in cfg.masking.levels]
    seeds = list(range(args.seeds))
    banks = {s: load_bank(cfg, s, levels) for s in SPLITS}
    thr = float(cfg.eval.threshold)

    print(f"device={device}  seeds={seeds}  variant={args.variant}")
    print("Regime: R2 / ed_r2 (degradation-aware) throughout.\n")

    rows, probs_by_seed = [], {}
    t0 = time.time()
    for seed in seeds:
        # Arms B and C, exactly as in the original experiment.
        for arm in ("optical", "fusion"):
            head, _ = train_head(cfg, arm, "degraded", banks["train"], banks["validation"],
                                 len(classes), seed, device)
            for lvl in levels:
                p = predict(head, arm, banks["test"], lvl, device, seed)
                rows.append({"arm": {"optical": "B", "fusion": "C"}[arm], "seed": seed,
                             "test_level": lvl,
                             "macro_f1": compute_metrics(banks["test"].targets, p, thr)["macro_f1"]})
                probs_by_seed[(arm, seed, lvl)] = p

        # Arm ED, same seed.
        dec, _ = train_decoder(cfg, args.variant, banks["train"], banks["validation"],
                               seed, device)
        head, _ = train_ed_head(cfg, args.variant, "ed_r2", dec, banks["train"],
                                banks["validation"], len(classes), seed, device)
        for lvl in levels:
            p = predict_ed(head, args.variant, dec, banks["test"], lvl, device)
            rows.append({"arm": "ED", "seed": seed, "test_level": lvl,
                         "macro_f1": compute_metrics(banks["test"].targets, p, thr)["macro_f1"]})
            probs_by_seed[("ed", seed, lvl)] = p
        print(f"  seed {seed} done  [{time.time()-t0:5.0f}s]")

    df = pd.DataFrame(rows)
    out = results_dir(cfg, "metrics")
    df.to_csv(out / "ed_matched_runs.csv", index=False)

    # ------------------------------------------------------------- summary
    recs = []
    for lvl in levels:
        piv = df[df.test_level == lvl].pivot_table(index="seed", columns="arm",
                                                   values="macro_f1")
        d_ec = piv["ED"] - piv["C"]        # paired on seed
        d_eb = piv["ED"] - piv["B"]
        # Paired bootstrap over test patches, seed 0 predictions.
        lo, hi = paired_bootstrap(banks["test"].targets, probs_by_seed[("fusion", 0, lvl)],
                                  probs_by_seed[("ed", 0, lvl)], thr)
        recs.append({
            "masking_pct": int(round(lvl * 100)),
            "B_mean": round(float(piv["B"].mean()), 4), "B_std": round(float(piv["B"].std()), 4),
            "C_mean": round(float(piv["C"].mean()), 4), "C_std": round(float(piv["C"].std()), 4),
            "ED_mean": round(float(piv["ED"].mean()), 4), "ED_std": round(float(piv["ED"].std()), 4),
            "ED_minus_B": round(float(d_eb.mean()), 4),
            "ED_minus_B_std": round(float(d_eb.std()), 4),
            "ED_minus_C": round(float(d_ec.mean()), 4),
            "ED_minus_C_std": round(float(d_ec.std()), 4),
            # How many of the seeds agree with the sign of the mean difference?
            "ED_beats_C_n_seeds": int((d_ec > 0).sum()),
            "n_seeds": len(seeds),
            "ED_vs_C_boot_lo": round(lo, 4), "ED_vs_C_boot_hi": round(hi, 4),
            "seed_ci_excludes_zero": bool(
                abs(d_ec.mean()) > 1.96 * d_ec.std() / np.sqrt(len(seeds))
            ),
        })
    t = pd.DataFrame(recs)
    t.to_csv(out / "ed_matched_comparison.csv", index=False)

    print(f"\n{'='*118}\nMATCHED COMPARISON, R2 regime, {len(seeds)} paired seeds "
          f"(macro F1)\n{'='*118}")
    print(f"{'Mask':>5} | {'B optical':>16} | {'C fusion':>16} | {'ED recon':>16} | "
          f"{'ED - C':>17} | {'seeds ED>C':>10} | {'sig?':>5}")
    print("-" * 118)
    for _, r in t.iterrows():
        sig = "yes" if r.seed_ci_excludes_zero else "no"
        print(f"{r.masking_pct:4d}% | {r.B_mean:.4f}+-{r.B_std:.4f} | "
              f"{r.C_mean:.4f}+-{r.C_std:.4f} | {r.ED_mean:.4f}+-{r.ED_std:.4f} | "
              f"{r.ED_minus_C:+.4f}+-{r.ED_minus_C_std:.4f} | "
              f"{r.ED_beats_C_n_seeds:>4}/{r.n_seeds:<5} | {sig:>5}")
    print("\n'sig?' = |mean seed-paired difference| > 1.96 * SEM over seeds.")
    print("The bootstrap columns in the CSV resample test patches on seed 0 and")
    print("answer a different question (test-set sampling variability).")
    print(f"\nWrote {out/'ed_matched_comparison.csv'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
