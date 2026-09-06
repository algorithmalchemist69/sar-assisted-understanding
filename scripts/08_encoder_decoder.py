#!/usr/bin/env python3
"""Additional experiment: SAR -> optical feature reconstruction (arm ED).

Trains a decoder z_sar -> z_hat_clean, freezes it, then trains a classifier on
[z_degraded ; z_hat_clean] and evaluates at every masking level. Compares
against the existing arms B (degraded optical) and C (degraded optical + raw SAR
fusion), which this script does not touch or re-run -- it reads their results
from results/metrics/results.csv.

Writes to results/metrics/:
  ed_results.csv         one row per (regime, variant, seed, test level) x metrics
  ed_reconstruction.csv  reconstruction MSE / cosine per (variant, seed, level)
  ed_comparison.csv      B vs C vs ED at every level, side by side
  ed_per_class.csv       per-class F1 for ED, seed 0
  ed_test_probs.npz      ED test probabilities, seed 0
and checkpoints to results/checkpoints/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_device, load_config, results_dir
from src.dataset import SPLITS, build_subset
from src.encoder_decoder import (
    ED_REGIMES,
    predict_ed,
    reconstruction_metrics,
    train_decoder,
    train_ed_head,
)
from src.evaluate import compute_metrics, per_class_metrics
from src.features import load_features
from src.train import FeatureBank


def load_bank(cfg, split: str, levels: list[float]) -> FeatureBank:
    return FeatureBank(
        optical={l: load_features(cfg, split, "optical", l) for l in levels},
        sar=load_features(cfg, split, "sar", None),
        targets=np.load(Path("data/features") / f"{split}_targets.npy"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--regimes", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ed = cfg.encoder_decoder
    if not bool(ed.enabled):
        print("encoder_decoder.enabled is false; nothing to do.")
        return

    device = get_device()
    subset = build_subset(cfg, verbose=False)
    classes = subset.classes
    levels = [float(v) for v in cfg.masking.levels]
    seeds = [int(s) for s in cfg.train.seeds]
    variants = args.variants or [str(v) for v in ed.variants]
    regimes = args.regimes or [str(r) for r in ed.regimes]

    banks = {s: load_bank(cfg, s, levels) for s in SPLITS}
    # Baseline for the mean-relative reconstruction metrics. Computed on TRAIN
    # only: using the test mean would leak test statistics into the baseline the
    # decoder is scored against.
    train_mean = banks["train"].optical[0.0].mean(0)
    print(f"device={device}  classes={len(classes)}  levels={levels}  seeds={seeds}")
    print(f"variants={variants}  regimes={regimes}")
    for s in SPLITS:
        print(f"  {s:11s} n={len(banks[s].targets)}")
    print()

    rows, recon_rows, pc_rows, probs_out, histories = [], [], [], {}, {}
    ckpt_dir = results_dir(cfg, "metrics").parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for variant in variants:
        # ed_residual's target is the level-dependent residual, which is
        # identically zero under R1; the variant is only defined for R2.
        var_regimes = [r for r in regimes if not (variant == "ed_residual" and r == "ed_r1")]
        if variant == "ed_residual" and "ed_r1" in regimes:
            print(f"  (skipping ed_residual x ed_r1: residual target is zero under R1)")

        for seed in seeds:
            dec, dinfo = train_decoder(cfg, variant, banks["train"], banks["validation"],
                                       seed, device)
            histories[f"{variant}/decoder/{seed}"] = dinfo
            print(f"  {variant:12s} decoder      seed={seed} "
                  f"best_epoch={dinfo['best_epoch']:3d} "
                  f"val_recon={dinfo['best_val_recon_loss']:.5f}  [{time.time()-t0:5.0f}s]")

            if seed == seeds[0]:
                torch.save({"state_dict": dec.state_dict(), "variant": variant,
                            "seed": seed, "config": dict(cfg)},
                           ckpt_dir / f"{variant}_decoder_seed{seed}.pt")

            # Reconstruction quality is a property of the decoder alone, so it
            # is measured once per (variant, seed), independently of the head.
            for level in levels:
                m = reconstruction_metrics(variant, dec, banks["test"], level, device,
                                           train_mean=train_mean)
                recon_rows.append({"variant": variant, "seed": seed,
                                   "test_level": level, **m})

            # ed_shuf exists to test the decoder, not to be a classifier arm.
            for regime in ([] if variant == "ed_shuf" else var_regimes):
                head, info = train_ed_head(cfg, variant, regime, dec, banks["train"],
                                           banks["validation"], len(classes), seed, device)
                histories[f"{variant}/{regime}/{seed}"] = info
                if seed == seeds[0]:
                    torch.save({"state_dict": head.state_dict(), "variant": variant,
                                "regime": regime, "seed": seed, "classes": classes,
                                "config": dict(cfg)},
                               ckpt_dir / f"{variant}_{regime}_head_seed{seed}.pt")
                for level in levels:
                    p = predict_ed(head, variant, dec, banks["test"], level, device)
                    mm = compute_metrics(banks["test"].targets, p, float(cfg.eval.threshold))
                    rows.append({"variant": variant, "regime": regime, "seed": seed,
                                 "test_level": level, "best_epoch": info["best_epoch"],
                                 "val_macro_f1": info["best_val_macro_f1"], **mm})
                    if seed == seeds[0]:
                        probs_out[f"{variant}/{regime}/{level}"] = p.astype(np.float32)
                        for r in per_class_metrics(banks["test"].targets, p, classes,
                                                   float(cfg.eval.threshold)):
                            pc_rows.append({"variant": variant, "regime": regime,
                                            "seed": seed, "test_level": level, **r})
                print(f"  {variant:12s} {regime:12s} seed={seed} "
                      f"best_epoch={info['best_epoch']:3d} "
                      f"val_macroF1={info['best_val_macro_f1']:.4f}  [{time.time()-t0:5.0f}s]")

    out = results_dir(cfg, "metrics")
    df = pd.DataFrame(rows)
    rec = pd.DataFrame(recon_rows)
    df.to_csv(out / "ed_results.csv", index=False)
    rec.to_csv(out / "ed_reconstruction.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(out / "ed_per_class.csv", index=False)
    np.savez_compressed(out / "ed_test_probs.npz",
                        targets=banks["test"].targets, classes=np.array(classes), **probs_out)
    with open(out / "ed_training_histories.json", "w") as fh:
        json.dump(histories, fh, indent=2)

    # ---------------------------------------------------------------- tables
    base = pd.read_csv(out / "results.csv")
    # ED-R1 corresponds to the existing "clean" regime, ED-R2 to "degraded".
    regime_map = {"ed_r1": "clean", "ed_r2": "degraded"}
    comp = []
    for regime in sorted(df.regime.unique()):
        b_reg = regime_map[regime]
        for level in levels:
            sel = base[(base.regime == b_reg) & (base.test_level == level)]
            g = sel.groupby("arm").macro_f1
            row = {"regime": regime, "baseline_regime": b_reg,
                   "masking_pct": int(round(level * 100)),
                   "B_degraded_optical": round(float(g.mean()["optical"]), 4),
                   "C_fusion": round(float(g.mean()["fusion"]), 4),
                   "D_sar_only": round(float(g.mean()["sar"]), 4)}
            for variant in sorted(df.variant.unique()):
                d = df[(df.regime == regime) & (df.variant == variant)
                       & (df.test_level == level)]
                if len(d) == 0:
                    continue
                row[f"{variant}_mean"] = round(float(d.macro_f1.mean()), 4)
                row[f"{variant}_std"] = round(float(d.macro_f1.std()), 4)
                row[f"{variant}_minus_B"] = round(float(d.macro_f1.mean()) - row["B_degraded_optical"], 4)
                row[f"{variant}_minus_C"] = round(float(d.macro_f1.mean()) - row["C_fusion"], 4)
            comp.append(row)
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(out / "ed_comparison.csv", index=False)

    for regime in comp_df.regime.unique():
        t = comp_df[comp_df.regime == regime]
        print(f"\n{'='*104}\nCLASSIFICATION  {regime}  (macro F1, mean over {len(seeds)} seeds)\n{'='*104}")
        hdr = f"{'Mask':>5} | {'B optical':>10} | {'C fusion':>10} | {'D SAR-only':>10}"
        for v in sorted(df.variant.unique()):
            if f"{v}_mean" in t.columns and t[f"{v}_mean"].notna().any():
                hdr += f" | {v:>12} | {'vs B':>7} | {'vs C':>7}"
        print(hdr)
        print("-" * len(hdr))
        for _, r in t.iterrows():
            line = (f"{r.masking_pct:4d}% | {r.B_degraded_optical:10.4f} | "
                    f"{r.C_fusion:10.4f} | {r.D_sar_only:10.4f}")
            for v in sorted(df.variant.unique()):
                if f"{v}_mean" in t.columns and pd.notna(r.get(f"{v}_mean")):
                    line += (f" | {r[f'{v}_mean']:7.4f}+-{r[f'{v}_std']:.3f}"
                             f" | {r[f'{v}_minus_B']:+7.4f} | {r[f'{v}_minus_C']:+7.4f}")
            print(line)

    print(f"\n{'='*104}\nRECONSTRUCTION QUALITY (test split, mean over seeds)\n{'='*104}")
    print("Two questions, and they have different answers:")
    print("  (a) is z_hat closer to z_clean than z_degraded is?      -> compare MSE hat vs MSE deg")
    print("  (b) has the decoder learned anything patch-specific?    -> R2 > 0 and centred cosine")
    print("R2 = 1 - MSE(z_hat,clean)/MSE(mean,clean): 0 means no better than always")
    print("predicting the training-set mean. Raw cosine is inflated because this feature")
    print("space is anisotropic (two unrelated patches already sit at cosine 0.958).")
    for variant in sorted(rec.variant.unique()):
        t = rec[rec.variant == variant].groupby("test_level").mean(numeric_only=True)
        print(f"\n  variant = {variant}")
        print(f"  {'Mask':>5} | {'MSE hat':>9} | {'MSE deg':>9} | {'MSE mean':>9} | "
              f"{'R2 hat':>8} | {'R2 deg':>8} | {'cos hat':>8} | {'ctr-cos hat':>11} | "
              f"{'ctr-cos deg':>11}")
        print("  " + "-" * 106)
        for lvl, r in t.iterrows():
            print(f"  {int(round(lvl*100)):4d}% | {r.mse_hat_vs_clean:9.5f} | "
                  f"{r.mse_degraded_vs_clean:9.5f} | {r.mse_mean_predictor:9.5f} | "
                  f"{r.r2_hat:+8.4f} | {r.r2_degraded:+8.4f} | {r.cos_hat_vs_clean:8.4f} | "
                  f"{r.centered_cos_hat:11.4f} | {r.centered_cos_degraded:11.4f}")

    print(f"\nWrote ed_results.csv, ed_reconstruction.csv, ed_comparison.csv, "
          f"ed_per_class.csv to {out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
