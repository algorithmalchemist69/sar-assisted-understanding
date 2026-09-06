#!/usr/bin/env python3
"""Train every arm x regime x seed and evaluate on the test set at every level.

Writes machine-readable results to results/metrics/:
  results.csv          one row per (regime, arm, seed, test level) x metrics
  per_class.csv        per-class precision/recall/F1/AP/TP/FP/FN/TN
  error_matrices.npz   P(predict j | true i) per (regime, arm, level), seed 0
  test_probs.npz       test-set probabilities, seed 0, for the failure analysis
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
from src.evaluate import compute_metrics, error_matrix, per_class_metrics
from src.features import load_features
from src.models.fusion_model import ARMS
from src.train import FeatureBank, predict, train_head

REGIMES = ("clean", "degraded")


def load_bank(cfg, split: str, levels: list[float]) -> FeatureBank:
    return FeatureBank(
        optical={l: load_features(cfg, split, "optical", l) for l in levels},
        sar=load_features(cfg, split, "sar", None),
        targets=np.load(Path("data/features") / f"{split}_targets.npy"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--regimes", nargs="*", default=list(REGIMES))
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = get_device()
    subset = build_subset(cfg, verbose=False)
    classes = subset.classes
    levels = [float(v) for v in cfg.masking.levels]
    seeds = [int(s) for s in cfg.train.seeds]

    banks = {s: load_bank(cfg, s, levels) for s in SPLITS}
    print(f"device={device}  classes={len(classes)}  levels={levels}  seeds={seeds}")
    for s in SPLITS:
        print(f"  {s:11s} n={len(banks[s].targets)}")
    print()

    rows, per_class_rows, err_mats, test_probs, histories = [], [], {}, {}, {}
    ckpt_dir = results_dir(cfg, "checkpoints")
    t0 = time.time()

    for regime in args.regimes:
        for arm in args.arms:
            for seed in seeds:
                head, info = train_head(
                    cfg, arm, regime, banks["train"], banks["validation"],
                    len(classes), seed, device,
                )
                histories[f"{regime}/{arm}/{seed}"] = info
                if seed == seeds[0]:
                    torch.save(
                        {"state_dict": head.state_dict(), "arm": arm, "regime": regime,
                         "seed": seed, "classes": classes, "config": dict(cfg)},
                        ckpt_dir / f"{regime}_{arm}_seed{seed}.pt",
                    )
                for level in levels:
                    probs = predict(head, arm, banks["test"], level, device, seed)
                    m = compute_metrics(banks["test"].targets, probs, float(cfg.eval.threshold))
                    rows.append({"regime": regime, "arm": arm, "seed": seed,
                                 "test_level": level, "best_epoch": info["best_epoch"],
                                 "val_macro_f1": info["best_val_macro_f1"], **m})
                    if seed == seeds[0]:
                        key = f"{regime}/{arm}/{level}"
                        err_mats[key] = error_matrix(banks["test"].targets, probs,
                                                     float(cfg.eval.threshold))
                        test_probs[key] = probs.astype(np.float32)
                        for r in per_class_metrics(banks["test"].targets, probs, classes,
                                                   float(cfg.eval.threshold)):
                            per_class_rows.append({"regime": regime, "arm": arm,
                                                   "seed": seed, "test_level": level, **r})
                print(f"  {regime:9s} {arm:12s} seed={seed} "
                      f"best_epoch={info['best_epoch']:3d} "
                      f"val_macroF1={info['best_val_macro_f1']:.4f}  "
                      f"[{time.time()-t0:5.0f}s]")

    out = results_dir(cfg, "metrics")
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out / "per_class.csv", index=False)
    np.savez_compressed(out / "error_matrices.npz", classes=np.array(classes), **err_mats)
    np.savez_compressed(out / "test_probs.npz",
                        targets=banks["test"].targets, classes=np.array(classes), **test_probs)
    with open(out / "training_histories.json", "w") as fh:
        json.dump(histories, fh, indent=2)

    print(f"\nWrote {len(df)} result rows to {out/'results.csv'} in {time.time()-t0:.0f}s")

    # Headline table: macro F1, mean +/- std over seeds.
    piv = df.pivot_table(index=["regime", "test_level"], columns="arm",
                         values="macro_f1", aggfunc=["mean", "std"])
    print("\nMacro F1 (mean over seeds):")
    print(piv["mean"].to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
