#!/usr/bin/env python3
"""Ablation: does the conclusion depend on how "missing" is encoded?

The main study fills masked pixels with the per-band mean (0 after
normalisation) -- the neutral "no observation" encoding. Here they are filled
with a bright reflectance instead, imitating an opaque cloud top. If the SAR
gain survives the change, it is a property of the missing information rather
than of our particular fill choice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_device, load_config, results_dir, set_seed
from src.dataset import SPLITS, build_subset
from src.features import cache_path, extract_optical, extract_sar, feature_dir
from src.models import build_optical_encoder, build_sar_encoder
from src.evaluate import compute_metrics
from src.train import FeatureBank, predict, train_head


def main() -> None:
    cfg = load_config("configs/config_bright_fill.yaml")
    set_seed(cfg.seed)
    device = get_device()
    subset = build_subset(cfg, verbose=False)
    levels = [float(v) for v in cfg.masking.levels]
    fd = feature_dir(cfg)
    print(f"fill={cfg.masking.fill} (value {cfg.masking.bright_value}) -> {fd}")

    vit = build_optical_encoder(cfg, verbose=False).to(device)
    rn = build_sar_encoder(cfg, verbose=False).to(device)
    for split in SPLITS:
        frame = subset.split(split)
        np.save(fd / f"{split}_targets.npy", np.stack(frame.y.to_numpy()))
        p = cache_path(cfg, split, "sar", None)
        if not p.exists():
            np.save(p, extract_sar(cfg, frame, rn, device, f"{split} sar"))
        for level in levels:
            p = cache_path(cfg, split, "optical", level)
            if not p.exists():
                np.save(p, extract_optical(cfg, frame, level, vit, device,
                                           f"{split} L{level:.0%}"))
        print(f"  {split} features ready")

    banks = {
        s: FeatureBank(
            optical={l: np.load(cache_path(cfg, s, "optical", l)) for l in levels},
            sar=np.load(cache_path(cfg, s, "sar", None)),
            targets=np.load(fd / f"{s}_targets.npy"),
        )
        for s in SPLITS
    }

    rows = []
    for arm in ("optical", "fusion"):
        for seed in [int(s) for s in cfg.train.seeds]:
            head, _ = train_head(cfg, arm, "degraded", banks["train"], banks["validation"],
                                 len(subset.classes), seed, device)
            for level in levels:
                probs = predict(head, arm, banks["test"], level, device, seed)
                m = compute_metrics(banks["test"].targets, probs, float(cfg.eval.threshold))
                rows.append({"arm": arm, "seed": seed, "test_level": level, **m})
        print(f"  trained {arm}")

    df = pd.DataFrame(rows)
    out = results_dir(cfg, "metrics")
    df.to_csv(out / "ablation_bright_fill_results.csv", index=False)

    piv = df.pivot_table(index="test_level", columns="arm", values="macro_f1", aggfunc="mean")
    piv["sar_gain"] = piv["fusion"] - piv["optical"]

    main_tab = pd.read_csv(out / "sar_gain_table.csv")
    main_tab = main_tab[main_tab.regime == "degraded"].set_index("masking_pct")
    piv["sar_gain_mean_fill"] = [main_tab.loc[int(round(l * 100)), "sar_gain"] for l in piv.index]

    print("\nAblation: bright cloud fill vs mean fill (degradation-aware regime, macro F1)")
    print(piv.to_string(float_format=lambda v: f"{v:+.4f}"))
    piv.to_csv(out / "ablation_bright_fill_summary.csv")
    print(f"\nWrote {out/'ablation_bright_fill_summary.csv'}")


if __name__ == "__main__":
    main()
