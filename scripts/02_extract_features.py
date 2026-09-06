#!/usr/bin/env python3
"""Run the frozen encoders once per (split, modality, masking level) and cache.

Optical: 3 splits x 6 masking levels. SAR: 3 splits, unmasked.
Outputs land in data/features/ as .npy, aligned row-for-row with the split
frames produced by src.dataset.build_subset (which sorts by patch_id).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_device, load_config, results_dir, set_seed
from src.dataset import SPLITS, build_subset
from src.features import cache_path, extract_optical, extract_sar, feature_dir
from src.models import build_optical_encoder, build_sar_encoder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = get_device()
    subset = build_subset(cfg, verbose=False)
    levels = [float(v) for v in cfg.masking.levels]

    print(f"device={device}  patches={len(subset.frame)}  levels={levels}")
    print(f"cache -> {feature_dir(cfg)}\n")

    print("Loading encoders:")
    vit = build_optical_encoder(cfg).to(device)
    rn = build_sar_encoder(cfg).to(device)
    print()

    manifest = {}
    t0 = time.time()
    for split in SPLITS:
        frame = subset.split(split)
        # Save the row order so nothing can silently misalign later.
        np.save(feature_dir(cfg) / f"{split}_patch_ids.npy", frame.patch_id.to_numpy())
        np.save(feature_dir(cfg) / f"{split}_targets.npy", np.stack(frame.y.to_numpy()))

        p = cache_path(cfg, split, "sar", None)
        if args.force or not p.exists():
            f = extract_sar(cfg, frame, rn, device, f"{split} sar")
            np.save(p, f)
            print(f"  {split:11s} sar        {f.shape}  [{time.time()-t0:6.0f}s]")
        else:
            print(f"  {split:11s} sar        cached")
        manifest[f"{split}/sar"] = str(p.name)

        for level in levels:
            p = cache_path(cfg, split, "optical", level)
            if args.force or not p.exists():
                f = extract_optical(cfg, frame, level, vit, device, f"{split} L{level:.0%}")
                if not np.isfinite(f).all():
                    raise RuntimeError(f"non-finite features for {split} level {level}")
                np.save(p, f)
                print(f"  {split:11s} opt L{level:.0%}    {f.shape}  [{time.time()-t0:6.0f}s]")
            else:
                print(f"  {split:11s} opt L{level:.0%}    cached")
            manifest[f"{split}/optical/{level}"] = str(p.name)

    with open(results_dir(cfg, "metrics") / "feature_manifest.json", "w") as fh:
        json.dump({"files": manifest, "levels": levels,
                   "classes": subset.classes,
                   "elapsed_s": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
