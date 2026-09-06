#!/usr/bin/env python3
"""Select classes/patches, audit the data, and write the frozen split manifest.

This runs *before* any masking or modelling. It writes
``results/metrics/subset_manifest.csv`` (patch_id, s1_name, split, labels) which
every later stage reads, so the splits cannot drift between runs or between arms.
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
from src.dataset import PatchReader, build_subset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--audit-n", type=int, default=400, help="patches to audit in depth")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print("=" * 78)
    print("BUILDING SUBSET")
    print("=" * 78)
    subset = build_subset(cfg)
    reader = PatchReader(cfg)

    frame = subset.frame
    print("\nSplit x class positives:")
    counts = {}
    for split in ("train", "validation", "test"):
        y = np.stack(frame[frame.split == split].y.to_numpy())
        counts[split] = y.sum(0).astype(int)
    table = pd.DataFrame(counts, index=subset.classes)
    table["total"] = table.sum(1)
    table["prevalence"] = table["total"] / len(frame)
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nLabels per patch:")
    npl = np.stack(frame.y.to_numpy()).sum(1)
    print(f"  mean {npl.mean():.2f}  min {int(npl.min())}  max {int(npl.max())}")

    # ---- pairing + integrity audit ------------------------------------------
    print("\n" + "=" * 78)
    print("DATA AUDIT")
    print("=" * 78)
    print(f"  unique patch_id : {frame.patch_id.nunique()} / {len(frame)}")
    print(f"  unique s1_name  : {frame.s1_name.nunique()} / {len(frame)}")
    assert frame.patch_id.nunique() == len(frame), "duplicate S2 patch ids"
    assert frame.s1_name.nunique() == len(frame), "duplicate S1 names"

    # Tile-level check: does one geographic tile straddle several splits?
    frame = frame.copy()
    frame["tile"] = frame.patch_id.str.split("_").str[5]
    frame["row"] = frame.patch_id.str.split("_").str[6].astype(int)
    frame["col"] = frame.patch_id.str.split("_").str[7].astype(int)
    per_tile = frame.groupby("tile").split.nunique()
    print(f"  tiles: {len(per_tile)}, of which {(per_tile > 1).sum()} span >1 split")
    print("  (reBEN splits are contiguous spatial blocks *within* tiles -- see README)")

    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(len(frame), size=min(args.audit_n, len(frame)), replace=False)
    s2_sum = np.zeros(len(cfg.bands.s2))
    s2_sq = np.zeros(len(cfg.bands.s2))
    s1_sum = np.zeros(len(cfg.bands.s1))
    s1_sq = np.zeros(len(cfg.bands.s1))
    n_px = 0
    bad = []
    for i in idx:
        row = frame.iloc[i]
        s2 = reader.read_s2(row.patch_id)
        s1 = reader.read_s1(row.s1_name)
        if s2.shape != (len(cfg.bands.s2), 120, 120) or s1.shape != (len(cfg.bands.s1), 120, 120):
            bad.append((row.patch_id, "shape", s2.shape, s1.shape))
        if not (np.isfinite(s2).all() and np.isfinite(s1).all()):
            bad.append((row.patch_id, "nonfinite", 0, 0))
        k = s2.shape[1] * s2.shape[2]
        s2_sum += s2.reshape(len(s2), -1).sum(1)
        s2_sq += (s2.reshape(len(s2), -1) ** 2).sum(1)
        s1_sum += s1.reshape(len(s1), -1).sum(1)
        s1_sq += (s1.reshape(len(s1), -1) ** 2).sum(1)
        n_px += k

    print(f"  audited {len(idx)} patch pairs: {len(bad)} problems")
    for b in bad[:10]:
        print("   ", b)

    s2_mean, s2_std = s2_sum / n_px, np.sqrt(s2_sq / n_px - (s2_sum / n_px) ** 2)
    s1_mean, s1_std = s1_sum / n_px, np.sqrt(s1_sq / n_px - (s1_sum / n_px) ** 2)
    print("\n  Sentinel-2 raw band statistics (reflectance x 10000):")
    for b, m, s in zip(cfg.bands.s2, s2_mean, s2_std):
        print(f"    {b:4s} mean {m:8.1f}  std {s:8.1f}   -> /10000 = {m/10000:.4f}")
    print("  Sentinel-1 band statistics (dB), vs SSL4EO pretraining stats:")
    for b, m, s, pm, ps in zip(cfg.bands.s1, s1_mean, s1_std, cfg.preprocess.s1_mean, cfg.preprocess.s1_std):
        print(f"    {b:4s} mean {m:7.2f} (SSL4EO {pm:7.2f})   std {s:5.2f} (SSL4EO {ps:5.2f})")

    out = results_dir(cfg, "metrics")
    manifest = frame[["patch_id", "s1_name", "split", "tile", "row", "col"]].copy()
    manifest["labels"] = frame.labels.apply(lambda v: "|".join(sorted(v)))
    manifest.to_csv(out / "subset_manifest.csv", index=False)
    summary = {
        "classes": subset.classes,
        "n_patches": int(len(frame)),
        "split_sizes": frame.split.value_counts().to_dict(),
        "class_positives": {c: int(table.loc[c, "total"]) for c in subset.classes},
        "s2_raw_mean": s2_mean.tolist(),
        "s2_raw_std": s2_std.tolist(),
        "s1_db_mean": s1_mean.tolist(),
        "s1_db_std": s1_std.tolist(),
        "n_audited": int(len(idx)),
        "n_audit_problems": len(bad),
    }
    with open(out / "subset_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {out/'subset_manifest.csv'} and {out/'subset_summary.json'}")


if __name__ == "__main__":
    main()
