#!/usr/bin/env python3
"""Leakage audit for the ED arm, on real trained models rather than toy tensors.

The experiment is only meaningful if the clean Sentinel-2 feature is used
*solely* as a regression target during decoder training, and never reaches the
classifier at inference. Reasoning about the code is not enough, so this
verifies it empirically:

  1. Destroy the clean test features (overwrite with noise) and confirm ED test
     predictions are bit-identical. If any clean feature were being read at
     inference, the predictions would change.
  2. Confirm the decoder output is invariant to the optical input entirely.
  3. Confirm the decoder rejects an optical-width tensor.
  4. Confirm the decoder was fit without labels (no label tensor in its graph)
     by checking its input/output dims are purely feature-shaped.
  5. Confirm train / validation / test patch ids are disjoint, so the decoder's
     regression targets never came from a test patch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_device, load_config
from src.dataset import SPLITS, build_subset
from src.encoder_decoder import predict_ed, train_decoder, train_ed_head
from src.features import load_features
from src.train import FeatureBank

ok = lambda m: print(f"  [ok] {m}")


def load_bank(cfg, split, levels) -> FeatureBank:
    return FeatureBank(
        optical={l: load_features(cfg, split, "optical", l) for l in levels},
        sar=load_features(cfg, split, "sar", None),
        targets=np.load(Path("data/features") / f"{split}_targets.npy"),
    )


def main() -> None:
    cfg = load_config()
    device = get_device()
    levels = [float(v) for v in cfg.masking.levels]
    subset = build_subset(cfg, verbose=False)
    banks = {s: load_bank(cfg, s, levels) for s in SPLITS}
    n_classes = len(subset.classes)

    print("Training one ED model (seed 0) to audit...\n")
    dec, _ = train_decoder(cfg, "ed", banks["train"], banks["validation"], 0, device)
    head, _ = train_ed_head(cfg, "ed", "ed_r2", dec, banks["train"], banks["validation"],
                            n_classes, 0, device)

    print("1. clean test features are unused at inference")
    lvl = 0.8
    base = predict_ed(head, "ed", dec, banks["test"], lvl, device)
    # Overwrite the clean (level 0.0) test features with noise and re-predict.
    saved = banks["test"].optical[0.0].copy()
    rng = np.random.default_rng(0)
    banks["test"].optical[0.0] = rng.normal(
        size=saved.shape, scale=float(saved.std())
    ).astype(saved.dtype)
    after = predict_ed(head, "ed", dec, banks["test"], lvl, device)
    banks["test"].optical[0.0] = saved
    assert np.array_equal(base, after), (
        "ED predictions changed when the clean test feature was destroyed -- LEAKAGE"
    )
    ok(f"destroying z_clean on test leaves predictions bit-identical "
       f"(max |delta| = {np.abs(base - after).max():.2e})")

    # Sanity check in the other direction: the prediction *must* depend on the
    # degraded feature, or the test above would pass trivially.
    saved80 = banks["test"].optical[lvl].copy()
    banks["test"].optical[lvl] = rng.normal(size=saved80.shape,
                                            scale=float(saved80.std())).astype(saved80.dtype)
    perturbed = predict_ed(head, "ed", dec, banks["test"], lvl, device)
    banks["test"].optical[lvl] = saved80
    assert not np.array_equal(base, perturbed), "prediction ignores z_degraded -- test is vacuous"
    ok(f"but destroying z_degraded DOES change predictions "
       f"(mean |delta| = {np.abs(base - perturbed).mean():.4f}) -- the check is not vacuous")

    print("\n2. decoder output depends on SAR alone")
    from src.encoder_decoder import reconstruct
    a = reconstruct(dec, "ed", banks["test"].sar, banks["test"].optical[0.0], device)
    b = reconstruct(dec, "ed", banks["test"].sar, banks["test"].optical[1.0], device)
    assert np.array_equal(a, b)
    ok("z_hat identical whether clean or 100%-masked optical is passed alongside")

    print("\n3. decoder rejects optical-width input")
    try:
        dec(torch.from_numpy(banks["test"].optical[0.0]).to(device))
        raise AssertionError("decoder accepted a 384-d input")
    except ValueError:
        ok("ValueError raised on a 384-d tensor (clean feature cannot be fed in)")

    print("\n4. decoder is label-free")
    assert dec.in_dim == int(cfg.models.sar.feature_dim)
    assert dec.out_dim == int(cfg.models.optical.feature_dim)
    assert dec.out_dim != n_classes, "decoder output must not be class-shaped"
    ok(f"decoder maps {dec.in_dim} -> {dec.out_dim}; neither dim is the {n_classes}-class "
       f"label space")

    print("\n5. decoder targets never came from a test patch")
    ids = {s: set(np.load(Path("data/features") / f"{s}_patch_ids.npy",
                          allow_pickle=True).tolist())
           for s in SPLITS}
    for a_, b_ in [("train", "test"), ("train", "validation"), ("validation", "test")]:
        inter = ids[a_] & ids[b_]
        assert not inter, f"{a_} and {b_} share {len(inter)} patch ids"
    ok(f"train({len(ids['train'])}) / validation({len(ids['validation'])}) / "
       f"test({len(ids['test'])}) patch ids are pairwise disjoint")

    print("\nLeakage audit passed: z_clean is a training target only.")


if __name__ == "__main__":
    main()
