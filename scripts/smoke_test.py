#!/usr/bin/env python3
"""End-to-end smoke test on a handful of patches.

Exercises: LMDB load -> preprocessing -> masking -> ViT -> SAR encoder ->
fusion -> loss -> prediction, asserting shapes and finiteness at each step.
Run this before the full experiment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_device, load_config, set_seed
from src.dataset import PatchReader, build_subset
from src.features import extract_optical, extract_sar
from src.models import ClassifierHead, arm_input_dim, build_optical_encoder, build_sar_encoder
from src.masking import generate_mask

N = 16
ok = lambda msg: print(f"  [ok] {msg}")


def main() -> None:
    cfg = load_config()
    set_seed(cfg.seed)
    device = get_device()
    print(f"device: {device}\n")

    print("1. subset + LMDB read")
    subset = build_subset(cfg, verbose=False)
    frame = subset.split("test").head(N)
    reader = PatchReader(cfg)
    s2 = reader.read_s2(frame.patch_id.iloc[0])
    s1 = reader.read_s1(frame.s1_name.iloc[0])
    assert s2.shape == (12, 120, 120) and s1.shape == (2, 120, 120), (s2.shape, s1.shape)
    assert np.isfinite(s2).all() and np.isfinite(s1).all()
    ok(f"S2 {s2.shape} {s2.dtype} range [{s2.min():.0f}, {s2.max():.0f}]")
    ok(f"S1 {s1.shape} {s1.dtype} range [{s1.min():.1f}, {s1.max():.1f}] dB")
    ok(f"{len(subset.classes)} classes, target vector {frame.y.iloc[0].shape}")

    print("\n2. masking determinism and B/C identity")
    for lvl in cfg.masking.levels:
        m_b = generate_mask(frame.patch_id.iloc[0], lvl, sigma=cfg.masking.smoothing_sigma,
                            mask_seed=cfg.masking.mask_seed, mode=cfg.masking.mode)
        m_c = generate_mask(frame.patch_id.iloc[0], lvl, sigma=cfg.masking.smoothing_sigma,
                            mask_seed=cfg.masking.mask_seed, mode=cfg.masking.mode)
        assert np.array_equal(m_b, m_c), "arm B and arm C would see different masks"
        assert abs(m_b.mean() - lvl) < 0.005, (lvl, m_b.mean())
    ok("masks identical across arms and exact to <0.5% at every level")

    print("\n3. encoders")
    vit = build_optical_encoder(cfg).to(device)
    rn = build_sar_encoder(cfg).to(device)

    print("\n4. forward passes (frozen)")
    d_opt = int(cfg.models.optical.feature_dim)
    d_sar = int(cfg.models.sar.feature_dim)
    f_clean = extract_optical(cfg, frame, 0.0, vit, device, "clean")
    f_mask = extract_optical(cfg, frame, 0.8, vit, device, "80% masked")
    f_sar = extract_sar(cfg, frame, rn, device, "sar")
    assert f_clean.shape == (N, d_opt), f_clean.shape
    assert f_sar.shape == (N, d_sar), f_sar.shape
    assert np.isfinite(f_clean).all() and np.isfinite(f_sar).all()
    ok(f"optical features {f_clean.shape}, SAR features {f_sar.shape}, all finite")
    drift = np.linalg.norm(f_clean - f_mask, axis=1) / np.linalg.norm(f_clean, axis=1)
    ok(f"masking moves optical features: relative L2 shift {drift.mean():.3f} (must be >0)")
    assert drift.mean() > 0.01, "masking had no effect on features -- pipeline bug"

    print("\n5. fusion head, loss, prediction")
    y = torch.from_numpy(np.stack(frame.y.to_numpy()))
    for arm, x in [
        ("optical", torch.from_numpy(f_mask)),
        ("fusion", torch.from_numpy(np.hstack([f_mask, f_sar]))),
        ("sar", torch.from_numpy(f_sar)),
    ]:
        head = ClassifierHead(arm_input_dim(arm, cfg), len(subset.classes),
                              cfg.head.hidden_dim, cfg.head.dropout)
        assert x.shape[1] == arm_input_dim(arm, cfg), (arm, x.shape)
        logits = head(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        gnorm = sum(p.grad.pow(2).sum().item() for p in head.parameters() if p.grad is not None) ** 0.5
        probs = torch.sigmoid(logits)
        assert logits.shape == (N, len(subset.classes))
        assert torch.isfinite(loss) and gnorm > 0
        n_par = sum(p.numel() for p in head.parameters())
        ok(f"{arm:8s} in={x.shape[1]:5d} loss={loss.item():.4f} |grad|={gnorm:.3f} "
           f"params={n_par:,} probs in [{probs.min():.3f}, {probs.max():.3f}]")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
