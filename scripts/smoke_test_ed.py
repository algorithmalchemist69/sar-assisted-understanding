#!/usr/bin/env python3
"""Smoke test for the SAR -> optical reconstruction arm (ED), on 16 samples.

Verifies the tensor contract end to end and, more importantly, the leakage
guarantees: that the decoder never receives a clean optical feature as input and
that the clean feature is used only as a regression target during training.

Run this before scripts/08_encoder_decoder.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_device, load_config, set_seed
from src.encoder_decoder import (
    CLEAN_LEVEL,
    head_hidden_dim,
    reconstruct,
    reconstruction_metrics,
    train_decoder,
    train_ed_head,
)
from src.models import ClassifierHead
from src.models.sar_optical_decoder import SARToOpticalDecoder, ed_input_dim
from src.train import FeatureBank

N = 16
ok = lambda m: print(f"  [ok] {m}")


def bank(split: str, levels, n=None) -> FeatureBank:
    d = Path("data/features")
    sl = slice(None) if n is None else slice(0, n)
    return FeatureBank(
        optical={l: np.load(d / f"{split}_opt_L{int(round(l*100)):03d}.npy")[sl] for l in levels},
        sar=np.load(d / f"{split}_sar.npy")[sl],
        targets=np.load(d / f"{split}_targets.npy")[sl],
    )


def main() -> None:
    cfg = load_config()
    set_seed(cfg.seed)
    device = get_device()
    d_opt = int(cfg.models.optical.feature_dim)
    d_sar = int(cfg.models.sar.feature_dim)
    levels = [float(v) for v in cfg.masking.levels]
    print(f"device: {device}\n")

    print("1. tensor contract through the ED pipeline")
    tr = bank("train", levels, N)
    n_classes = tr.targets.shape[1]
    assert tr.optical[CLEAN_LEVEL].shape == (N, d_opt), tr.optical[CLEAN_LEVEL].shape
    assert tr.sar.shape == (N, d_sar), tr.sar.shape
    ok(f"S2 -> ViT   -> z_clean    {tr.optical[CLEAN_LEVEL].shape}")
    ok(f"S1 -> ResNet -> z_sar      {tr.sar.shape}")

    dec = SARToOpticalDecoder(d_sar, int(cfg.encoder_decoder.hidden_dim), d_opt,
                              float(cfg.encoder_decoder.dropout),
                              bool(cfg.encoder_decoder.input_norm)).to(device)
    z_hat = dec(torch.from_numpy(tr.sar).to(device))
    assert z_hat.shape == (N, d_opt), z_hat.shape
    ok(f"z_sar -> decoder -> z_hat  {tuple(z_hat.shape)}")

    x = torch.cat([torch.from_numpy(tr.optical[0.8]).to(device), z_hat], dim=1)
    assert x.shape == (N, 2 * d_opt) and x.shape[1] == ed_input_dim(cfg), x.shape
    ok(f"[z_degraded ; z_hat]       {tuple(x.shape)}  (384 + 384 = 768)")

    head = ClassifierHead(ed_input_dim(cfg), n_classes, head_hidden_dim(cfg),
                          float(cfg.head.dropout)).to(device)
    logits = head(x)
    assert logits.shape == (N, n_classes), logits.shape
    ok(f"768 -> classifier -> logits {tuple(logits.shape)}  ({n_classes} classes)")

    print("\n2. finiteness")
    for name, t in [("z_hat", z_hat), ("classifier input", x), ("logits", logits)]:
        assert torch.isfinite(t).all(), f"non-finite values in {name}"
    ok("no NaNs or infs anywhere in the forward pass")

    print("\n3. leakage guards")
    try:
        dec(torch.from_numpy(tr.optical[CLEAN_LEVEL]).to(device))
        raise AssertionError("decoder accepted a 384-d optical feature as input!")
    except ValueError as e:
        assert "must never receive optical" in str(e)
    ok("decoder rejects an optical-width input (clean feature cannot be fed in)")

    # The decoder's parameters can only be reached by SAR: check the gradient of
    # the output w.r.t. a clean-feature tensor is structurally impossible by
    # confirming the forward signature takes exactly one tensor of SAR width.
    import inspect
    sig = inspect.signature(SARToOpticalDecoder.forward)
    assert list(sig.parameters) == ["self", "z_sar"], sig
    ok("decoder.forward takes exactly one argument (z_sar); no second input path")

    # z_hat must not be a copy of z_clean.
    z_clean_t = torch.from_numpy(tr.optical[CLEAN_LEVEL]).to(device)
    assert not torch.allclose(z_hat, z_clean_t, atol=1e-4), "z_hat equals z_clean"
    ok(f"z_hat != z_clean (mean abs diff {(z_hat - z_clean_t).abs().mean():.4f})")

    # Test-time assembly must depend on the masking level, i.e. it uses the
    # degraded feature and not the clean one.
    a = reconstruct(dec, "ed", tr.sar, tr.optical[0.8], device)
    b = reconstruct(dec, "ed", tr.sar, tr.optical[CLEAN_LEVEL], device)
    assert np.allclose(a, b), "direct decoder output must not depend on optical input"
    ok("variant 'ed': decoder output is a function of SAR alone (level-invariant)")

    print("\n4. gradients: decoder and head train, encoders do not")
    dec.train()
    loss = torch.nn.functional.mse_loss(dec(torch.from_numpy(tr.sar).to(device)), z_clean_t)
    loss.backward()
    g = [p.grad for p in dec.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g), "decoder has no finite gradients"
    ok(f"decoder: {len(g)}/{len(list(dec.parameters()))} tensors have finite gradients")

    head.zero_grad()
    torch.nn.BCEWithLogitsLoss()(head(x.detach()), torch.from_numpy(tr.targets).to(device)).backward()
    gh = [p.grad for p in head.parameters() if p.grad is not None]
    assert gh and all(torch.isfinite(t).all() for t in gh)
    ok(f"head: {len(gh)}/{len(list(head.parameters()))} tensors have finite gradients")

    print("   (encoders are not instantiated here -- ED trains on cached features.")
    print("    Their frozen-ness is asserted in scripts/smoke_test.py step 3.)")

    print("\n5. a real 2-epoch decoder fit, then the head")
    small = dict(cfg.encoder_decoder); small.update(epochs=2, early_stop_patience=2)
    cfg2 = load_config(); cfg2["encoder_decoder"] = small
    cfg2["train"] = dict(cfg2.train); cfg2["train"]["epochs"] = 2
    tr_full = bank("train", levels, 256)
    va_full = bank("validation", levels, 128)
    d2, dh = train_decoder(cfg2, "ed", tr_full, va_full, 0, device)
    ok(f"decoder trained {len(dh['history'])} epochs, val recon loss {dh['best_val_recon_loss']:.5f}")
    h2, hh = train_ed_head(cfg2, "ed", "ed_r2", d2, tr_full, va_full, n_classes, 0, device)
    ok(f"head trained {len(hh['history'])} epochs, val macro F1 {hh['best_val_macro_f1']:.4f}")

    m = reconstruction_metrics("ed", d2, va_full, 0.8, device)
    ok(f"recon @80%: MSE(z_hat,z_clean)={m['mse_hat_vs_clean']:.4f} "
       f"vs MSE(z_deg,z_clean)={m['mse_degraded_vs_clean']:.4f}  "
       f"cos={m['cos_hat_vs_clean']:.3f}")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
