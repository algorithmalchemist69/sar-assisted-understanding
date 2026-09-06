"""Sentinel-2 optical encoder: SSL4EO-S12 MoCo ViT-S/16, used frozen.

Why this checkpoint
-------------------
It is self-supervised (MoCo-v2) on SSL4EO-S12, a *label-free* Sentinel-1/2
corpus. The obvious alternative -- the BIFOLD ViTs trained on BigEarthNet v2.0
itself -- would leak: our test patches are in their training set.

The 12-vs-13 band problem
-------------------------
SSL4EO-S12 pretrained on 13 Sentinel-2 **L1C** bands
(B1..B9, B10, B11, B12). reBEN is **L2A** and has no B10 (cirrus), leaving 12.
Since the SSL4EO preprocessing is a plain ``x / 10000`` with *no* mean
subtraction, feeding ``B10 = 0`` contributes exactly zero to the patch
embedding. Deleting input-channel slice 10 from ``patch_embed.proj.weight`` is
therefore mathematically identical to imputing B10 with its pretraining mean --
not an approximation, and not a random re-initialisation.
"""
from __future__ import annotations

import timm
import torch
from huggingface_hub import hf_hub_download
from torch import nn


def build_optical_encoder(cfg, verbose: bool = True) -> nn.Module:
    spec = cfg.models.optical
    n_bands = len(cfg.bands.s2)
    drop_index = int(cfg.bands.ssl4eo_b10_index)

    model = timm.create_model(spec.timm_name, in_chans=n_bands, num_classes=0)

    ckpt_path = hf_hub_download(repo_id=spec.hf_repo, filename=spec.weight_file)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    key = "patch_embed.proj.weight"
    w = state[key]
    if w.shape[1] == n_bands + 1:
        keep = [i for i in range(w.shape[1]) if i != drop_index]
        state[key] = w[:, keep]
        if verbose:
            print(
                f"  patch_embed: {tuple(w.shape)} -> {tuple(state[key].shape)} "
                f"(dropped SSL4EO band index {drop_index} = B10, absent from L2A)"
            )
    elif w.shape[1] != n_bands:
        raise ValueError(f"checkpoint expects {w.shape[1]} bands, config has {n_bands}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    # The classification head is intentionally absent (num_classes=0).
    unexpected = [k for k in unexpected if not k.startswith("head.")]
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"  optical encoder: {spec.timm_name} ({n/1e6:.1f}M params), frozen")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
