"""Sentinel-1 SAR encoder: SSL4EO-S12 MoCo ResNet-50 (VV/VH), used frozen.

Why a pretrained encoder rather than a small CNN trained from scratch
--------------------------------------------------------------------
Symmetry. Both branches are then frozen, both are MoCo-pretrained on the *same*
SSL4EO-S12 corpus, and neither has seen a BigEarthNet label. Any gain from B to
C is therefore attributable to the extra *modality*, not to the SAR branch
receiving trainable capacity that the optical baseline does not get. A
from-scratch CNN would confound those two explanations, which is exactly the
distinction the experiment is supposed to resolve.

``build_sar_encoder(..., scratch=True)`` returns the small from-scratch CNN
instead, kept as an ablation.
"""
from __future__ import annotations

import timm
import torch
from huggingface_hub import hf_hub_download
from torch import nn


class SmallSARCNN(nn.Module):
    """~0.2M-parameter conv net, the ablation alternative to the frozen ResNet."""

    def __init__(self, in_chans: int = 2, feature_dim: int = 256):
        super().__init__()
        widths = [32, 64, 128, feature_dim]
        layers: list[nn.Module] = []
        prev = in_chans
        for w in widths:
            layers += [
                nn.Conv2d(prev, w, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
            ]
            prev = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)


def build_sar_encoder(cfg, scratch: bool = False, verbose: bool = True) -> nn.Module:
    n_bands = len(cfg.bands.s1)
    if scratch:
        model = SmallSARCNN(in_chans=n_bands)
        if verbose:
            n = sum(p.numel() for p in model.parameters())
            print(f"  SAR encoder: SmallSARCNN ({n/1e6:.2f}M params), trained from scratch")
        return model

    spec = cfg.models.sar
    model = timm.create_model(spec.timm_name, in_chans=n_bands, num_classes=0)
    ckpt_path = hf_hub_download(repo_id=spec.hf_repo, filename=spec.weight_file)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    conv1 = state["conv1.weight"]
    if conv1.shape[1] != n_bands:
        raise ValueError(f"checkpoint expects {conv1.shape[1]} SAR bands, config has {n_bands}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith("fc.")]
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
    if verbose:
        n = sum(p.numel() for p in model.parameters())
        print(f"  SAR encoder: {spec.timm_name} ({n/1e6:.1f}M params), frozen")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
