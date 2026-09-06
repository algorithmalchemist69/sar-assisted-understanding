"""Fusion head. The *only* thing that is trained in this study.

Arms
----
``optical``    384-d ViT features                       -> conditions A and B
``fusion``     [384-d optical ; 2048-d SAR]             -> condition C
``fusion_shuf``same as ``fusion`` but the SAR features are permuted across
               samples at load time. This is the control that separates "SAR
               carries complementary information" from "the fusion head simply
               has more parameters".
``sar``        2048-d SAR features only                 -> reference D

Every arm uses the identical head topology and training recipe, so the arms
differ only in what is fed to them.
"""
from __future__ import annotations

from torch import nn

ARMS = ("optical", "fusion", "fusion_shuf", "sar")


def arm_input_dim(arm: str, cfg) -> int:
    d_opt = int(cfg.models.optical.feature_dim)
    d_sar = int(cfg.models.sar.feature_dim)
    if arm == "optical":
        return d_opt
    if arm in ("fusion", "fusion_shuf"):
        return d_opt + d_sar
    if arm == "sar":
        return d_sar
    raise ValueError(f"unknown arm: {arm}")


class ClassifierHead(nn.Module):
    """LayerNorm -> Linear -> ReLU -> Dropout -> Linear, on concatenated features.

    LayerNorm at the input matters here: the optical (ViT, 384-d) and SAR
    (ResNet, 2048-d) features have quite different scales, and without it the
    larger-magnitude branch would dominate the first layer purely by scale.
    """

    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)
