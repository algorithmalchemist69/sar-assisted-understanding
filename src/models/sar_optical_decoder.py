"""SAR -> optical feature decoder. An *additional* experiment, not a replacement.

What this tests, and how it differs from the fusion arm
------------------------------------------------------
The existing arm C (``fusion``) concatenates the degraded optical feature with
the raw SAR feature and lets the classifier head work out what to do with it.
It answers:

    "Does SAR carry information that is *useful for classification* on top of
     degraded optical?"

This module tests a different hypothesis:

    "Can SAR *predict the optical representation that a clean Sentinel-2 image
     would have produced*?"

That is a strictly stronger claim. Fusion only needs SAR to be discriminative;
reconstruction needs SAR to be discriminative *in the same coordinate system the
ViT uses*. The two can come apart -- SAR can help a classifier while being a
poor predictor of ViT features, and vice versa -- which is precisely why it is
worth running as a separate arm rather than assuming one implies the other.

Nothing here reconstructs Sentinel-2 *pixels*. The target is the 384-d ViT
feature vector, never the image.

Leakage
-------
The decoder's only input is ``z_sar``. The clean optical feature ``z_clean``
appears exclusively as a regression *target* during decoder training, and only
on the training split. ``forward()`` takes a single tensor and its input
dimension is asserted to be the SAR dimension, so there is no code path by which
a clean optical feature can reach the decoder as an input.
"""
from __future__ import annotations

import torch
from torch import nn

VARIANTS = ("ed", "ed_residual")


class SARToOpticalDecoder(nn.Module):
    """``z_sar`` (2048) -> ``z_hat_clean`` (384). The only trained component.

    Deliberately a plain MLP. The point of the experiment is to test whether the
    mapping exists at all, not to maximise capacity; a transformer decoder here
    would confound "SAR predicts optical features" with "a bigger head fits
    better".

    ``input_norm`` (LayerNorm on the SAR input) is a small departure from the
    minimal spec and is on by default for a measured reason: the SSL4EO ResNet-50
    features have mean L2 norm 43.7 on the training split but 24.0 on test, a
    scale shift large enough that an unnormalised first layer transfers a
    training-set scale the test set does not share. Set ``input_norm: false`` in
    the config to reproduce the strictly minimal version.
    """

    def __init__(
        self,
        in_dim: int = 2048,
        hidden_dim: int = 512,
        out_dim: int = 384,
        dropout: float = 0.3,
        input_norm: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        layers: list[nn.Module] = []
        if input_norm:
            layers.append(nn.LayerNorm(self.in_dim))
        layers += [
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.out_dim),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z_sar: torch.Tensor) -> torch.Tensor:
        # Hard guard against the one leakage mode that would invalidate the
        # experiment: anything of optical width must never arrive here.
        if z_sar.shape[-1] != self.in_dim:
            raise ValueError(
                f"decoder expects SAR features of dim {self.in_dim}, "
                f"got {z_sar.shape[-1]}. The decoder must never receive optical "
                f"features as input."
            )
        return self.net(z_sar)


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str = "mse",
    cosine_weight: float = 1.0,
) -> torch.Tensor:
    """MSE, cosine, or their sum.

    MSE is the primary target because it is directly interpretable against the
    baseline "how far is the degraded feature from the clean one" in the same
    units. Cosine is offered because the downstream classifier is a linear map
    of the feature, which is more sensitive to direction than to magnitude.
    """
    if kind == "mse":
        return nn.functional.mse_loss(pred, target)
    cos = 1.0 - nn.functional.cosine_similarity(pred, target, dim=-1).mean()
    if kind == "cosine":
        return cos
    if kind in ("mse+cosine", "both"):
        return nn.functional.mse_loss(pred, target) + float(cosine_weight) * cos
    raise ValueError(f"unknown reconstruction_loss: {kind}")


def ed_input_dim(cfg) -> int:
    """Classifier input width: [z_degraded ; z_hat_clean], both optical-sized."""
    return 2 * int(cfg.models.optical.feature_dim)
