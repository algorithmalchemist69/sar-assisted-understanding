"""Frozen-encoder feature extraction, with an on-disk cache.

Both encoders are frozen, so a patch's representation at a given masking level
never changes. We therefore run the encoders exactly once per
(split, modality, masking level) and cache the vectors; every model in the study
then trains on cached features in seconds. This is what keeps the full
experiment matrix -- 4 arms x 2 regimes x 6 levels x 3 seeds -- cheap enough to
run end to end on a laptop.

SAR features are extracted once only: Sentinel-1 is never degraded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import PROJECT_ROOT, Config
from .dataset import PatchReader
from .masking import generate_mask
from .preprocessing import (
    assert_finite,
    normalize_s1,
    normalize_s2,
    resize,
    resize_mask,
    s2_fill_values,
)


def feature_dir(cfg: Config) -> Path:
    """Cache directory, namespaced by the masked-value strategy.

    The fill strategy changes the pixels the encoder sees, so features computed
    under different fills must never share a cache.
    """
    fill = cfg.masking.fill
    name = "features" if fill == "mean" else f"features_{fill}"
    d = PROJECT_ROOT / "data" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


class OpticalPatches(Dataset):
    """Sentinel-2 patches, masked at a fixed level, ready for the ViT."""

    def __init__(self, cfg: Config, frame: pd.DataFrame, level: float):
        self.cfg = cfg
        self.ids = frame.patch_id.tolist()
        self.level = float(level)
        self.reader = PatchReader(cfg)
        self.size = int(cfg.preprocess.patch_size)
        self.out_size = int(cfg.preprocess.model_input_size)
        self.fill = s2_fill_values(
            float(cfg.preprocess.s2_scale),
            len(cfg.bands.s2),
            cfg.masking.fill,
            float(cfg.masking.bright_value),
        )
        self.mask_kwargs = dict(
            size=self.size,
            sigma=float(cfg.masking.smoothing_sigma),
            mask_seed=int(cfg.masking.mask_seed),
            mode=cfg.masking.mode,
        )

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        pid = self.ids[i]
        img = self.reader.read_s2(pid)                       # (12, 120, 120) raw
        mask = generate_mask(pid, self.level, **self.mask_kwargs)
        return torch.from_numpy(img), torch.from_numpy(mask)


class SARPatches(Dataset):
    """Sentinel-1 patches. Never masked -- SAR is the observation that survives."""

    def __init__(self, cfg: Config, frame: pd.DataFrame):
        self.cfg = cfg
        self.names = frame.s1_name.tolist()
        self.reader = PatchReader(cfg)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        return torch.from_numpy(self.reader.read_s1(self.names[i]))


@torch.no_grad()
def extract_optical(
    cfg: Config, frame: pd.DataFrame, level: float, encoder, device, desc: str = ""
) -> np.ndarray:
    ds = OpticalPatches(cfg, frame, level)
    dl = DataLoader(
        ds,
        batch_size=int(cfg.eval.extract_batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    out_size = int(cfg.preprocess.model_input_size)
    fill = torch.tensor(ds.fill, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    scale = float(cfg.preprocess.s2_scale)

    feats = []
    for img, mask in tqdm(dl, desc=desc, leave=False):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        # Resize the imagery bilinearly but the mask by nearest neighbour, then
        # re-apply the fill at model resolution so cloud edges stay hard.
        img = resize(img, out_size, mode="bilinear")
        m = resize_mask(mask, out_size).unsqueeze(1)
        img = torch.where(m, fill.expand_as(img), img)
        x = normalize_s2(img, scale)
        assert_finite(x, "normalised S2")
        feats.append(encoder(x).float().cpu())
    return torch.cat(feats).numpy()


@torch.no_grad()
def extract_sar(cfg: Config, frame: pd.DataFrame, encoder, device, desc: str = "") -> np.ndarray:
    ds = SARPatches(cfg, frame)
    dl = DataLoader(
        ds,
        batch_size=int(cfg.eval.extract_batch_size),
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    out_size = int(cfg.preprocess.model_input_size)
    feats = []
    for img in tqdm(dl, desc=desc, leave=False):
        img = img.to(device, non_blocking=True)
        x = normalize_s1(
            img, list(cfg.preprocess.s1_mean), list(cfg.preprocess.s1_std),
            list(cfg.preprocess.s1_clip_db),
        )
        x = resize(x, out_size, mode="bilinear")
        assert_finite(x, "normalised S1")
        feats.append(encoder(x).float().cpu())
    return torch.cat(feats).numpy()


def cache_path(cfg: Config, split: str, modality: str, level: float | None) -> Path:
    tag = "sar" if modality == "sar" else f"opt_L{int(round((level or 0.0) * 100)):03d}"
    return feature_dir(cfg) / f"{split}_{tag}.npy"


def load_features(cfg: Config, split: str, modality: str, level: float | None) -> np.ndarray:
    p = cache_path(cfg, split, modality, level)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing. Run scripts/02_extract_features.py first.")
    return np.load(p)
