"""reBEN (BigEarthNet v2.0) Lithuania-summer subset: metadata, splits, patch I/O.

The LMDB stores one SafeTensors record per patch, keyed by the Sentinel-2
``patch_id`` (12 L2A bands) and by the Sentinel-1 ``s1_name`` (VV, VH). Both keys
come from the same metadata row, which is how S1/S2 pairing is guaranteed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from safetensors.numpy import load as safetensor_load

from .config import Config, data_dir

SPLITS = ("train", "validation", "test")


@dataclass
class Subset:
    """The experiment's data selection: a frame plus the fixed class vocabulary."""

    frame: pd.DataFrame          # patch_id, s1_name, split, labels, y (multi-hot)
    classes: list[str]

    def split(self, name: str) -> pd.DataFrame:
        return self.frame[self.frame.split == name].reset_index(drop=True)

    @property
    def targets(self) -> np.ndarray:
        return np.stack(self.frame.y.to_numpy())


def load_metadata(cfg: Config) -> pd.DataFrame:
    path = data_dir(cfg) / cfg.data.metadata_name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/00_download.py first."
        )
    return pd.read_parquet(path)


def build_subset(cfg: Config, verbose: bool = True) -> Subset:
    """Select classes and patches. Deterministic: no RNG is involved.

    Class selection keeps every CLC-19 class with at least
    ``data.min_class_positives`` positive patches in the subset; patches left
    without any positive label are dropped.
    """
    meta = load_metadata(cfg)
    counts = pd.Series(
        [c for labels in meta.labels for c in labels]
    ).value_counts()
    classes = sorted(counts[counts >= cfg.data.min_class_positives].index.tolist())
    if not classes:
        raise ValueError("no class passed the min_class_positives filter")
    index = {c: i for i, c in enumerate(classes)}

    def multi_hot(labels) -> np.ndarray:
        y = np.zeros(len(classes), dtype=np.float32)
        for c in labels:
            if c in index:
                y[index[c]] = 1.0
        return y

    meta = meta.copy()
    meta["y"] = [multi_hot(v) for v in meta.labels]
    if cfg.data.drop_empty_label_patches:
        keep = np.array([y.sum() > 0 for y in meta.y])
        if verbose:
            print(f"  dropping {int((~keep).sum())} patches with no kept label")
        meta = meta[keep]
    meta = meta.sort_values("patch_id").reset_index(drop=True)

    if verbose:
        print(f"  classes kept: {len(classes)} of {len(counts)}")
        for c in classes:
            print(f"    {counts[c]:6d}  {c}")
        print(f"  patches: {len(meta)}")
        print(f"  split sizes: {meta.split.value_counts().to_dict()}")
    return Subset(frame=meta, classes=classes)


class PatchReader:
    """Random-access reader over the reBEN LMDB.

    Sentinel-2 bands are stored at native resolution (120/60/20 px for the
    10/20/60 m bands); every band is resampled to 120x120 by nearest-neighbour
    upsampling so that no band invents detail it does not have.
    """

    def __init__(self, cfg: Config):
        self.path = data_dir(cfg) / cfg.data.lmdb_name
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run scripts/00_download.py first."
            )
        self.s2_bands = list(cfg.bands.s2)
        self.s1_bands = list(cfg.bands.s1)
        self.size = int(cfg.preprocess.patch_size)
        self._env: lmdb.Environment | None = None

    @property
    def env(self) -> lmdb.Environment:
        # Opened lazily so the reader survives being sent to dataloader workers.
        if self._env is None:
            self._env = lmdb.open(
                str(self.path), readonly=True, lock=False, readahead=False, meminit=False
            )
        return self._env

    def _record(self, key: str) -> dict[str, np.ndarray]:
        with self.env.begin(write=False) as txn:
            raw = txn.get(key.encode())
        if raw is None:
            raise KeyError(f"{key} not present in {self.path.name}")
        return safetensor_load(raw)

    def _upsample(self, band: np.ndarray) -> np.ndarray:
        if band.shape == (self.size, self.size):
            return band
        factor = self.size // band.shape[0]
        if factor * band.shape[0] != self.size:
            raise ValueError(f"band shape {band.shape} does not tile into {self.size}")
        return np.repeat(np.repeat(band, factor, axis=0), factor, axis=1)

    def read_s2(self, patch_id: str) -> np.ndarray:
        """(12, 120, 120) float32, raw L2A reflectance counts (scaled by 10000)."""
        rec = self._record(patch_id)
        return np.stack(
            [self._upsample(rec[b]).astype(np.float32) for b in self.s2_bands]
        )

    def read_s1(self, s1_name: str) -> np.ndarray:
        """(2, 120, 120) float32, sigma-nought in dB."""
        rec = self._record(s1_name)
        return np.stack(
            [self._upsample(rec[b]).astype(np.float32) for b in self.s1_bands]
        )
