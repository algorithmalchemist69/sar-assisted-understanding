"""Configuration loading and global seeding."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


class Config(dict):
    """dict with attribute access, so cfg.data.lmdb_name works."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as fh:
        cfg = Config(yaml.safe_load(fh))
    # $DATA_DIR wins over the config file so the project relocates without edits.
    env_dir = os.environ.get("DATA_DIR")
    if env_dir:
        cfg["data"]["data_dir"] = env_dir
    return cfg


def data_dir(cfg: Config) -> Path:
    d = Path(cfg.data.data_dir)
    return d if d.is_absolute() else PROJECT_ROOT / d


def results_dir(cfg: Config, *parts: str) -> Path:
    d = PROJECT_ROOT / "results"
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int) -> None:
    """Seed every RNG we rely on."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)  # MPS lacks deterministic kernels


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
