#!/usr/bin/env python3
"""Fetch the dataset subset and the two pretrained encoders (~2.7 GB total).

Data:  BigEarthNet v2.0 (reBEN), Lithuania / summer subset, LMDB-converted with
       rico-hdl.  https://huggingface.co/datasets/hackelle/BigEarthNetV2-Lithuania-Summer-LMDB
Models: SSL4EO-S12 MoCo ViT-S/16 (Sentinel-2) and ResNet-50 (Sentinel-1),
       redistributed by torchgeo.

Everything lands under $DATA_DIR (default ./data/raw) except the model weights,
which go to the standard Hugging Face cache.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download

from src.config import data_dir, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = data_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    lmdb_dir = out / cfg.data.lmdb_name
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {out}")
    for remote, local in [
        (f"{cfg.data.lmdb_name}/data.mdb", lmdb_dir / "data.mdb"),
        (cfg.data.metadata_name, out / cfg.data.metadata_name),
    ]:
        if local.exists():
            print(f"  [skip] {local.name} already present ({local.stat().st_size/1e9:.2f} GB)")
            continue
        print(f"  downloading {remote} ...")
        path = hf_hub_download(
            repo_id=cfg.data.hf_repo, filename=remote, repo_type="dataset"
        )
        shutil.copyfile(path, local)
        print(f"  -> {local} ({local.stat().st_size/1e9:.2f} GB)")

    # LMDB refuses to open a directory without a lock file it can create; readonly
    # access with lock=False does not need one, but we make sure the dir is sane.
    if not (lmdb_dir / "data.mdb").exists():
        raise SystemExit(f"{lmdb_dir/'data.mdb'} missing after download")

    if not args.skip_models:
        print("Pretrained encoders (into the Hugging Face cache):")
        for spec in (cfg.models.optical, cfg.models.sar):
            p = hf_hub_download(repo_id=spec.hf_repo, filename=spec.weight_file)
            print(f"  {spec.hf_repo} -> {p}")

    print("Done.")


if __name__ == "__main__":
    main()
