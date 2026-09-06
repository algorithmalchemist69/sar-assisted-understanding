# Data directory

Nothing here is committed; everything is downloaded by `scripts/00_download.py`.
Set `DATA_DIR` to relocate the raw data (default: `./data/raw`).

```
data/
├── raw/                                   # $DATA_DIR
│   ├── BENv2_lithuania_summer.lmdb/
│   │   └── data.mdb                       # 2.48 GB
│   └── metadata_lithuania_summer.parquet  # 0.25 MB
├── features/                              # derived: frozen-encoder cache (mean fill)
└── features_bright/                       # derived: cache for the fill ablation
```

## Source

**BigEarthNet v2.0 (reBEN)**, Lithuania / summer subset — 8,775 paired
Sentinel-1 / Sentinel-2 patches.

* Mirror used: <https://huggingface.co/datasets/hackelle/BigEarthNetV2-Lithuania-Summer-LMDB>
  (LMDB conversion produced with [rico-hdl](https://github.com/kai-tub/rico-hdl),
  the officially recommended conversion tool, by a reBEN co-author at TU Berlin/RSiM)
* Authoritative original: <https://zenodo.org/records/10891137> (118 GB — not used; see README)
* Licence: CDLA-Permissive-1.0

## Layout inside the LMDB

One SafeTensors record per patch, keyed by:

* `patch_id`  → Sentinel-2, 12 L2A bands, `uint16` reflectance × 10000,
  native sizes 120² (10 m), 60² (20 m), 20² (60 m)
* `s1_name`   → Sentinel-1, `VV`/`VH`, `float32` σ⁰ in dB, 120²

Both keys come from the same metadata row, which is what guarantees the pairing.
