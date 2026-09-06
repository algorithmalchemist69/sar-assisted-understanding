#!/usr/bin/env bash
# Full pipeline, start to finish. ~55 minutes on an Apple M4 Pro (MPS).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 0/6 download (~2.7 GB) ==" && python3 scripts/00_download.py
echo "== 1/6 subset + data audit ==" && python3 scripts/01_build_subset.py
echo "== 2/6 frozen-encoder features (~12 min) ==" && python3 scripts/02_extract_features.py
echo "== 3/6 train + evaluate all arms (~1 min) ==" && python3 scripts/03_run_experiments.py
echo "== 4/6 gain tables + bootstrap CIs ==" && python3 scripts/04_analyse.py
echo "== 5/6 figures ==" && python3 scripts/05_make_figures.py
echo "== 6/6 qualitative examples ==" && python3 scripts/06_qualitative.py
echo
echo "Optional ablation (~13 min): python3 scripts/07_ablation_fill.py"
echo "Done. See results/metrics and results/figures."
