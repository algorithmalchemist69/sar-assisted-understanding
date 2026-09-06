#!/usr/bin/env bash
# Full pipeline, start to finish. ~62 minutes on an Apple M4 Pro (MPS).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 0/13 download (~2.7 GB) ==" && python3 scripts/00_download.py
echo "== 1/13 subset + data audit ==" && python3 scripts/01_build_subset.py
echo "== 2/13 frozen-encoder features (~12 min) ==" && python3 scripts/02_extract_features.py
echo "== 3/13 train + evaluate all arms (~1 min) ==" && python3 scripts/03_run_experiments.py
echo "== 4/13 gain tables + bootstrap CIs ==" && python3 scripts/04_analyse.py
echo "== 5/13 figures ==" && python3 scripts/05_make_figures.py
echo "== 6/13 qualitative examples ==" && python3 scripts/06_qualitative.py

# --- Additional experiment: SAR -> optical feature reconstruction (arm ED) ---
# Reuses the cached features from stage 2, so this adds ~5 min, not another run.
echo "== 7/13 ED smoke test ==" && python3 scripts/smoke_test_ed.py
echo "== 8/13 ED train + evaluate (~1 min) ==" && python3 scripts/08_encoder_decoder.py
echo "== 9/13 matched B/C/ED, 10 seeds (~3.5 min) ==" && python3 scripts/09_ed_compare.py --seeds 10
echo "== 10/13 ED figure ==" && python3 scripts/10_ed_figures.py
echo "== 11/13 unified tables, all arms ==" && python3 scripts/12_unified_analysis.py
echo "== 12/13 unified figures ==" && python3 scripts/13_unified_figures.py
echo "== 13/13 unified qualitative ==" && python3 scripts/14_qualitative_unified.py

echo
echo "Optional ablation (~13 min):  python3 scripts/07_ablation_fill.py"
echo "Optional leakage audit:       python3 scripts/11_ed_leakage_audit.py"
echo "Done. See results/metrics and results/figures."
