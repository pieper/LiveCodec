#!/bin/zsh
# Overnight Phase 1 run: extend the local cohort while training on what's
# here, then resume-train on the full corpus. Run under caffeinate:
#   caffeinate -dims zsh scripts/overnight.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results data

echo "=== $(date) manifest + background download (36 series total) ==="
uv run livecodec-cohort manifest --collection ct_lymph_nodes --modality CT \
    --max-series 36 --out data/ctln36.csv
uv run livecodec-cohort download --manifest data/ctln36.csv --n 36 --dest data/dicom \
    > results/overnight-download.log 2>&1 &
DL=$!

echo "=== $(date) run A: 50k steps from scratch on current data ==="
uv run --extra train livecodec-train2d --data data/dicom --steps 50000 --batch 16 \
    --ssim-weight 0.2 --out results/phase1-longA > results/overnight-trainA.log 2>&1

echo "=== $(date) waiting for download ==="
wait $DL || echo "download exited nonzero (continuing with whatever arrived)"

echo "=== $(date) run B: resume on enlarged corpus, 50k more steps ==="
uv run --extra train livecodec-train2d --data data/dicom --steps 50000 --batch 16 \
    --ssim-weight 0.2 --ckpt results/phase1-longA/model.pt \
    --out results/phase1-longB > results/overnight-trainB.log 2>&1

echo "=== $(date) overnight complete ==="
tail -8 results/overnight-trainB.log
