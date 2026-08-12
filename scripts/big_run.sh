#!/bin/bash
# The H100 big run: 256-wide/4-deep encoder on the ~960-volume corpus, with a
# demo "encoding version" published at each checkpoint milestone so the codec
# race can compare training effort. Cumulative milestones: 25k 50k 100k 200k 400k.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
. ~/.livecodec-s3.env
export S3_ACCESS S3_SECRET

SCHED=(25000 25000 50000 100000 200000)
TOTAL=0
CKPT=""
for DELTA in "${SCHED[@]}"; do
  TOTAL=$((TOTAL + DELTA))
  TAG=$(printf "big-%03dk" $((TOTAL / 1000)))
  echo "=== $(date) train to ${TOTAL} steps (${TAG}) ==="
  uv run --no-sync livecodec-train3d --data data/dicom --cache data/npy \
    --steps "$DELTA" ${CKPT:+--ckpt "$CKPT"} \
    --dec-arch 2.5d --enc-width 256 --enc-depth 4 \
    --batch 8 --crop-z 32 --crop-xy 128 --dash-every 4000 \
    --run-name "h100-${TAG}" --out "results/${TAG}" \
    > "logs/train-${TAG}.log" 2>&1 || { echo "TRAIN FAILED ${TAG}"; exit 1; }
  CKPT="results/${TAG}/model.pt"
  echo "=== $(date) publish ${TAG} ==="
  uv run --no-sync --with boto3 python scripts/publish_version.py \
    --tag "${TAG}" --steps "${TOTAL}" --ckpt "${CKPT}" --ood-dir data/ood \
    --note "~960-vol IDC corpus, 256w/4d encoder" \
    > "logs/publish-${TAG}.log" 2>&1 || { echo "PUBLISH FAILED ${TAG}"; exit 1; }
done
echo "=== $(date) big run complete ==="
