#!/bin/bash
# v3 architecture runs. Two questions, two runs:
#   v3-fast : decoder-only vs the FROZEN big-400k encoder — does the fused
#             output projection + 128^2 preview head hit the <60 ms/chunk
#             target without losing quality? (published latents stay valid)
#   v3-rich : full model at fine_stride 4 — 4x more latent sites (~3 MB fine
#             tier) to answer whether the fuzz is a RATE problem.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
mkdir -p logs
. ~/.livecodec-s3.env
export S3_ACCESS S3_SECRET
ENC=results/big-400k/model.pt

echo "=== $(date) v3-fast (decoder-only, frozen 400k encoder) ==="
uv run --no-sync livecodec-train3d --data data/dicom --cache data/npy \
  --steps 40000 --ckpt "$ENC" --freeze-encoder \
  --dec-arch v3 --enc-width 256 --enc-depth 4 \
  --dec-stages 64,48,32 --dec-mix-depth 1 --dec-d64 1 \
  --edge-weight 0.15 --preview-weight 0.3 \
  --batch 8 --crop-z 32 --crop-xy 128 --dash-every 5000 \
  --run-name h100-v3-fast --out results/v3-fast > logs/train-v3-fast.log 2>&1 \
  || { echo "TRAIN FAILED v3-fast"; exit 1; }
uv run --no-sync --with boto3 python scripts/publish_version.py \
  --tag v3-fast --steps 400000 --ckpt results/v3-fast/model.pt --ood-dir data/ood \
  --note "400k encoder + v3 decoder (fused output, 128px preview head)" \
  > logs/publish-v3-fast.log 2>&1 || { echo "PUBLISH FAILED v3-fast"; exit 1; }

echo "=== $(date) v3-rich (full model, fine_stride 4 => ~3 MB fine tier) ==="
uv run --no-sync livecodec-train3d --data data/dicom --cache data/npy \
  --steps 150000 --dec-arch v3 --enc-width 256 --enc-depth 4 --fine-stride 4 \
  --dec-stages 64,48,32 --dec-mix-depth 1 --dec-d64 1 \
  --edge-weight 0.15 --preview-weight 0.3 \
  --batch 8 --crop-z 32 --crop-xy 128 --dash-every 5000 \
  --run-name h100-v3-rich --out results/v3-rich > logs/train-v3-rich.log 2>&1 \
  || { echo "TRAIN FAILED v3-rich"; exit 1; }
uv run --no-sync --with boto3 python scripts/publish_version.py \
  --tag v3-rich --steps 150000 --ckpt results/v3-rich/model.pt --ood-dir data/ood \
  --note "v3, fine_stride 4: ~3 MB fine tier (rate experiment)" \
  > logs/publish-v3-rich.log 2>&1 || { echo "PUBLISH FAILED v3-rich"; exit 1; }
echo "=== $(date) v3 runs complete ==="
