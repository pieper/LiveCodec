#!/bin/bash
# Decoder-only runs against the FROZEN big-400k encoder (published latents stay
# valid — only the decoder changes, so each publish reuses the same code files).
# Isolates the two candidate fixes for preview fuzz:
#   dec-edge : baseline-size decoder (0.37M, 150 ms/chunk) + edge loss
#              -> does the LOSS fix the fuzz, at no decode cost?
#   dec-big  : config-A decoder (3.75M, 512 ms/chunk) + edge loss
#              -> does CAPACITY fix it, and is the 3.4x decode cost worth it?
# Then re-pack every published version so residuals stream res-progressively.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
mkdir -p logs
. ~/.livecodec-s3.env
export S3_ACCESS S3_SECRET
ENC=results/big-400k/model.pt

echo "=== $(date) dec-edge (0.37M + edge loss, warm start) ==="
uv run --no-sync livecodec-train3d --data data/dicom --cache data/npy \
  --steps 30000 --ckpt "$ENC" --freeze-encoder \
  --dec-arch 2.5d --enc-width 256 --enc-depth 4 --dec-width 64 \
  --edge-weight 0.15 --batch 8 --crop-z 32 --crop-xy 128 --dash-every 5000 \
  --run-name h100-dec-edge --out results/dec-edge > logs/train-dec-edge.log 2>&1 \
  || { echo "TRAIN FAILED dec-edge"; exit 1; }
uv run --no-sync --with boto3 python scripts/publish_version.py \
  --tag dec-edge --steps 400000 --ckpt results/dec-edge/model.pt --ood-dir data/ood \
  --note "400k encoder + 0.37M decoder retrained with edge loss (150 ms/chunk)" \
  > logs/publish-dec-edge.log 2>&1 || { echo "PUBLISH FAILED dec-edge"; exit 1; }

echo "=== $(date) dec-big (3.75M config A + edge loss, fresh decoder) ==="
uv run --no-sync livecodec-train3d --data data/dicom --cache data/npy \
  --steps 60000 --ckpt "$ENC" --freeze-encoder \
  --dec-arch 2.5d --enc-width 256 --enc-depth 4 \
  --dec-stages 160,128,96,48,24 --dec-mix-depth 2 --dec-d64 2 --dec-d128 1 \
  --edge-weight 0.15 --batch 8 --crop-z 32 --crop-xy 128 --dash-every 5000 \
  --run-name h100-dec-big --out results/dec-big > logs/train-dec-big.log 2>&1 \
  || { echo "TRAIN FAILED dec-big"; exit 1; }
uv run --no-sync --with boto3 python scripts/publish_version.py \
  --tag dec-big --steps 400000 --ckpt results/dec-big/model.pt --ood-dir data/ood \
  --note "400k encoder + 3.75M decoder, edge loss (512 ms/chunk, 7.5 MB weights)" \
  > logs/publish-dec-big.log 2>&1 || { echo "PUBLISH FAILED dec-big"; exit 1; }

echo "=== $(date) re-pack published versions with res-progressive residuals ==="
for TAG in v3-031vol big-025k big-050k big-100k big-200k big-400k; do
  CK=results/$TAG/model.pt
  [ "$TAG" = "v3-031vol" ] && CK=results/model-v3.pt
  [ -f "$CK" ] || { echo "skip $TAG (no checkpoint)"; continue; }
  rm -rf "bundles-versions/$TAG"
  uv run --no-sync --with boto3 python scripts/publish_version.py \
    --tag "$TAG" --steps "$(python3 -c "print({'v3-031vol':0}.get('$TAG', int('$TAG'.split('-')[1].rstrip('k'))*1000))")" \
    --ckpt "$CK" --ood-dir data/ood --note "res-progressive residual" \
    >> logs/repack.log 2>&1 || echo "REPACK FAILED $TAG"
done
echo "=== $(date) decoder run complete ==="
