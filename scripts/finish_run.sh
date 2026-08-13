#!/bin/bash
# Everything still owed, in value order, as ONE sequential script (no waiter
# processes — the previous chain deadlocked on a pgrep -f that matched itself).
# A failing stage logs and is skipped; later stages still run.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
mkdir -p logs
. ~/.livecodec-s3.env
export S3_ACCESS S3_SECRET
ENC=results/big-400k/model.pt

stage() { echo "=== $(date -u +%H:%M) $* ==="; }

# 1. re-pack every published version so residuals stream res-progressively
stage "re-pack published versions"
for TAG in v3-031vol big-025k big-050k big-100k big-200k big-400k dec-edge; do
  CK=results/$TAG/model.pt
  [ "$TAG" = "v3-031vol" ] && CK=results/model-v3.pt
  [ -f "$CK" ] || { echo "skip $TAG (no checkpoint)"; continue; }
  STEPS=$(python3 -c "
t='$TAG'
print(0 if t=='v3-031vol' else 400000 if t=='dec-edge' else int(t.split('-')[1].rstrip('k'))*1000)")
  rm -rf "bundles-versions/$TAG"
  uv run --no-sync --with boto3 python scripts/publish_version.py \
    --tag "$TAG" --steps "$STEPS" --ckpt "$CK" --ood-dir data/ood \
    --note "res-progressive residual" >> logs/repack.log 2>&1 \
    && echo "repacked $TAG" || echo "REPACK FAILED $TAG"
done

# 2. v3-fast: the decode-speed architecture (decoder-only, frozen encoder)
stage "v3-fast"
uv run --no-sync livecodec-train3d --data data/dicom --cache data/npy \
  --steps 40000 --ckpt "$ENC" --freeze-encoder \
  --dec-arch v3 --enc-width 256 --enc-depth 4 \
  --dec-stages 64,48,32 --dec-mix-depth 1 --dec-d64 1 \
  --edge-weight 0.15 --preview-weight 0.3 \
  --batch 8 --crop-z 32 --crop-xy 128 --dash-every 5000 \
  --run-name h100-v3-fast --out results/v3-fast > logs/train-v3-fast.log 2>&1 \
  && uv run --no-sync --with boto3 python scripts/publish_version.py \
       --tag v3-fast --steps 400000 --ckpt results/v3-fast/model.pt --ood-dir data/ood \
       --note "400k encoder + v3 decoder (fused output, 128px preview head)" \
       > logs/publish-v3-fast.log 2>&1 \
  && echo "v3-fast done" || echo "V3-FAST FAILED"

# 3. v3-rich: the rate knob (~3 MB fine tier) — the direct test of the fuzz
stage "v3-rich"
uv run --no-sync livecodec-train3d --data data/dicom --cache data/npy \
  --steps 150000 --dec-arch v3 --enc-width 256 --enc-depth 4 --fine-stride 4 \
  --dec-stages 64,48,32 --dec-mix-depth 1 --dec-d64 1 \
  --edge-weight 0.15 --preview-weight 0.3 \
  --batch 8 --crop-z 32 --crop-xy 128 --dash-every 5000 \
  --run-name h100-v3-rich --out results/v3-rich > logs/train-v3-rich.log 2>&1 \
  && uv run --no-sync --with boto3 python scripts/publish_version.py \
       --tag v3-rich --steps 150000 --ckpt results/v3-rich/model.pt --ood-dir data/ood \
       --note "v3, fine_stride 4: ~3 MB fine tier (rate experiment)" \
       > logs/publish-v3-rich.log 2>&1 \
  && echo "v3-rich done" || echo "V3-RICH FAILED"

# 4. dec-big: capacity diagnostic (lowest value — profiling says we can't ship it)
stage "dec-big (diagnostic)"
uv run --no-sync livecodec-train3d --data data/dicom --cache data/npy \
  --steps 60000 --ckpt "$ENC" --freeze-encoder \
  --dec-arch 2.5d --enc-width 256 --enc-depth 4 \
  --dec-stages 160,128,96,48,24 --dec-mix-depth 2 --dec-d64 2 --dec-d128 1 \
  --edge-weight 0.15 --batch 8 --crop-z 32 --crop-xy 128 --dash-every 5000 \
  --run-name h100-dec-big --out results/dec-big > logs/train-dec-big.log 2>&1 \
  && uv run --no-sync --with boto3 python scripts/publish_version.py \
       --tag dec-big --steps 400000 --ckpt results/dec-big/model.pt --ood-dir data/ood \
       --note "400k encoder + 3.75M decoder (512 ms/chunk — diagnostic only)" \
       > logs/publish-dec-big.log 2>&1 \
  && echo "dec-big done" || echo "DEC-BIG FAILED"

stage "finish_run complete"
