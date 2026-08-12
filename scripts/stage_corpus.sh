#!/bin/bash
# Stage the big-run corpus + fixed OOD demo scans on the GPU instance.
# Training: ~960 series across 5 collections (~110 GB). OOD: 6 scans from
# collections EXCLUDED from training. Run from the LiveCodec repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data logs
m() { uv run --no-sync livecodec-cohort manifest "$@"; }
d() { uv run --no-sync livecodec-cohort download --n 9999 "$@"; }

m --collection ct_lymph_nodes            --order random --min-mb 60  --max-series 100 --out data/m_ctln.csv
m --collection mediastinal_lymph_node_seg --order random --min-mb 20 --max-series 250 --out data/m_mlns.csv
m --collection nlst                      --order random --min-mb 60  --max-series 400 --out data/m_nlst.csv
m --collection ct_colonography           --order random --min-mb 80  --max-series 150 --out data/m_colo.csv
m --collection pancreas_ct               --order random --min-mb 60  --max-series 60  --out data/m_panc.csv
for f in data/m_*.csv; do d --manifest "$f" --dest data/dicom; done

# OOD (never trained): spine mets x2, lung x2, pediatric x1, pancreas-contrast x1
m --collection spine_mets_ct_seg         --order random --min-mb 100 --max-series 2 --seed 7 --out data/o_spine.csv
m --collection qin_lung_ct               --order random --min-mb 30  --max-series 2 --seed 7 --out data/o_lung.csv
m --collection pediatric_ct_seg          --order random --min-mb 60  --max-series 1 --seed 7 --out data/o_ped.csv
m --collection ctpred_sunitinib_pannet   --order random --min-mb 80  --max-series 1 --seed 7 --out data/o_pannet.csv
for f in data/o_*.csv; do d --manifest "$f" --dest data/ood; done

# demo decoder is baked to 512x512 slices — drop any OOD series that isn't
uv run --no-sync python - <<'EOF'
from pathlib import Path
import shutil, sys
sys.path.insert(0, "src")
from livecodec.dicom import load_series
for d in sorted({p.parent for p in Path("data/ood").rglob("*.dcm")}):
    try:
        vol, _ = load_series(d)
        if vol.shape[1] != 512 or vol.shape[2] != 512 or vol.shape[0] < 64:
            print("dropping (shape)", d.name[-24:], vol.shape); shutil.rmtree(d)
    except Exception as e:
        print("dropping (error)", d.name[-24:], e); shutil.rmtree(d)
EOF
du -sh data/dicom data/ood
