#!/bin/bash
# Recover the training corpus from the saved manifests, disk-safely:
# re-download -> convert to npy (deleting each series' DICOM after caching)
# -> validate the cache with the public numpy API -> launch the big run.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
mkdir -p logs

echo "=== $(date) downloads ===" >> logs/restage.log
for f in data/m_*.csv; do
  echo "--- $f ---" >> logs/restage.log
  uv run --no-sync livecodec-cohort download --n 9999 --manifest "$f" --dest data/dicom \
    >> logs/restage.log 2>&1
done

echo "=== $(date) convert + validate ===" >> logs/restage.log
uv run --no-sync python scripts/convert_and_validate.py >> logs/restage.log 2>&1 \
  || { echo "CONVERT FAILED" >> logs/restage.log; exit 1; }

echo "=== $(date) big run ===" >> logs/restage.log
. ~/.livecodec-s3.env
export S3_ACCESS S3_SECRET
exec bash scripts/big_run.sh > logs/bigrun.log 2>&1
