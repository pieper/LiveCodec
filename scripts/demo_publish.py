"""Pack demo scans with the v2 model and upload bundles to the Ceph bucket.

Runs on the GPU instance. Expects S3 credentials in the environment
(S3_ACCESS / S3_SECRET; source them from a chmod-600 env file, never argv).

  uv run --no-sync --with boto3 python scripts/demo_publish.py \
      --ckpt results/phase2-v2/model.pt --n 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livecodec.train2d import cache_volumes, find_series_dirs, is_val  # noqa: E402

ENDPOINT = "https://js2.jetstream-cloud.org:8001"
BUCKET = "livecodec-demo"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/dicom")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--bundles", default="bundles")
    args = ap.parse_args()

    s3 = boto3.client(
        "s3", endpoint_url=ENDPOINT,
        aws_access_key_id=os.environ["S3_ACCESS"],
        aws_secret_access_key=os.environ["S3_SECRET"],
    )

    # pick val series first (honest demo cases), top up with train series
    dirs = find_series_dirs(args.data)
    key_of = {}
    for d in dirs:
        import hashlib

        key_of[d] = hashlib.md5(d.name.encode()).hexdigest()[:16]
    npy = Path("data/npy")
    usable = [d for d in dirs if (npy / f"{key_of[d]}.npy").exists()]
    val = [d for d in usable if is_val(npy / f"{key_of[d]}.npy")]
    train = [d for d in usable if d not in val]
    picks = (val + train)[: args.n]

    scans = []
    for d in picks:
        name = key_of[d]
        out = Path(args.bundles) / name
        if not (out / "meta.json").exists():
            print(f"packing {name} ({d.name[-20:]})", flush=True)
            subprocess.run(
                [sys.executable, "-m", "livecodec.pack", "--series", str(d),
                 "--ckpt", args.ckpt, "--out", str(out)],
                check=True, env={**os.environ, "PYTHONPATH": "src"},
            )
        meta = json.loads((out / "meta.json").read_text())
        for f in ["meta.json", "coarse.zst", "fine.zst", "dc.zst", "slices.bin", "index.json"]:
            p = out / f
            if p.exists():
                s3.upload_file(str(p), BUCKET, f"scans/{name}/{f}")
        print(f"uploaded {name}", flush=True)
        scans.append({
            "id": name,
            "heldout": d in val,
            "shape": meta["shape"],
            "spacing": meta["spacing"],
            "bytes": meta["bytes"],
        })

    s3.put_object(
        Bucket=BUCKET, Key="scans.json", Body=json.dumps(scans, indent=1),
        ContentType="application/json",
    )
    print(f"published {len(scans)} scans to {ENDPOINT}/{BUCKET}/scans.json")


if __name__ == "__main__":
    main()
