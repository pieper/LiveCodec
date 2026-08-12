"""Publish one training checkpoint as a demo "encoding version": neural bundles
for the fixed out-of-distribution scan set + that checkpoint's decoder, under
versions/<tag>/ in the bucket. The HTJ2K arm is version-independent and lives
under ood/<id>/ (uploaded once with --first).

Bucket layout:
  versions.json                        [{tag, steps, trained_vols, params, note}]
  ood-scans.json                       the OOD scan list (shape/spacing/htj2k bytes)
  ood/<id>/slices.bin|index.json       shared lossless res-progressive HTJ2K
  versions/<tag>/model/...             decoder25.graph.json/.weights.bin + decoder.json
  versions/<tag>/<id>/...              coarse.gz fine.gz dc.gz residual.* meta.json

Run on the GPU instance (S3_ACCESS/S3_SECRET in env):
  uv run --no-sync --with boto3 python scripts/publish_version.py \
      --tag big-050k --steps 50000 --ckpt results/big-50000/model.pt \
      --ood-dir data/ood [--first] [--note "..."]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import boto3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ENDPOINT = "https://js2.jetstream-cloud.org:8001"
BUCKET = "livecodec-demo"
PY = sys.executable


def series_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("*.dcm")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ood-dir", default="data/ood")
    ap.add_argument("--bundles", default="bundles-versions")
    ap.add_argument("--first", action="store_true", help="also publish shared HTJ2K + scan list")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    s3 = boto3.client(
        "s3", endpoint_url=ENDPOINT,
        aws_access_key_id=os.environ["S3_ACCESS"],
        aws_secret_access_key=os.environ["S3_SECRET"],
    )
    env = {**os.environ, "PYTHONPATH": "src"}

    from livecodec.model3d import load_model  # noqa: E402

    model = load_model(args.ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    del model

    dirs = series_dirs(Path(args.ood_dir))
    if not dirs:
        raise SystemExit(f"no OOD series under {args.ood_dir}")

    # decoder export for THIS version
    vdir = Path(args.bundles) / args.tag
    mdir = vdir / "model"
    mdir.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, "-m", "livecodec.export_onnx", "--ckpt", args.ckpt,
                    "--out", str(mdir / "decoder25"), "--dec-arch", "2.5d"], check=True, env=env)
    subprocess.run([PY, "scripts/dump_graph25.py", str(mdir / "decoder25.onnx")], check=True, env=env)
    # dump_graph25 writes next to the onnx; ensure names in mdir
    for f in ["decoder25.graph.json", "decoder25.weights.bin"]:
        src = Path("web/demo") / f
        if not (mdir / f).exists() and src.exists():
            shutil.copy(src, mdir / f)
    (mdir / "decoder.json").write_bytes((mdir / "decoder25.json").read_bytes())

    ood_entries = []
    for d in dirs:
        import hashlib

        sid = hashlib.md5(d.name.encode()).hexdigest()[:16]
        sdir = vdir / sid
        if not (sdir / "meta.json").exists():
            print(f"[{args.tag}] packing {sid} ({d.name[-20:]})", flush=True)
            subprocess.run([PY, "-m", "livecodec.pack", "--series", str(d),
                            "--ckpt", args.ckpt, "--out", str(sdir), "--skip-htj2k"],
                           check=True, env=env)
        meta = json.loads((sdir / "meta.json").read_text())
        for f in ["meta.json", "coarse.gz", "fine.gz", "dc.gz", "residual.bin", "residual-index.json"]:
            if (sdir / f).exists():
                s3.upload_file(str(sdir / f), BUCKET, f"versions/{args.tag}/{sid}/{f}")

        if args.first:
            hdir = Path(args.bundles) / "_htj2k" / sid
            if not (hdir / "index.json").exists():
                print(f"[shared] HTJ2K for {sid}", flush=True)
                subprocess.run([PY, "-m", "livecodec.pack", "--series", str(d),
                                "--ckpt", args.ckpt, "--out", str(hdir)], check=True, env=env)
            for f in ["slices.bin", "index.json"]:
                s3.upload_file(str(hdir / f), BUCKET, f"ood/{sid}/{f}")
            hmeta = json.loads((hdir / "meta.json").read_text())
            ood_entries.append({
                "id": sid, "source": d.name[-40:], "shape": hmeta["shape"],
                "spacing": hmeta["spacing"],
                "bytes": {"raw": hmeta["bytes"]["raw"], "htj2k": hmeta["bytes"]["htj2k"]},
            })
        print(f"[{args.tag}] uploaded {sid}", flush=True)

    if args.first and ood_entries:
        s3.put_object(Bucket=BUCKET, Key="ood-scans.json",
                      Body=json.dumps(ood_entries, indent=1), ContentType="application/json")

    for f in ["decoder25.graph.json", "decoder25.weights.bin", "decoder.json"]:
        s3.upload_file(str(mdir / f), BUCKET, f"versions/{args.tag}/model/{f}")

    # append to versions.json (read-modify-write; single writer)
    try:
        versions = json.loads(s3.get_object(Bucket=BUCKET, Key="versions.json")["Body"].read())
    except Exception:
        versions = []
    versions = [v for v in versions if v["tag"] != args.tag]
    versions.append({"tag": args.tag, "steps": args.steps,
                     "params": f"{n_params/1e6:.1f}M", "note": args.note})
    versions.sort(key=lambda v: v["steps"])
    s3.put_object(Bucket=BUCKET, Key="versions.json",
                  Body=json.dumps(versions, indent=1), ContentType="application/json")
    print(f"published version {args.tag} ({len(dirs)} scans); versions.json now {len(versions)} entries")


if __name__ == "__main__":
    main()
