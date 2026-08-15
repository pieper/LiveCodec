"""Extract LiveCodec latent features for the whole LNQ cohort, disk-safely.

The cohort is ~35 GB of DICOM but the features are ~5 MB, so we never hold
more than one batch on disk: download a batch, encode it, delete it, repeat.
Progress is checkpointed after every batch, so an interruption resumes.

  uv run --no-sync python scripts/extract_lnq.py --ckpt results/big-400k-encoder.pt \
      --out features/lnq.npz --batch 40
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

COLLECTION = "mediastinal_lymph_node_seg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="features/lnq.npz")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--work", default="data/lnq-batch")
    ap.add_argument("--grid", default="4,8,8")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch  # noqa: F401  (imported for side effects / device init)
    from idc_index import IDCClient

    from livecodec.dicom import load_series
    from livecodec.features import latent_features
    from livecodec.model3d import load_model
    from livecodec.train2d import pick_device

    grid = tuple(int(v) for v in args.grid.split(","))
    device = pick_device()
    model = load_model(args.ckpt, device)
    model.eval()

    client = IDCClient.client() if hasattr(IDCClient, "client") else IDCClient()
    idx = client.index
    ct = idx[(idx["collection_id"] == COLLECTION) & (idx["Modality"] == "CT")]
    uids = sorted(ct["SeriesInstanceUID"].unique())
    if args.limit:
        uids = uids[: args.limit]
    print(f"device={device.type} cohort={len(uids)} series, batch={args.batch}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, meta = [], []
    if out.exists():                                    # resume
        prev = np.load(out, allow_pickle=True)
        rows = list(prev["X"])
        meta = [json.loads(m) for m in prev["meta"]]
        done = {m["series_uid"] for m in meta}
        uids = [u for u in uids if u not in done]
        print(f"resuming: {len(meta)} done, {len(uids)} remaining", flush=True)

    work = Path(args.work)
    for b0 in range(0, len(uids), args.batch):
        batch = uids[b0 : b0 + args.batch]
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True, exist_ok=True)
        client.download_from_selection(seriesInstanceUID=batch, downloadDir=str(work))
        for d in sorted({p.parent for p in work.rglob("*.dcm")}):
            try:
                vol, info = load_series(d)
            except Exception as e:
                print(f"  skip {d.name[-14:]}: {type(e).__name__}", flush=True)
                continue
            f = latent_features(model, vol.clip(-1024, 3071), device, grid)
            rows.append(np.concatenate([f[k].ravel() for k in sorted(f)]))
            meta.append({
                "series_uid": info["series_uid"], "shape": list(vol.shape),
                "spacing": list(info["spacing"]),
                "key": hashlib.md5(d.name.encode()).hexdigest()[:16],
            })
        np.savez_compressed(out, X=np.stack(rows),
                            meta=np.array([json.dumps(m) for m in meta]))
        print(f"batch {b0 // args.batch + 1}: {len(meta)}/{len(meta) + len(uids) - b0 - len(batch)} "
              f"encoded, saved {out}", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    print(f"done: X={np.stack(rows).shape} -> {out}")


if __name__ == "__main__":
    main()
