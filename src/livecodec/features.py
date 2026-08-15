"""Turn a CT series into a compact feature vector from the codec's latents.

The point: the encoder is a learned CT representation, and its latents are
~1000x smaller than the volume. Extract once on a GPU, then iterate on
downstream models cheaply anywhere — the feature file for a whole cohort is
megabytes, not gigabytes.

Features are built from the DEQUANTIZED latent codes (exactly what the decoder
consumes, i.e. what the transmitted bitstream carries), average-pooled to a
fixed grid so volumes of different z-extent give equal-length vectors:

  fine   (C, gz, gy, gx)   the detail scale
  coarse (C, gz, gy, gx)   the preview scale
  plus per-channel mean/std over the whole volume

Usage:
  uv run livecodec-features --manifest data/lnq.csv --root data/lnq \
      --ckpt results/big-400k/model.pt --out features/lnq.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .dicom import load_series
from .model2d import hu_to_unit
from .model3d import load_model
from .pack import CHUNK_Z
from .train2d import pick_device


@torch.no_grad()
def latent_features(
    model, vol: np.ndarray, device, grid: tuple[int, int, int] = (4, 8, 8)
) -> dict[str, np.ndarray]:
    """Encode a volume and pool its latents to fixed-size grids."""
    zpad = (-vol.shape[0]) % CHUNK_Z
    padded = np.pad(vol, ((0, zpad), (0, 0), (0, 0)), mode="edge")
    fine, coarse = [], []
    for z in range(0, padded.shape[0], CHUNK_Z):
        x = hu_to_unit(torch.from_numpy(padded[z : z + CHUNK_Z].astype(np.float32))[None, None])
        cf, cc = model.compress(x.to(device))
        # dequantize: the values the decoder actually sees
        fine.append(model.fsq.dequantize(cf).squeeze(0).cpu())
        coarse.append(model.fsq.dequantize(cc).squeeze(0).cpu())
    out = {}
    for name, parts in (("fine", fine), ("coarse", coarse)):
        t = torch.cat(parts, dim=1)                      # (C, Z, H, W) over the whole volume
        pooled = torch.nn.functional.adaptive_avg_pool3d(t[None], grid).squeeze(0)
        out[name] = pooled.numpy().astype(np.float32)
        out[f"{name}_mean"] = t.mean(dim=(1, 2, 3)).numpy().astype(np.float32)
        out[f"{name}_std"] = t.std(dim=(1, 2, 3)).numpy().astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="directory holding downloaded series")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid", default="4,8,8")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    grid = tuple(int(v) for v in args.grid.split(","))
    device = pick_device()
    model = load_model(args.ckpt, device)
    model.eval()

    dirs = sorted({p.parent for p in Path(args.root).rglob("*.dcm")})
    if args.limit:
        dirs = dirs[: args.limit]
    print(f"device={device.type} series={len(dirs)} grid={grid}", flush=True)

    rows, meta = [], []
    for i, d in enumerate(dirs):
        try:
            vol, info = load_series(d)
        except Exception as e:                      # a few IDC series are unreadable
            print(f"skip {d.name[-16:]}: {type(e).__name__}", flush=True)
            continue
        vol = vol.clip(-1024, 3071)
        f = latent_features(model, vol, device, grid)
        rows.append(np.concatenate([f[k].ravel() for k in sorted(f)]))
        meta.append({
            "dir": str(d), "series_uid": info["series_uid"], "shape": list(vol.shape),
            "spacing": list(info["spacing"]),
            "key": hashlib.md5(d.name.encode()).hexdigest()[:16],
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(dirs)}", flush=True)

    X = np.stack(rows)
    names = []
    probe = latent_features(model, np.zeros((CHUNK_Z, 512, 512), np.int16), device, grid)
    for k in sorted(probe):
        names += [f"{k}[{j}]" for j in range(probe[k].size)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, feature_names=np.array(names),
                        meta=np.array([json.dumps(m) for m in meta]))
    print(f"wrote {out}: X={X.shape} ({X.nbytes/1e6:.1f} MB dense, compressed on disk)")


if __name__ == "__main__":
    main()
