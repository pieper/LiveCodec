"""CLI: pack a CT series into demo bundles for the download-speed comparison.

Neural bundle (per scan):
  meta.json            shape/spacing/levels/sizes
  coarse.gz            gzip'd coarse FSQ codes   (the ~instant preview)
  fine.gz              gzip'd fine FSQ codes     (streamed refinement)
  dc.gz                per-block mean sideband for the fine tier
  (gzip so the browser's native DecompressionStream can decode)

HTJ2K bundle (per scan):
  index.json           ordered slice files + byte sizes
  slices/NNNN.j2c      one HT codestream per slice (ojph_compress)

Usage:
  uv run livecodec-pack --series <dir> --ckpt results/phase2-v2/model.pt --out bundles/<name> [--skip-htj2k]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import gzip

from .dicom import load_series
from .model2d import hu_to_unit, unit_to_hu
from .model3d import FSQAutoencoder3D
from .train2d import pick_device

CHUNK_Z = 32  # encoder z-chunking; latent z = CHUNK_Z / 4


def _zc(data: bytes) -> bytes:
    return gzip.compress(data, 9)


def neural_encode(model: FSQAutoencoder3D, vol: np.ndarray, device) -> dict:
    """Encode a full volume in z-chunks; returns dict of payloads + recon."""
    zpad = (-vol.shape[0]) % CHUNK_Z
    padded = np.pad(vol, ((0, zpad), (0, 0), (0, 0)), mode="edge")
    cfs, ccs, recons = [], [], []
    for z in range(0, padded.shape[0], CHUNK_Z):
        chunk = padded[z : z + CHUNK_Z].astype(np.float32)
        x = hu_to_unit(torch.from_numpy(chunk)[None, None]).to(device)
        cf, cc = model.compress(x)
        recons.append(
            unit_to_hu(model.decompress(cf, cc)).squeeze(0).squeeze(0).cpu().numpy()
        )
        cfs.append(cf.cpu().numpy())
        ccs.append(cc.cpu().numpy())
    recon = np.concatenate(recons)[: vol.shape[0]].astype(np.int16)
    recon, dc_bytes_arr = _dc_payload(vol, recon)
    return {
        "coarse": _zc(np.concatenate(ccs).tobytes()),
        "fine": _zc(np.concatenate(cfs).tobytes()),
        "dc": dc_bytes_arr,
        "recon": recon,
        "latent_shapes": {
            "fine": [list(a.shape) for a in cfs][0],
            "coarse": [list(a.shape) for a in ccs][0],
            "chunks": len(cfs),
        },
    }


def _dc_payload(vol: np.ndarray, recon: np.ndarray) -> tuple[np.ndarray, bytes]:
    bz = max(1, min(64, vol.shape[0]))
    zb, yb, xb = (max(1, s // b) for s, b in zip(vol.shape, (bz, 64, 64)))
    err = (recon.astype(np.float32) - vol.astype(np.float32))[: zb * bz, : yb * 64, : xb * 64]
    means = err.reshape(zb, bz, yb, 64, xb, 64).mean(axis=(1, 3, 5))
    q = np.clip(np.round(means / 4.0), -128, 127).astype(np.int8)
    corr = torch.nn.functional.interpolate(
        torch.from_numpy(q.astype(np.float32) * 4.0)[None, None],
        size=vol.shape, mode="trilinear", align_corners=False,
    ).squeeze().numpy()
    fixed = np.clip(recon.astype(np.float32) - corr, -1024, 3071).astype(np.int16)
    return fixed, _zc(q.tobytes())


def htj2k_encode(vol: np.ndarray, out_dir: Path) -> list[dict]:
    """Single concatenated slices.bin + per-slice byte offsets: the JS2 RGW
    throttles per-request, so the demo streams one object and splits it
    client-side (offsets let a reader decode slices as bytes arrive)."""
    index, offset = [], 0
    with tempfile.TemporaryDirectory() as tmp, open(out_dir / "slices.bin", "wb") as bin_out:
        for z in range(vol.shape[0]):
            img = np.clip(vol[z].astype(np.int32) + 1024, 0, 65535).astype(">u2")
            pgm = Path(tmp) / "s.pgm"
            with open(pgm, "wb") as f:
                f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n65535\n".encode())
                f.write(img.tobytes())
            j2c = Path(tmp) / "s.j2c"
            subprocess.run(
                ["ojph_compress", "-i", str(pgm), "-o", str(j2c), "-qstep", "0.002"],
                check=True, capture_output=True,
            )
            data = j2c.read_bytes()
            bin_out.write(data)
            index.append({"z": z, "offset": offset, "bytes": len(data)})
            offset += len(data)
    return index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-htj2k", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    vol, info = load_series(args.series)
    vol = vol.clip(-1024, 3071)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = FSQAutoencoder3D().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    model.eval()

    enc = neural_encode(model, vol, device)
    (out / "coarse.gz").write_bytes(enc["coarse"])
    (out / "fine.gz").write_bytes(enc["fine"])
    (out / "dc.gz").write_bytes(enc["dc"])

    meta = {
        "series_uid": info["series_uid"],
        "shape": list(vol.shape),
        "spacing": list(info["spacing"]),
        "levels": model.levels,
        "chunk_z": CHUNK_Z,
        "latent": enc["latent_shapes"],
        "bytes": {
            "raw": vol.nbytes,
            "coarse": len(enc["coarse"]),
            "fine": len(enc["fine"]),
            "dc": len(enc["dc"]),
        },
    }

    if not args.skip_htj2k:
        index = htj2k_encode(vol, out)
        (out / "index.json").write_text(json.dumps(index))
        meta["bytes"]["htj2k"] = sum(e["bytes"] for e in index)

    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    n = meta["bytes"]
    print(
        f"{out.name}: raw {n['raw']/1e6:.0f} MB | neural coarse {n['coarse']/1e3:.0f} KB"
        f" + fine {n['fine']/1e3:.0f} KB + dc {n['dc']/1e3:.1f} KB"
        + (f" | htj2k {n['htj2k']/1e6:.1f} MB" if "htj2k" in n else "")
    )


if __name__ == "__main__":
    main()
