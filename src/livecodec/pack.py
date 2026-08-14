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
from .model3d import FSQAutoencoder3D, load_model
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


def split_tileparts(data: bytes) -> list[bytes]:
    """Split a single-tile HT codestream (encoded with -tileparts R) into
    [main_header + tilepart0, tilepart1, ..., tilepartN + EOC]. Each prefix of
    this sequence truncated at a segment boundary is decodable at a reduced
    resolution (decodeSubResolution in the browser)."""
    sots = []
    i = 0
    while True:
        i = data.find(b"\xff\x90", i)
        if i < 0:
            break
        psot = int.from_bytes(data[i + 6 : i + 10], "big")
        sots.append((i, psot))
        i += max(psot, 12)
    if not sots:
        return [data]
    segs = []
    for n, (off, psot) in enumerate(sots):
        start = 0 if n == 0 else off
        end = off + psot if n < len(sots) - 1 else len(data)  # last keeps EOC
        segs.append(data[start:end])
    return segs


def htj2k_encode(
    vol: np.ndarray, out_dir: Path, name: str = "slices",
    value_offset: int = 1024, reversible: bool = True, res_progressive: bool = False,
) -> list[dict] | dict:
    """Single concatenated <name>.bin of per-slice HT codestreams + index (the
    JS2 RGW throttles per-request, so the demo streams one object and splits it
    client-side). Values are stored as uint16 = value + value_offset.

    res_progressive: encode with -tileparts R and lay the file out RESOLUTION-
    MAJOR — round 0 holds every slice's lowest-resolution tile-part, so a whole-
    volume preview decodes from the first few percent of the stream, sharpening
    round by round to lossless. Index: {"layout": "res-progressive",
    "rounds": R, "slices": [{"z", "parts": [[offset, bytes], ...]}]}."""
    args = ["-reversible", "true"] if reversible else ["-qstep", "0.002"]
    if res_progressive:
        args += ["-tileparts", "R"]
    per_slice: list[list[bytes]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for z in range(vol.shape[0]):
            img = np.clip(vol[z].astype(np.int32) + value_offset, 0, 65535).astype(">u2")
            pgm = Path(tmp) / "s.pgm"
            with open(pgm, "wb") as f:
                f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n65535\n".encode())
                f.write(img.tobytes())
            j2c = Path(tmp) / "s.j2c"
            subprocess.run(
                ["ojph_compress", "-i", str(pgm), "-o", str(j2c), *args],
                check=True, capture_output=True,
            )
            data = j2c.read_bytes()
            per_slice.append(split_tileparts(data) if res_progressive else [data])

    if not res_progressive:
        index, offset = [], 0
        with open(out_dir / f"{name}.bin", "wb") as bin_out:
            for z, segs in enumerate(per_slice):
                bin_out.write(segs[0])
                index.append({"z": z, "offset": offset, "bytes": len(segs[0])})
                offset += len(segs[0])
        return index

    rounds = max(len(s) for s in per_slice)
    slices = [{"z": z, "parts": []} for z in range(len(per_slice))]
    offset = 0
    with open(out_dir / f"{name}.bin", "wb") as bin_out:
        for r in range(rounds):
            for z, segs in enumerate(per_slice):
                if r >= len(segs):
                    continue
                bin_out.write(segs[r])
                slices[z]["parts"].append([offset, len(segs[r])])
                offset += len(segs[r])
    return {"layout": "res-progressive", "rounds": rounds, "slices": slices}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-htj2k", action="store_true")
    ap.add_argument("--dec-arch", default=None, choices=["3d", "2.5d", "v3"],
                    help="override the checkpoint arch sidecar (rarely needed)")
    args = ap.parse_args()

    device = pick_device()
    vol, info = load_series(args.series)
    vol = vol.clip(-1024, 3071)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.ckpt, device,
                       **({"dec_arch": args.dec_arch} if args.dec_arch else {}))
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

    # residual tier -> bit-exact: residual = original - DC-corrected fine recon,
    # stored as reversible HT slices (uint16 = residual + 4096). The browser adds
    # decoded residuals to the refined volume; equality with the original is then
    # exact by construction (integer arithmetic end to end).
    residual = vol.astype(np.int32) - enc["recon"].astype(np.int32)
    assert np.abs(residual).max() < 4096, "residual out of range"
    # res-progressive: real detail streams in continuously right after the fine
    # tier (low-res residual rounds first), so the neural curve refines smoothly
    # to lossless instead of plateauing until a monolithic residual arrives.
    ridx = htj2k_encode(residual, out, name="residual", value_offset=4096,
                        reversible=True, res_progressive=True)
    (out / "residual-index.json").write_text(json.dumps(ridx))
    meta["bytes"]["residual"] = sum(p[1] for s in ridx["slices"] for p in s["parts"])

    if not args.skip_htj2k:
        # the lossless HTJ2K arm, resolution-progressive: round 0 = every slice's
        # lowest-resolution tile-part -> whole-volume preview from the first few
        # percent of the stream, refining round by round to bit-exact.
        index = htj2k_encode(vol, out, res_progressive=True)
        (out / "index.json").write_text(json.dumps(index))
        meta["bytes"]["htj2k"] = sum(
            p[1] for s in index["slices"] for p in s["parts"]
        )

    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    n = meta["bytes"]
    print(
        f"{out.name}: raw {n['raw']/1e6:.0f} MB | neural coarse {n['coarse']/1e3:.0f} KB"
        f" + fine {n['fine']/1e3:.0f} KB + dc {n['dc']/1e3:.1f} KB"
        + (f" | htj2k {n['htj2k']/1e6:.1f} MB" if "htj2k" in n else "")
    )


if __name__ == "__main__":
    main()
