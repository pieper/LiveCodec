"""Generate HTJ2K codestreams from a DICOM series for the browser decode bench.

HU values are shifted by +1024 into uint16 (ojph_compress takes 16-bit PGM);
this preserves values exactly for the decode-speed measurement.

Usage:
  uv run python web/make_streams.py --series data/dicom/<dir> --out web/streams
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from livecodec.dicom import load_series

QSTEPS = [0.05, 0.01, 0.002]  # coarse -> fine lossy; plus one reversible


def write_pgm16(path: Path, img: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n65535\n".encode())
        f.write(img.astype(">u2").tobytes())


def compress(pgm: Path, out: Path, args: list[str]) -> int:
    subprocess.run(
        ["ojph_compress", "-i", str(pgm), "-o", str(out), *args],
        check=True,
        capture_output=True,
    )
    return out.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--series", required=True)
    ap.add_argument("--out", default="web/streams")
    ap.add_argument("--max-slices", type=int, default=16)
    args = ap.parse_args()

    volume, info = load_series(args.series)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zs = np.linspace(0, volume.shape[0] - 1, num=min(args.max_slices, volume.shape[0]), dtype=int)

    variants = [("reversible", ["-reversible", "true"])] + [
        (f"qstep{q}", ["-qstep", str(q)]) for q in QSTEPS
    ]
    manifest = {"shape": list(volume.shape), "series": info["series_uid"], "slices": []}
    with tempfile.TemporaryDirectory() as tmp:
        for z in zs:
            img = np.clip(volume[z].astype(np.int32) + 1024, 0, 65535).astype(np.uint16)
            pgm = Path(tmp) / f"s{z}.pgm"
            write_pgm16(pgm, img)
            entry = {"z": int(z), "files": {}}
            for name, extra in variants:
                j2c = out_dir / f"slice{z:04d}_{name}.j2c"
                size = compress(pgm, j2c, extra)
                entry["files"][name] = {"path": j2c.name, "bytes": size}
            manifest["slices"].append(entry)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n = len(manifest["slices"])
    print(f"wrote {n} slices x {len(variants)} variants to {out_dir}")
    for name, _ in variants:
        total = sum(s["files"][name]["bytes"] for s in manifest["slices"])
        raw = n * volume.shape[1] * volume.shape[2] * 2
        print(f"  {name}: {total / 1e6:.2f} MB for {n} slices ({raw / total:.1f}x)")


if __name__ == "__main__":
    main()
