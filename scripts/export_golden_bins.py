"""Emit golden-smoke.npz tensors as raw little-endian f32 .bin files (JS-friendly).

Usage: python scripts/export_golden_bins.py [web/demo/golden-smoke.npz] [outdir]
Writes golden-zf.bin, golden-zc_up.bin, golden-expected.bin next to the npz.
"""
import os
import sys

import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "web/demo/golden-smoke.npz"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(SRC)
d = np.load(SRC)
for k in d.files:
    arr = d[k].astype("<f4").ravel()
    path = os.path.join(OUT, f"golden-{k}.bin")
    arr.tofile(path)
    print(path, arr.size, "f32")
