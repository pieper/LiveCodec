"""CLI: dump the decoder's weights as a flat f32 blob + JSON manifest for the
hand-written WebGPU (WGSL) decoder — no ONNX Runtime in the loop.

decoder.bin       all tensors concatenated, little-endian f32, 16-byte aligned
decoder-arch.json manifest: ordered tensors {name, shape, offset_floats} plus
                  the dequant constants and architecture hyperparameters the
                  browser needs to schedule the pipeline.

Usage:
  uv run livecodec-export-weights --ckpt results/phase2-v3/model.pt \
      --dec-arch 2.5d --out web/demo/decoder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model3d import FSQAutoencoder3D, load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dec-arch", default="2.5d", choices=["3d", "2.5d"])
    ap.add_argument("--out", required=True, help="output path prefix")
    args = ap.parse_args()

    model = load_model(args.ckpt, "cpu", dec_arch=args.dec_arch)
    model.eval()

    tensors, blob, offset = [], [], 0
    for name, t in model.decoder.state_dict().items():
        a = t.detach().numpy().astype("<f4")
        pad = (-a.size) % 4  # keep every tensor 16-byte aligned
        tensors.append({"name": name, "shape": list(a.shape), "offset_floats": offset})
        blob.append(a.ravel())
        if pad:
            blob.append(np.zeros(pad, "<f4"))
        offset += a.size + pad

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate(blob).tofile(out.parent / (out.name + ".bin"))

    levels = list(model.levels)
    arch = {
        "dec_arch": args.dec_arch,
        "width": model.decoder.mix[0].out_channels if args.dec_arch == "2.5d" else None,
        "levels": levels,
        "offset": [(lv - 1) / 2 for lv in levels],
        "half": [max((lv - 1) / 2, 0.5) for lv in levels],
        "hu_min": -1024, "hu_max": 3071,
        "groupnorm_groups": 8,
        "tensors": tensors,
        "total_floats": offset,
    }
    (out.parent / (out.name + "-arch.json")).write_text(json.dumps(arch, indent=1))
    print(f"wrote {out}.bin ({offset * 4 / 1e6:.1f} MB) + {out}-arch.json ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
