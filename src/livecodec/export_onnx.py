"""CLI: export the 3D decoder to ONNX for browser decode (ONNX Runtime Web).

The browser dequantizes FSQ codes itself (trivial arithmetic; constants are
written to decoder.json) and feeds float latents to the ONNX decoder:
  inputs:  zf (1,C,z,h,w), zc_up (1,C,z,h,w)  -- coarse already upsampled 2x
  output:  hu volume chunk (1,1,4z,8h,8w) in normalized [-1,1] units

Usage:
  uv run livecodec-export --ckpt results/phase2-v2/model.pt --out web/demo/decoder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model3d import FSQAutoencoder3D, load_model


class DecoderWrapper(torch.nn.Module):
    """head='full' (512^2 pixels) or 'preview' (1/4 scale, v3 only)."""

    def __init__(self, model: FSQAutoencoder3D, head: str = "full"):
        super().__init__()
        self.decoder = model.decoder
        self.head = head

    def forward(self, zf, zc_up):
        d = self.decoder
        if hasattr(d, "preview"):
            return d.preview(zf, zc_up) if self.head == "preview" else d.full(zf, zc_up)
        return d(zf, zc_up)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="output path prefix (.onnx/.json added)")
    ap.add_argument("--dec-arch", default=None, choices=["3d", "2.5d", "v3"],
                    help="override the checkpoint arch sidecar (rarely needed)")
    ap.add_argument("--head", default="full", choices=["full", "preview"])
    args = ap.parse_args()

    model = load_model(args.ckpt, "cpu",
                       **({"dec_arch": args.dec_arch} if args.dec_arch else {}))
    model.eval()
    wrapper = DecoderWrapper(model, args.head)

    # Fixed 512^2-scan chunk shape: the 2.5D decoder's z->batch reshapes trace to
    # constants under the legacy exporter, so the model is baked to this shape —
    # exactly what the demo feeds it (one 32-slice chunk of a 512^2 scan).
    c = len(model.levels)
    hw = 64 if getattr(model, "fine_stride", 8) == 8 else 128
    zf = torch.zeros(1, c, 8, hw, hw)
    zc_up = torch.zeros(1, c, 8, hw, hw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (zf, zc_up), str(out.with_suffix(".onnx")),
        input_names=["zf", "zc_up"], output_names=["volume"],
        opset_version=17,
        dynamo=False,  # legacy exporter -> single self-contained .onnx (no .data sidecar)
    )
    levels = list(model.levels)
    meta = {
        "levels": levels,
        "offset": [(lv - 1) / 2 for lv in levels],
        "half": [max((lv - 1) / 2, 0.5) for lv in levels],
        "hu_min": -1024, "hu_max": 3071,
        "downsample": {"fine": [4, fs := getattr(model, "fine_stride", 8), fs],
                       "coarse": [8, 2 * fs, 2 * fs]},
        "head": args.head,
        "preview_scale": (2 ** (getattr(model.decoder, "ups", 3) - 1))
                         if args.head == "preview" else 1,
        "note": "dequant: (code - offset[c]) / half[c]; coarse is upsampled 2x "
                "(nearest) before the decoder; output maps [-1,1] -> [hu_min, hu_max]",
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    size = out.with_suffix(".onnx").stat().st_size
    print(f"exported {out.with_suffix('.onnx')} ({size/1e6:.1f} MB) + {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
