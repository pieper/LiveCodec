"""Capture fixed-seed random-latent decodes every time the checkpoint updates,
so the evolution can be played back as a movie.

Runs ALONGSIDE training (it only reads model.pt), so no restart is needed. The
trainer rewrites model.pt every --dash-every steps; we snapshot each new one,
decode the same three random latents, and write a frame per sample.

  uv run --no-sync python scripts/frame_grabber.py --run results/prior-112M
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WINDOW = (-160.0, 240.0)   # soft tissue W400 L40


def png_bytes(img: np.ndarray, size: int = 256) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image

    lo, hi = WINDOW
    g = np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)
    if g.shape[0] != size or g.shape[1] != size:      # cheap nearest resize
        yi = (np.linspace(0, g.shape[0] - 1, size)).astype(int)
        xi = (np.linspace(0, g.shape[1] - 1, size)).astype(int)
        g = g[yi][:, xi]
    buf = io.BytesIO()
    matplotlib.image.imsave(buf, g, cmap="gray", vmin=0, vmax=1, format="png")
    return buf.getvalue()


def current_step(log: Path) -> int:
    try:
        txt = log.read_text()[-4000:]
        hits = re.findall(r"step (\d+)/", txt)
        return int(hits[-1]) if hits else -1
    except Exception:
        return -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/prior-112M")
    ap.add_argument("--log", default="logs/train-prior.log")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--modes", default="random,coarse,interp",
                    help="random=uniform codes (harshest, off-manifold); "
                         "coarse=random COARSE codes with fine zeroed (anatomy lives "
                         "in the coarse scale, so this probes the prior at its own "
                         "scale); interp=midpoint between two REAL latents (on-manifold)")
    args = ap.parse_args()

    from livecodec.model2d import hu_to_unit, unit_to_hu
    from livecodec.model3d import load_model
    from livecodec.train2d import is_val, pick_device

    run = Path(args.run)
    frames = run / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    ckpt = run / "model.pt"
    device = pick_device()
    seen_mtime = 0.0
    index_path = frames / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []

    modes = [m for m in args.modes.split(",") if m]
    # the encoder is FROZEN, so real latents never change: encode once and reuse
    real = None
    if "interp" in modes:
        vps = [p for p in sorted(Path("data/npy").glob("*.npy")) if is_val(p)][:2]
        if len(vps) == 2:
            real = []
            import numpy as _np
            for vp in vps:
                v = _np.load(vp, mmap_mode="r")
                z0 = (v.shape[0] - 32) // 2
                a = _np.ascontiguousarray(v[z0:z0 + 32, :512, :512]).astype("float32")
                real.append(hu_to_unit(torch.from_numpy(a)[None, None]).to(device))
        else:
            modes = [m for m in modes if m != "interp"]
    print(f"watching {ckpt} (device={device.type}); modes={modes}; frames -> {frames}", flush=True)
    while True:
        try:
            mt = ckpt.stat().st_mtime
        except FileNotFoundError:
            time.sleep(args.poll)
            continue
        if mt <= seen_mtime:
            time.sleep(args.poll)
            continue
        time.sleep(5)                                  # let the write settle
        snap = frames / "_snap.pt"
        try:
            shutil.copy2(ckpt, snap)
            shutil.copy2(ckpt.with_suffix(".json"), snap.with_suffix(".json"))
            model = load_model(snap, device)
            model.eval()
        except Exception as e:                          # torn write -> retry next poll
            print(f"  reload failed ({type(e).__name__}); retrying", flush=True)
            time.sleep(args.poll)
            continue
        seen_mtime = mt
        step = current_step(Path(args.log))
        L = list(model.levels)
        entry = {"step": step, "modes": {}}
        with torch.no_grad():
            rl = None
            if real is not None:
                rl = [model.compress(r) for r in real]
            for mode in modes:
                files, stats = [], []
                for si in range(args.samples):
                    g = torch.Generator().manual_seed(1000 + si)
                    if mode == "random":
                        cf = torch.stack([torch.randint(0, int(L[c]), (8, 64, 64), generator=g)
                                          for c in range(len(L))])[None].to(torch.uint8)
                        cc = cf[:, :, ::2, ::2, ::2].contiguous()
                        out = model.decompress(cf.to(device), cc.to(device))
                    elif mode == "coarse":
                        # anatomy lives in the coarse scale: draw it, leave fine empty
                        cc = torch.stack([torch.randint(0, int(L[c]), (4, 32, 32), generator=g)
                                          for c in range(len(L))])[None].to(torch.uint8)
                        cf = torch.zeros((1, len(L), 8, 64, 64), dtype=torch.uint8)
                        for c in range(len(L)):
                            cf[0, c] = int(L[c]) // 2          # FSQ centre = zero latent
                        out = model.decompress(cf.to(device), cc.to(device), coarse_only=True)
                    else:                                       # interp between real scans
                        t = (si + 1) / (args.samples + 1)
                        zf = (1 - t) * model.fsq.dequantize(rl[0][0]) + t * model.fsq.dequantize(rl[1][0])
                        zc = (1 - t) * model.fsq.dequantize(rl[0][1]) + t * model.fsq.dequantize(rl[1][1])
                        second = zc if type(model.decoder).__name__ == "DecoderPrior2" else \
                            torch.nn.functional.interpolate(zc, scale_factor=2, mode="nearest")
                        o = model.decoder(zf, second)
                        out = o[0] if isinstance(o, tuple) else o
                    a = unit_to_hu(out).squeeze().cpu().numpy().astype(np.int16)
                    name = f"{mode}_s{si}_step{step:07d}.png"
                    (frames / name).write_bytes(png_bytes(a[a.shape[0] // 2], args.size))
                    files.append(name)
                    stats.append({
                        "soft_pct": round(100 * float(np.mean((a > -200) & (a < 300))), 1),
                        "air_pct": round(100 * float(np.mean(a < -900)), 1),
                        "mean": int(a.mean()), "sd": int(a.std()),
                    })
                entry["modes"][mode] = {"files": files, "stats": stats}
        index = [e for e in index if e["step"] != step] + [entry]
        index.sort(key=lambda e: e["step"])
        index_path.write_text(json.dumps(index, indent=1))
        print(f"captured step {step}: " + " | ".join(
            f"{m}: " + " ".join(f"{d['soft_pct']}%" for d in entry["modes"][m]["stats"])
            for m in entry["modes"]), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
