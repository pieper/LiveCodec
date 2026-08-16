"""Fairer probes of what the decoder's prior actually holds.

Uniform-random FSQ codes are a MAXIMALLY atypical latent: real latents are
spatially smooth and correlated, so decoding white-noise codes asks the model
about a region it never sees. These probes stay closer to the real manifold:

  interp  midpoints between two REAL scans' latents. If the decoder holds a
          usable prior, midpoints should look like plausible CTs.
  shuffle a real latent with its spatial blocks permuted — right marginal
          statistics, wrong arrangement.
  smooth  spatially smoothed random codes (correct correlation length).

  uv run --no-sync python scripts/latent_probe_manifold.py --ckpt <ckpt> --out probe/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def save(img: np.ndarray, path: Path, window=(-160.0, 240.0)) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image
    lo, hi = window
    matplotlib.image.imsave(path, np.clip((img - lo) / (hi - lo), 0, 1),
                            cmap="gray", vmin=0, vmax=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="data/npy")
    ap.add_argument("--out", default="results/prior-112M/manifold")
    args = ap.parse_args()

    from livecodec.model2d import hu_to_unit, unit_to_hu
    from livecodec.model3d import load_model
    from livecodec.train2d import is_val, pick_device

    device = pick_device()
    model = load_model(args.ckpt, device)
    model.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vols = [p for p in sorted(Path(args.cache).glob("*.npy")) if is_val(p)][:2]
    arrs = []
    for p in vols:
        v = np.load(p, mmap_mode="r")
        z0 = (v.shape[0] - 32) // 2
        arrs.append(np.ascontiguousarray(v[z0:z0 + 32]).astype(np.float32))
    stats = {}

    with torch.no_grad():
        codes = []
        for i, a in enumerate(arrs):
            x = hu_to_unit(torch.from_numpy(a)[None, None]).to(device)
            cf, cc = model.compress(x)
            codes.append((cf, cc))
            rec = unit_to_hu(model.decompress(cf, cc)).squeeze().cpu().numpy()
            save(rec[16], out / f"real{i}_recon.png")
            save(a[16], out / f"real{i}_original.png")

        # dequantized latents, so we can interpolate continuously
        zf = [model.fsq.dequantize(c[0]) for c in codes]
        zc = [model.fsq.dequantize(c[1]) for c in codes]
        for t in (0.25, 0.5, 0.75):
            f = (1 - t) * zf[0] + t * zf[1]
            c = (1 - t) * zc[0] + t * zc[1]
            cu = torch.nn.functional.interpolate(c, scale_factor=2, mode="nearest")
            o = model.decoder(f, cu)
            o = o[0] if isinstance(o, tuple) else o
            a = unit_to_hu(o).squeeze().cpu().numpy()
            save(a[16], out / f"interp_{int(t*100):02d}.png")
            stats[f"interp{t}"] = (float(np.mean((a > -200) & (a < 300)) * 100), float(a.std()))

        # spatially shuffled real latent: right marginals, wrong arrangement
        g = torch.Generator(device="cpu").manual_seed(0)
        f0 = zf[0].cpu()
        flat = f0.reshape(f0.shape[1], -1)
        perm = torch.randperm(flat.shape[1], generator=g)
        sh = flat[:, perm].reshape(f0.shape).to(device)
        shc = torch.nn.functional.interpolate(zc[0], scale_factor=2, mode="nearest")
        o = model.decoder(sh, shc)
        a = unit_to_hu(o[0] if isinstance(o, tuple) else o).squeeze().cpu().numpy()
        save(a[16], out / "shuffled.png")
        stats["shuffled"] = (float(np.mean((a > -200) & (a < 300)) * 100), float(a.std()))

        # smoothed random: correct correlation length, random content
        rnd = torch.randn(zf[0].shape, generator=g).to(device)
        rnd = torch.nn.functional.avg_pool3d(rnd, 3, stride=1, padding=1) * 3.0
        rndc = torch.nn.functional.interpolate(
            torch.nn.functional.avg_pool3d(torch.randn(zc[0].shape, generator=g).to(device),
                                           3, stride=1, padding=1) * 3.0,
            scale_factor=2, mode="nearest")
        o = model.decoder(rnd.clamp(-1, 1), rndc.clamp(-1, 1))
        a = unit_to_hu(o[0] if isinstance(o, tuple) else o).squeeze().cpu().numpy()
        save(a[16], out / "smooth_random.png")
        stats["smooth_random"] = (float(np.mean((a > -200) & (a < 300)) * 100), float(a.std()))

    for k, (soft, sd) in stats.items():
        print(f"{k:16s} soft-tissue {soft:5.1f}%  sd {sd:6.1f} HU")
    print(f"\nimages -> {out}")


if __name__ == "__main__":
    main()
