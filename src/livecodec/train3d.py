"""Phase 2 CLI: train the 3D FSQ autoencoder with a live visual dashboard.

Usage:
  uv run livecodec-train3d --data data/dicom --steps 100000 --out results/phase2
  uv run livecodec-train3d --eval-only --ckpt results/phase2/model.pt --out results/phase2

The dashboard (results/<out>/dashboard.html) is fully self-contained and is
rewritten every --dash-every steps: stat tiles, loss curve, and axial/coronal
reconstruction panels at both bitrates (coarse-only and coarse+fine) next to
J2K encodes of the same window at matched bytes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import zstandard

from . import j2k, metrics
from .dashboard import Dashboard
from .model2d import hu_to_unit, unit_to_hu
from .model3d import FSQAutoencoder3D, save_model
from .train2d import cache_volumes, find_series_dirs, is_val, open_volumes, pick_device, ssim_loss


class Crop3D(torch.utils.data.Dataset):
    """Random (cz, cxy, cxy) crops, biased away from air."""

    def __init__(self, volumes, cz=32, cxy=128, length=10_000_000):
        self.volumes = [v for v in volumes if v.shape[0] >= cz]
        if not self.volumes:
            raise SystemExit(f"no volumes with >= {cz} slices")
        self.cz, self.cxy, self.length = cz, cxy, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        for _ in range(4):
            vol = self.volumes[rng.integers(len(self.volumes))]
            z = rng.integers(0, vol.shape[0] - self.cz + 1)
            y = rng.integers(0, vol.shape[1] - self.cxy + 1)
            x = rng.integers(0, vol.shape[2] - self.cxy + 1)
            patch = vol[z : z + self.cz, y : y + self.cxy, x : x + self.cxy]
            if (patch[self.cz // 2] > -900).mean() > 0.2:
                break
        patch = patch.astype(np.float32)
        if rng.random() < 0.5:
            patch = patch[:, :, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0)
        return hu_to_unit(t)


def _zbytes(codes: torch.Tensor) -> int:
    return len(zstandard.ZstdCompressor(level=19).compress(codes.cpu().numpy().tobytes()))


def dc_sideband(win: np.ndarray, recon: np.ndarray, block: int = 64) -> tuple[np.ndarray, int]:
    """Wavelet-style DC guarantee for the neural decode: per-block mean error,
    quantized to 4 HU steps (int8), trilinearly upsampled and subtracted.
    Returns (corrected recon, sideband bytes after zstd)."""
    bz = max(1, min(block, win.shape[0]))
    zb, yb, xb = (max(1, s // b) for s, b in zip(win.shape, (bz, block, block)))
    err = (recon.astype(np.float32) - win.astype(np.float32))[
        : zb * bz, : yb * block, : xb * block
    ]
    means = err.reshape(zb, bz, yb, block, xb, block).mean(axis=(1, 3, 5))
    q = np.clip(np.round(means / 4.0), -128, 127).astype(np.int8)
    nbytes = len(zstandard.ZstdCompressor(level=19).compress(q.tobytes()))
    corr = torch.nn.functional.interpolate(
        torch.from_numpy(q.astype(np.float32) * 4.0)[None, None],
        size=win.shape, mode="trilinear", align_corners=False,
    ).squeeze().numpy()
    fixed = np.clip(recon.astype(np.float32) - corr, -1024, 3071).astype(np.int16)
    return fixed, nbytes


@torch.no_grad()
def eval_and_illustrate(model, val_vols, device, dash: Dashboard, ez: int, exy: int):
    model.eval()
    dash.rd_rows = []
    for vi, vol in enumerate(val_vols[:3]):
        if vol.shape[0] < ez:
            continue
        z0 = (vol.shape[0] - ez) // 2
        y0 = (vol.shape[1] - exy) // 2
        win = np.ascontiguousarray(vol[z0 : z0 + ez, y0 : y0 + exy, y0 : y0 + exy]).astype(
            np.int16
        )
        x = hu_to_unit(torch.from_numpy(win.astype(np.float32))[None, None]).to(device)
        cf, cc = model.compress(x)
        b_coarse = _zbytes(cc)
        b_total = b_coarse + _zbytes(cf)
        recon = {
            "coarse": unit_to_hu(model.decompress(cf, cc, coarse_only=True)),
            "fine": unit_to_hu(model.decompress(cf, cc)),
        }
        recon = {k: v.squeeze().cpu().numpy().astype(np.int16) for k, v in recon.items()}
        # DC sideband: guarantees the intensity profile like J2K's LL band;
        # its (tiny) cost is added to the byte accounting.
        recon["coarse"], dc_c = dc_sideband(win, recon["coarse"])
        recon["fine"], dc_f = dc_sideband(win, recon["fine"])
        b_coarse += dc_c
        b_total += dc_c + dc_f
        j2k_rec = {}
        for tier, budget in (("coarse", b_coarse), ("fine", b_total)):
            streams, _ = j2k.encode_to_budget(win, budget)
            j2k_rec[tier] = j2k.decode_volume(streams, win)

        name = f"val{vi} ({win.shape[0]}x{win.shape[1]}x{win.shape[2]}, raw {win.nbytes/1e6:.0f} MB)"
        views = []
        for view, sel in (("axial", lambda a: a[ez // 2]), ("coronal", lambda a: a[:, exy // 2])):
            panels = [("original", f"{win.nbytes/1e6:.1f} MB raw", sel(win))]
            for tier, b in (("coarse", b_coarse), ("fine", b_total)):
                m = metrics.evaluate(win, recon[tier])
                panels.append(
                    (
                        f"neural {tier}",
                        f"{b/1e3:.1f} KB · {win.nbytes/b:.0f}:1 · "
                        f"{m['psnr']:.1f} dB / {m['ssim_soft_tissue']:.3f}",
                        sel(recon[tier]),
                    )
                )
                mj = metrics.evaluate(win, j2k_rec[tier])
                panels.append(
                    (
                        f"J2K @ {tier} bytes",
                        f"{b/1e3:.1f} KB · {mj['psnr']:.1f} dB / {mj['ssim_soft_tissue']:.3f}",
                        sel(j2k_rec[tier]),
                    )
                )
            views.append((view, panels))
        dash.add_case(name, views)

        for tier, b in (("coarse", b_coarse), ("fine", b_total)):
            m = metrics.evaluate(win, recon[tier])
            mj = metrics.evaluate(win, j2k_rec[tier])
            dash.rd_rows.append(
                {
                    "case": f"val{vi}",
                    "tier": tier,
                    "KB": round(b / 1e3, 1),
                    "ratio": f"{win.nbytes/b:.0f}:1",
                    "neural dB": round(m["psnr"], 2),
                    "J2K dB": round(mj["psnr"], 2),
                    "neural SSIM": round(m["ssim_soft_tissue"], 4),
                    "J2K SSIM": round(mj["ssim_soft_tissue"], 4),
                }
            )
    model.train()
    return dash.rd_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/dicom")
    ap.add_argument("--cache", default="data/npy")
    ap.add_argument("--out", default="results/phase2")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--crop-z", type=int, default=32)
    ap.add_argument("--crop-xy", type=int, default=128)
    ap.add_argument("--eval-z", type=int, default=32)
    ap.add_argument("--eval-xy", type=int, default=512)
    ap.add_argument("--ssim-weight", type=float, default=0.2)
    ap.add_argument("--dc-weight", type=float, default=0.1,
                    help="weight on |mean(recon)-mean(x)| per crop (DC anchor)")
    ap.add_argument("--p-drop-fine", type=float, default=0.3)
    ap.add_argument("--dash-every", type=int, default=2000)
    ap.add_argument("--enc-width", type=int, default=96)
    ap.add_argument("--enc-depth", type=int, default=2)
    ap.add_argument("--dec-width", type=int, default=64)
    ap.add_argument("--dec-arch", default="3d", choices=["3d", "2.5d"])
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="train the decoder only (published latents stay valid)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or out_dir.name

    paths = cache_volumes(find_series_dirs(args.data), Path(args.cache))
    val_paths = [p for p in paths if is_val(p)]
    train_paths = [p for p in paths if not is_val(p)]
    if len(val_paths) < 2:
        val_paths += train_paths[-(2 - len(val_paths)):]
        train_paths = [p for p in train_paths if p not in val_paths]
    train_vols, val_vols = open_volumes(train_paths), open_volumes(val_paths)

    model = FSQAutoencoder3D(
        enc_width=args.enc_width, dec_width=args.dec_width, dec_arch=args.dec_arch,
        enc_depth=args.enc_depth,
    ).to(device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        dropped = [k for k in missing + unexpected if not k.startswith("decoder.")]
        if dropped:
            raise SystemExit(f"ckpt mismatch beyond the decoder: {dropped[:6]}")
        if missing or unexpected:
            print(f"loaded encoder/fsq from ckpt; decoder starts fresh ({args.dec_arch})")
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        model.encoder.eval()
    n_all = sum(p.numel() for p in model.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())

    dash = Dashboard(run_name=run_name, out_path=out_dir / "dashboard.html")
    dash.meta = {
        "step": 0,
        "device": device.type,
        "params": f"{n_all/1e6:.1f}M",
        "decoder": f"{n_dec/1e6:.1f}M",
        "train vols": len(train_vols),
        "val vols": len(val_vols),
    }
    print(json.dumps({k: str(v) for k, v in dash.meta.items()}), flush=True)

    if not args.eval_only:
        loader = torch.utils.data.DataLoader(
            Crop3D(train_vols, args.crop_z, args.crop_xy), batch_size=args.batch, num_workers=2
        )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        model.train()

        def batches():
            while True:
                yield from loader

        it, t0, ema = batches(), time.time(), None
        for step in range(1, args.steps + 1):
            x = next(it).to(device)
            recon = model(x, p_drop_fine=args.p_drop_fine)
            loss = 0.7 * torch.nn.functional.l1_loss(recon, x) + 0.3 * torch.nn.functional.mse_loss(
                recon, x
            )
            if args.dc_weight:
                loss = loss + args.dc_weight * (
                    recon.mean(dim=(1, 2, 3, 4)) - x.mean(dim=(1, 2, 3, 4))
                ).abs().mean()
            if args.ssim_weight:
                b, _, cz, cy, cx = recon.shape
                loss = loss + args.ssim_weight * ssim_loss(
                    recon.permute(0, 2, 1, 3, 4).reshape(b * cz, 1, cy, cx),
                    x.permute(0, 2, 1, 3, 4).reshape(b * cz, 1, cy, cx),
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % 100 == 0 or step == 1:
                dash.log_loss(step, ema)
            if step % 250 == 0 or step == 1:
                rate = step / (time.time() - t0)
                print(
                    f"step {step}/{args.steps} loss {ema:.5f} "
                    f"({rate:.2f} it/s, eta {(args.steps-step)/rate/3600:.1f} h)",
                    flush=True,
                )
            if step % args.dash_every == 0 or step == 500:
                save_model(model, out_dir / "model.pt")
                dash.meta.update(
                    step=f"{step:,}",
                    **{"it/s": f"{step/(time.time()-t0):.2f}", "loss (ema)": f"{ema:.4f}"},
                )
                eval_and_illustrate(model, val_vols, device, dash, args.eval_z, args.eval_xy)
                dash.render()
        save_model(model, out_dir / "model.pt")

    dash.meta["step"] = "final" if not args.eval_only else "eval-only"
    rows = eval_and_illustrate(model, val_vols, device, dash, args.eval_z, args.eval_xy)
    dash.render()
    print(json.dumps(rows, indent=1))
    print(f"dashboard: {out_dir / 'dashboard.html'}")


if __name__ == "__main__":
    main()
