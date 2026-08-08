"""Phase 1 CLI: train the 2D FSQ autoencoder and compare R-D against J2K.

Usage:
  uv run livecodec-train2d --data data/dicom --steps 5000 --out results/phase1
  uv run livecodec-train2d --data data/dicom --eval-only --ckpt results/phase1/model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import zstandard

from . import j2k, metrics
from .dicom import load_series
from .model2d import FSQAutoencoder, hu_to_unit, unit_to_hu


def find_series_dirs(root: str | Path) -> list[Path]:
    """Leaf directories that contain DICOM files."""
    leaves = set()
    for p in Path(root).rglob("*.dcm"):
        leaves.add(p.parent)
    return sorted(leaves)


def load_volumes(dirs: list[Path]) -> list[np.ndarray]:
    vols = []
    for d in dirs:
        try:
            vol, info = load_series(d)
        except ValueError:
            continue
        if info["modality"] == "CT" and vol.shape[1] == 512 and vol.shape[0] >= 8:
            vols.append(vol)
    return vols


class SliceCrops(torch.utils.data.Dataset):
    """Random 256^2 crops of random slices, biased away from empty air."""

    def __init__(self, volumes: list[np.ndarray], crop: int = 256, length: int = 100_000):
        self.volumes = volumes
        self.crop = crop
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        for _ in range(4):
            vol = self.volumes[rng.integers(len(self.volumes))]
            sl = vol[rng.integers(vol.shape[0])]
            c = self.crop
            y = rng.integers(0, sl.shape[0] - c + 1)
            x = rng.integers(0, sl.shape[1] - c + 1)
            patch = sl[y : y + c, x : x + c]
            if (patch > -900).mean() > 0.25:  # reject mostly-air crops
                break
        patch = patch.astype(np.float32)
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0)
        return hu_to_unit(t)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def neural_encode_bytes(model: FSQAutoencoder, x: torch.Tensor) -> tuple[bytes, torch.Tensor]:
    """Compress one (1,1,H,W) slice; returns (zstd bytes, integer codes)."""
    codes = model.compress(x)
    cctx = zstandard.ZstdCompressor(level=19)
    payload = cctx.compress(codes.cpu().numpy().tobytes())
    return payload, codes


def evaluate(model: FSQAutoencoder, volumes: list[np.ndarray], device, out_dir: Path) -> dict:
    """Slice-level R-D of the neural codec vs J2K at matched bytes."""
    model.eval()
    rows = []
    for vol in volumes:
        zs = np.linspace(0, vol.shape[0] - 1, num=min(8, vol.shape[0]), dtype=int)
        for z in zs:
            sl = vol[z]
            x = hu_to_unit(torch.from_numpy(sl.astype(np.float32))[None, None]).to(device)
            payload, codes = neural_encode_bytes(model, x)
            recon = unit_to_hu(model.decompress(codes.to(device)))
            recon_np = recon.squeeze().cpu().numpy().astype(np.int16)
            m_neural = metrics.evaluate(sl[None], recon_np[None])

            ratio = sl.nbytes / len(payload)
            j2k_bytes = j2k.encode_slice(sl, ratio)
            j2k_recon = j2k.decode_slice(j2k_bytes).astype(np.int16)
            m_j2k = metrics.evaluate(sl[None], j2k_recon[None])
            rows.append(
                {
                    "z": int(z),
                    "neural_bytes": len(payload),
                    "j2k_bytes": len(j2k_bytes),
                    "compression": round(sl.nbytes / len(payload), 1),
                    **{f"neural_{k}": round(v, 4) for k, v in m_neural.items()},
                    **{f"j2k_{k}": round(v, 4) for k, v in m_j2k.items()},
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase1_rd.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = {
        "slices": len(rows),
        "mean_compression": float(np.mean([r["compression"] for r in rows])),
        "neural_psnr": float(np.mean([r["neural_psnr"] for r in rows])),
        "j2k_psnr_at_matched_bytes": float(np.mean([r["j2k_psnr"] for r in rows])),
        "neural_ssim": float(np.mean([r["neural_ssim_soft_tissue"] for r in rows])),
        "j2k_ssim_at_matched_bytes": float(np.mean([r["j2k_ssim_soft_tissue"] for r in rows])),
    }
    (out_dir / "phase1_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/dicom")
    ap.add_argument("--out", default="results/phase1")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-series", type=int, default=2)
    ap.add_argument("--ckpt", default=None, help="checkpoint to load")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = find_series_dirs(args.data)
    volumes = load_volumes(dirs)
    if len(volumes) < args.val_series + 1:
        raise SystemExit(f"need more data: only {len(volumes)} usable volumes in {args.data}")
    train_vols, val_vols = volumes[args.val_series :], volumes[: args.val_series]
    n_slices = sum(v.shape[0] for v in train_vols)
    print(f"device={device.type} train={len(train_vols)} vols ({n_slices} slices) "
          f"val={len(val_vols)} vols")

    model = FSQAutoencoder().to(device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    n_params = sum(p.numel() for p in model.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"params: {n_params/1e6:.1f}M total, {n_dec/1e6:.1f}M decoder")

    if not args.eval_only:
        loader = torch.utils.data.DataLoader(
            SliceCrops(train_vols), batch_size=args.batch, num_workers=2
        )
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        model.train()
        t0, it = time.time(), iter(loader)
        for step in range(1, args.steps + 1):
            x = next(it).to(device)
            recon = model(x)
            loss = 0.7 * torch.nn.functional.l1_loss(recon, x) + 0.3 * torch.nn.functional.mse_loss(recon, x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 250 == 0 or step == 1:
                rate = step / (time.time() - t0)
                print(f"step {step}/{args.steps} loss {loss.item():.5f} "
                      f"({rate:.1f} it/s, eta {(args.steps-step)/rate/60:.0f} min)", flush=True)
        torch.save(model.state_dict(), out_dir / "model.pt")
        print(f"saved {out_dir / 'model.pt'}")

    summary = evaluate(model, val_vols, device, out_dir)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
