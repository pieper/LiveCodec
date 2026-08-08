"""CLI: HTJ2K/J2K byte->quality baseline curve for a CT volume.

Usage:
  livecodec-baseline --series data/some_series_dir --out results/
  livecodec-baseline --synthetic --out results/       # no data needed
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import j2k, metrics
from .dicom import load_series

DEFAULT_BUDGETS_MB = [0.5, 1, 2, 5, 10, 20, 50]


def synthetic_volume(shape=(64, 256, 256), seed=0) -> np.ndarray:
    """CT-ish phantom: ellipsoid body, lung-ish voids, bones, HU noise."""
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, s) for s in shape], indexing="ij")
    vol = np.full(shape, -1000.0)
    body = (x**2 / 0.81 + y**2 / 0.49) < 1
    vol[body] = 40.0
    lungs = ((np.abs(x) - 0.35) ** 2 / 0.04 + y**2 / 0.09 + z**2 / 0.5) < 1
    vol[lungs & body] = -800.0
    spine = (x**2 + (y - 0.45) ** 2) < 0.01
    vol[spine & body] = 700.0
    vol += rng.normal(0, 12, shape)  # quantum noise floor
    return np.clip(vol, -1024, 3071).astype(np.int16)


def run_curve(volume: np.ndarray, budgets_mb, out_dir: Path, label: str) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mb in budgets_mb:
        budget = int(mb * 1e6)
        if budget >= volume.nbytes:
            continue
        t0 = time.time()
        streams, ratio = j2k.encode_to_budget(volume, budget)
        actual = sum(len(s) for s in streams)
        t1 = time.time()
        recon = j2k.decode_volume(streams, volume)
        t2 = time.time()
        row = {
            "label": label,
            "budget_mb": mb,
            "actual_mb": round(actual / 1e6, 3),
            "ratio_param": round(ratio, 1),
            "compression": round(volume.nbytes / actual, 1),
            **{k: round(v, 4) for k, v in metrics.evaluate(volume, recon).items()},
            "encode_s": round(t1 - t0, 2),
            "decode_s": round(t2 - t1, 2),
        }
        rows.append(row)
        print(json.dumps(row))
    csv_path = out_dir / f"baseline_{label}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, out_dir / f"baseline_{label}.png")
    print(f"wrote {csv_path}")
    return rows


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mb = [r["actual_mb"] for r in rows]
    axes[0].plot(mb, [r["psnr"] for r in rows], "o-")
    axes[0].set(xscale="log", xlabel="MB transferred", ylabel="PSNR (dB)", title="J2K baseline")
    axes[1].plot(mb, [r["ssim_soft_tissue"] for r in rows], "o-", color="tab:orange")
    axes[1].set(xscale="log", xlabel="MB transferred", ylabel="SSIM (soft-tissue window)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--series", help="DICOM series directory")
    src.add_argument("--synthetic", action="store_true", help="use a synthetic CT phantom")
    ap.add_argument("--budgets-mb", type=float, nargs="+", default=DEFAULT_BUDGETS_MB)
    ap.add_argument("--out", default="results", help="output directory")
    args = ap.parse_args()

    if args.synthetic:
        volume, label = synthetic_volume(), "synthetic"
    else:
        volume, info = load_series(args.series)
        label = info["series_uid"][-12:] or Path(args.series).name
        print(f"loaded {info['modality']} volume {info['shape']} spacing {info['spacing']}")
    print(f"raw size: {volume.nbytes / 1e6:.1f} MB")
    run_curve(volume, args.budgets_mb, Path(args.out), label)


if __name__ == "__main__":
    main()
