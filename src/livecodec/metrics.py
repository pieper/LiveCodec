"""Fidelity metrics between an original HU volume and a reconstruction."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

SOFT_TISSUE_WINDOW = (-160.0, 240.0)  # W400 L40


def window_uint8(vol: np.ndarray, window=SOFT_TISSUE_WINDOW) -> np.ndarray:
    lo, hi = window
    v = np.clip(vol.astype(np.float32), lo, hi)
    return ((v - lo) / (hi - lo) * 255.0).astype(np.uint8)


def evaluate(original: np.ndarray, recon: np.ndarray) -> dict:
    o = original.astype(np.float32)
    r = recon.astype(np.float32)
    err = o - r
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    data_range = float(np.percentile(o, 99.9) - np.percentile(o, 0.1)) or 1.0
    psnr = float(10 * np.log10(data_range**2 / mse)) if mse > 0 else float("inf")

    # SSIM on the soft-tissue window, middle slice band, to reflect what a reviewer sees
    ow, rw = window_uint8(o), window_uint8(r)
    zs = np.linspace(0, o.shape[0] - 1, num=min(9, o.shape[0]), dtype=int)
    ssim = float(np.mean([structural_similarity(ow[z], rw[z], data_range=255) for z in zs]))
    return {"hu_mae": mae, "psnr": psnr, "ssim_soft_tissue": ssim}
