"""JPEG 2000 slice codec (pylibjpeg-openjpeg) with volume-level byte-budget search.

The baseline curve encodes each slice at a compression ratio chosen so the
whole volume hits a target byte budget. Each budget gets its own rate-optimal
encode, i.e. an upper bound on what progressive HTJ2K streaming can deliver at
that budget — a deliberately generous baseline. (HTJ2K's HT block coder trades
~5-10% coding efficiency for decode speed, so classic J2K rate-distortion is a
near-identical, slightly favorable stand-in; decode *speed* is measured
separately in the browser harness.)
"""

from __future__ import annotations

import numpy as np
import openjpeg


def encode_slice(img: np.ndarray, ratio: float) -> bytes:
    """Encode one int16 slice; ratio<=1 means lossless."""
    if ratio <= 1.0:
        return openjpeg.encode(img)
    return openjpeg.encode(img, compression_ratios=[float(ratio)])


def decode_slice(data: bytes) -> np.ndarray:
    return openjpeg.decode(data)


def encode_volume(volume: np.ndarray, ratio: float) -> list[bytes]:
    return [encode_slice(volume[z], ratio) for z in range(volume.shape[0])]


def decode_volume(streams: list[bytes], like: np.ndarray) -> np.ndarray:
    out = np.empty_like(like)
    for z, data in enumerate(streams):
        out[z] = decode_slice(data).astype(like.dtype)
    return out


def encode_to_budget(
    volume: np.ndarray, budget_bytes: int, tol: float = 0.08, max_iter: int = 8
) -> tuple[list[bytes], float]:
    """Choose a compression ratio so total encoded size approaches budget_bytes.

    OpenJPEG's rate control is approximate, so iterate a multiplicative
    correction on a coarse z-subsample, then encode the full volume.
    Returns (per-slice codestreams, ratio used).
    """
    nz = volume.shape[0]
    probe_idx = np.linspace(0, nz - 1, num=min(16, nz), dtype=int)
    probe = volume[probe_idx]
    target_probe = budget_bytes * (probe.nbytes / volume.nbytes)

    ratio = max(volume.nbytes / budget_bytes, 1.5)
    for _ in range(max_iter):
        size = sum(len(encode_slice(probe[i], ratio)) for i in range(probe.shape[0]))
        err = size / target_probe
        if abs(1.0 - err) < tol:
            break
        ratio = float(np.clip(ratio * err, 1.5, 10000.0))
    streams = encode_volume(volume, ratio)
    return streams, ratio
