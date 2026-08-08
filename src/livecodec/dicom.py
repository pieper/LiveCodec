"""Load a DICOM CT series directory into a HU int16 volume."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom


def load_series(series_dir: str | Path) -> tuple[np.ndarray, dict]:
    """Return (volume[z,y,x] as int16 HU, info dict with spacing etc.)."""
    paths = sorted(p for p in Path(series_dir).rglob("*") if p.is_file() and p.suffix != ".json")
    slices = []
    for p in paths:
        try:
            ds = pydicom.dcmread(p)
        except Exception:
            continue
        if not hasattr(ds, "PixelData"):
            continue
        slices.append(ds)
    if not slices:
        raise ValueError(f"no DICOM image slices found under {series_dir}")

    def z_of(ds):
        if hasattr(ds, "ImagePositionPatient"):
            return float(ds.ImagePositionPatient[2])
        return float(getattr(ds, "InstanceNumber", 0))

    slices.sort(key=z_of)
    vol = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        vol.append(arr * slope + intercept)
    volume = np.stack(vol).astype(np.int16)

    ds0 = slices[0]
    spacing_xy = [float(v) for v in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
    if len(slices) > 1:
        spacing_z = abs(z_of(slices[1]) - z_of(slices[0])) or 1.0
    else:
        spacing_z = float(getattr(ds0, "SliceThickness", 1.0))
    info = {
        "spacing": (spacing_z, spacing_xy[0], spacing_xy[1]),
        "shape": volume.shape,
        "series_uid": str(getattr(ds0, "SeriesInstanceUID", "")),
        "modality": str(getattr(ds0, "Modality", "")),
    }
    return volume, info
