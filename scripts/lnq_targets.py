"""Build the LNQ2023 burden targets, keyed to match the spatial-latent pipeline.

The SEG `SeriesDescription` tag is authoritative for annotation completeness:
393 "Partially Annotated" (index lesion only) and 120 "Fully Annotated" (true
total burden). Nothing else in the IDC release flags this.

Beyond total volume this records per-connected-component statistics, which give
two things the single number does not: `n_nodes`, a target far less sensitive to
reconstructed slice thickness than a volume is, and `largest_mm3`, the index
lesion proxy that lets the 393 partial cases be compared like-for-like with the
120 (on the fully annotated set the largest component ranks total burden at
rho +0.944, so the partial label is only mildly attenuated).

  uv run --no-sync python scripts/lnq_targets.py --out features/lnq_targets.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage

MIN_VOX = 20          # drop annotation specks below this many voxels


def per_case(path: Path) -> dict | None:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr[None]
    sf = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
    px, py = (float(v) for v in sf.PixelSpacing)
    dz = float(getattr(sf, "SliceThickness", 0) or
               getattr(sf, "SpacingBetweenSlices", 0) or 1.0)
    vox = px * py * dz

    lab, n = ndimage.label(arr > 0)
    if not n:
        return None
    sizes = np.asarray(ndimage.sum(arr > 0, lab, range(1, n + 1)))
    keep = sizes[sizes >= MIN_VOX]
    if not len(keep):
        keep = sizes[:1]

    # summed max in-plane cross-sectional area: a burden proxy built only from
    # in-plane geometry, so it cannot be inflated or deflated by slice thickness
    area = 0.0
    for c in np.where(sizes >= MIN_VOX)[0] + 1:
        per_slice = (lab == c).sum(axis=(1, 2))
        area += float(per_slice.max()) * px * py

    ref = ""
    try:
        ref = str(ds.ReferencedSeriesSequence[0].SeriesInstanceUID)
    except Exception:
        pass
    return {
        "seg_series_uid": str(ds.SeriesInstanceUID),
        "ref_ct_series_uid": ref,
        "key": ref[-16:],
        "group": str(getattr(ds, "SeriesDescription", "")),
        "patient_id": str(getattr(ds, "PatientID", "")),
        "total_mm3": round(float(sizes.sum()) * vox, 1),
        "largest_mm3": round(float(keep.max()) * vox, 1),
        "n_nodes": int(len(keep)),
        "sum_maxarea_mm2": round(area, 1),
        "slice_thickness": dz,
        "in_plane": px,
        "frames": int(arr.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segs", default="data/lnq-seg")
    ap.add_argument("--out", default="features/lnq_targets.csv")
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.segs).rglob("*.dcm")):
        r = per_case(p)
        if r:
            rows.append(r)
    if not rows:
        raise SystemExit(f"no SEG files under {args.segs}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import collections
    grp = collections.Counter(r["group"] for r in rows)
    print(f"wrote {out}  ({len(rows)} cases)")
    for g, n in grp.most_common():
        sub = [r for r in rows if r["group"] == g]
        med = lambda k: float(np.median([r[k] for r in sub]))
        print(f"  {g:22s} n={n:3d}  median total {med('total_mm3')/1000:6.1f} mL  "
              f"nodes {med('n_nodes'):.0f}  thickness {med('slice_thickness'):.2f} mm")
    missing = sum(1 for r in rows if not r["key"])
    if missing:
        print(f"  WARNING: {missing} cases have no ReferencedSeriesSequence -> no CT key")


if __name__ == "__main__":
    main()
