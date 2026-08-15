"""Compute annotated lymph-node volume per LNQ case from its DICOM SEG.

LNQ2023 annotations are PARTIAL — the index lesion was segmented at baseline,
not every node — so this is a lower bound on true nodal burden. It should still
rank cases: bigger annotated volume implies more nodal disease.

  uv run python scripts/lnq_node_volumes.py --dest data/lnq-seg --out features/lnq_nodes.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pydicom

COLLECTION = "mediastinal_lymph_node_seg"


def seg_volume_mm3(path: Path) -> dict:
    """Total segmented volume and voxel count from a DICOM SEG."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array                       # (frames, rows, cols), binary
    if arr.ndim == 2:
        arr = arr[None]
    n_vox = int((arr > 0).sum())

    shared = ds.SharedFunctionalGroupsSequence[0]
    pm = shared.PixelMeasuresSequence[0]
    py, px = (float(v) for v in pm.PixelSpacing)
    dz = float(getattr(pm, "SliceThickness", getattr(pm, "SpacingBetweenSlices", 1.0)))

    ref = ""
    try:
        ref = str(ds.ReferencedSeriesSequence[0].SeriesInstanceUID)
    except Exception:
        pass
    return {
        "seg_series_uid": str(ds.SeriesInstanceUID),
        "study_uid": str(ds.StudyInstanceUID),
        "patient_id": str(getattr(ds, "PatientID", "")),
        "ref_ct_series_uid": ref,
        "n_segments": len(getattr(ds, "SegmentSequence", [1])),
        "voxels": n_vox,
        "volume_mm3": n_vox * px * py * dz,
        "frames": int(arr.shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="data/lnq-seg")
    ap.add_argument("--out", default="features/lnq_nodes.csv")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    dest = Path(args.dest)
    if not args.skip_download:
        from idc_index import IDCClient

        c = IDCClient.client() if hasattr(IDCClient, "client") else IDCClient()
        idx = c.index
        seg = idx[(idx.collection_id == COLLECTION) & (idx.Modality == "SEG")]
        uids = sorted(seg["SeriesInstanceUID"].unique())
        dest.mkdir(parents=True, exist_ok=True)
        print(f"downloading {len(uids)} SEG series (~310 MB)…", flush=True)
        c.download_from_selection(seriesInstanceUID=uids, downloadDir=str(dest))

    files = sorted(dest.rglob("*.dcm"))
    print(f"parsing {len(files)} SEG files")
    rows = []
    for i, f in enumerate(files):
        try:
            rows.append(seg_volume_mm3(f))
        except Exception as e:
            print(f"  skip {f.name[:18]}: {type(e).__name__}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    v = np.array([r["volume_mm3"] for r in rows])
    print(f"\nwrote {out}: {len(rows)} cases")
    print(f"annotated node volume mm^3: median {np.median(v):.0f}, "
          f"IQR {np.percentile(v,25):.0f}-{np.percentile(v,75):.0f}, "
          f"range {v.min():.0f}-{v.max():.0f}")
    print(f"cases with zero annotation: {(v == 0).sum()}")


if __name__ == "__main__":
    main()
