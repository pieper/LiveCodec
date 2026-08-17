"""LNQ2023 spatial-latent pipeline: download -> TotalSegmentator -> mediastinum
crop -> frozen-encoder latents, one batch at a time.

Runs on a GPU instance with ~60 GB of disk, which is why everything is batched:
the full cohort is ~35 GB of DICOM, so each batch is downloaded, segmented,
cropped, encoded and then deleted, keeping only the latents and a row of image
features. Resumable — a case whose latents already exist is skipped.

Design notes that matter:

* Segmentation runs on a NIfTI we write ourselves from the same array
  `load_series` returns, so the label map comes back on EXACTLY our voxel grid.
  Running TotalSegmentator on the DICOM directory instead would return a
  canonically-reoriented volume, and recovering the mapping back to our (z,y,x)
  array would be a guess — a wrong guess mirrors anatomy silently, and
  `dicom.py` never reads ImageOrientationPatient to catch it. A landmark sanity
  check (vertebrae sit on bone, trachea sits on air) guards the result anyway.
* The crop is anchored on vertebrae T1..T10 and the carina, NOT on the bounding
  box of the mediastinal organs. That union runs to the aortic bifurcation in a
  chest-abdomen-pelvis study and stops at the diaphragm in a chest study, so a
  box anchored on it lands on different anatomy depending on scan coverage — it
  launders the very confound the crop exists to remove.
* Every crop is resampled to ONE common physical grid so latent sites are
  comparable across patients.
* Each crop is encoded in a SINGLE forward pass. The codec's normal path encodes
  in independent 32-slice chunks and every Res3/FSQ head uses GroupNorm, which
  normalises per-tensor; chunked encoding leaves periodic seam artefacts of ~3x a
  node's signal, and in a common frame they land at the same anatomy every time.
* Raw-image features are computed on the same crop, because a soft-tissue
  fraction / HU histogram is the baseline the latents have to beat.

  uv run --no-sync python scripts/lnq_pipeline.py --batch 16
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

COLLECTION = "mediastinal_lymph_node_seg"
GRID = (96, 192, 192)          # z, y, x -> fine latent 24x24x24
SPACING = (2.5, 1.0, 1.0)      # mm -> 240 mm SI x 192 x 192 mm in-plane
HU_LO, HU_HI = -1024, 3071
FAT_LO, FAT_HI = -140.0, -30.0
VERTEBRAE = [f"vertebrae_T{i}" for i in range(1, 11)]


def ts_label_map():
    from totalsegmentator.map_to_binary import class_map
    return class_map["total"]


def centroid(mask):
    idx = np.argwhere(mask)
    return idx.mean(0) if len(idx) else None


def write_nifti(vol_zyx, spacing, path):
    """Write our (z,y,x) array with an LPS->RAS affine so TotalSegmentator sees
    correct patient orientation while keeping our exact voxel grid."""
    import nibabel as nib
    sz, sy, sx = spacing
    data = np.transpose(vol_zyx, (2, 1, 0))          # -> (x, y, z)
    affine = np.diag([-sx, -sy, sz, 1.0])
    nib.save(nib.Nifti1Image(data.astype(np.int16), affine), str(path))


def read_labels(path):
    import nibabel as nib
    d = np.asanyarray(nib.load(str(path)).dataobj)   # (x, y, z), same grid we wrote
    return np.transpose(d, (2, 1, 0))                # -> (z, y, x)


def landmarks_ok(vol, lab, name2lab):
    """Vertebrae must sit on bone and trachea on air. Catches a silent mirror."""
    vt = np.isin(lab, [name2lab[v] for v in VERTEBRAE if v in name2lab])
    tr = lab == name2lab["trachea"]
    if not vt.any() or not tr.any():
        return False, "labels empty"
    hu_v, hu_t = float(vol[vt].mean()), float(vol[tr].mean())
    if hu_v < 100:
        return False, f"vertebrae mean {hu_v:.0f} HU (expected bone)"
    if hu_t > -400:
        return False, f"trachea mean {hu_t:.0f} HU (expected air)"
    return True, f"bone {hu_v:.0f} HU / air {hu_t:.0f} HU"


def resample_to_grid(vol, spacing_in, z0, y0, x0):
    """Sample the physical box with corner (z0,y0,x0) in mm onto the common grid."""
    gz, gy, gx = GRID
    dz, dy, dx = SPACING
    def norm(start, step, n, extent, sp):
        a = (start + np.arange(n) * step) / sp
        return torch.from_numpy((2 * a / max(extent - 1, 1) - 1).astype(np.float32))
    zz = norm(z0, dz, gz, vol.shape[0], spacing_in[0])
    yy = norm(y0, dy, gy, vol.shape[1], spacing_in[1])
    xx = norm(x0, dx, gx, vol.shape[2], spacing_in[2])
    gzz, gyy, gxx = torch.meshgrid(zz, yy, xx, indexing="ij")
    grid = torch.stack([gxx, gyy, gzz], dim=-1)[None]
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    return torch.nn.functional.grid_sample(t, grid, mode="bilinear",
                                           padding_mode="border",
                                           align_corners=True)[0, 0].numpy()


def image_features(crop):
    soft = float(((crop > 0) & (crop < 100)).mean())
    fat = float(((crop > FAT_LO) & (crop < FAT_HI)).mean())
    hist, _ = np.histogram(np.clip(crop, HU_LO, HU_HI), bins=24, range=(-200, 200))
    return {"soft_frac": soft, "fat_frac": fat,
            "soft_fat_ratio": soft / (fat + 1e-6),
            "mean_hu": float(crop.mean()), "sd_hu": float(crop.std()),
            "hist": (hist / crop.size).round(6).tolist()}


def ts_binary() -> str:
    """The TotalSegmentator console script lives next to the interpreter running
    us, which is not necessarily on PATH under `uv run` or a bare venv python."""
    cand = Path(sys.executable).parent / "TotalSegmentator"
    return str(cand) if cand.exists() else "TotalSegmentator"


def run_segmentations(niftis, workers, keep_logs=False):
    """TotalSegmentator --fast --ml. The GPU idles most of the run (I/O and
    resampling dominate), so several cases run concurrently."""
    exe = ts_binary()
    pending, running, done = list(niftis), [], {}
    while pending or running:
        while pending and len(running) < workers:
            src = pending.pop()
            dst = src.with_name(src.name.replace(".nii.gz", "_seg.nii.gz"))
            errf = src.with_suffix(".err") if keep_logs else None
            p = subprocess.Popen([exe, "-i", str(src), "-o", str(dst),
                                  "--fast", "--ml", "-q"],
                                 stdout=subprocess.DEVNULL,
                                 stderr=(errf.open("wb") if errf else subprocess.DEVNULL))
            running.append((p, src, dst))
        time.sleep(2)
        still = []
        for p, src, dst in running:
            if p.poll() is None:
                still.append((p, src, dst))
            else:
                done[src] = dst if (p.returncode == 0 and dst.exists()) else None
        running = still
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/big-400k-encoder.pt")
    ap.add_argument("--work", default="data/lnq-work")
    ap.add_argument("--out", default="features/lnq-spatial")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keep-logs", action="store_true",
                    help="write TotalSegmentator stderr next to each input")
    args = ap.parse_args()

    from idc_index import IDCClient
    from livecodec.dicom import load_series
    from livecodec.model2d import hu_to_unit
    from livecodec.model3d import load_model
    from livecodec.train2d import pick_device

    torch.set_grad_enabled(False)
    out = Path(args.out)
    (out / "latents").mkdir(parents=True, exist_ok=True)
    work = Path(args.work)

    device = pick_device()
    model = load_model(args.ckpt, device)
    model.eval()
    name2lab = {v: k for k, v in ts_label_map().items()}
    missing = [n for n in ["trachea"] + VERTEBRAE if n not in name2lab]
    if missing:
        raise SystemExit(f"TotalSegmentator class names not found: {missing}\n"
                         "run `TotalSegmentator --list_classes` and update VERTEBRAE/trachea")

    client = IDCClient()
    idx = client.index
    ct = idx[(idx.collection_id == COLLECTION) & (idx.Modality == "CT")]
    uids = sorted(ct.SeriesInstanceUID.unique())
    if args.limit:
        uids = uids[:args.limit]
    todo = [u for u in uids if not (out / "latents" / f"{u[-16:]}.npy").exists()]
    print(f"{len(uids)} CT series, {len(todo)} to do  (grid {GRID} @ {SPACING} mm)", flush=True)
    feat_path = out / "image_features.jsonl"

    for bi in range(0, len(todo), args.batch):
        batch = todo[bi:bi + args.batch]
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        t0 = time.time()
        print(f"\n=== batch {bi//args.batch + 1}/{(len(todo)-1)//args.batch + 1} "
              f"({len(batch)} cases) ===", flush=True)
        client.download_from_selection(seriesInstanceUID=batch, downloadDir=str(work))
        series = sorted({p.parent for p in work.rglob("*.dcm")})
        print(f"  downloaded {len(series)} series in {time.time()-t0:.0f}s", flush=True)

        loaded, niftis = {}, []
        for d in series:
            try:
                vol, info = load_series(d)
            except Exception as e:
                print(f"  {d.name[-12:]}: load failed ({type(e).__name__})", flush=True)
                continue
            vol = np.clip(vol, HU_LO, HU_HI).astype(np.float32)
            nii = work / f"{str(info['series_uid'])[-16:]}.nii.gz"
            write_nifti(vol, info["spacing"], nii)
            loaded[nii] = (vol, info)
            niftis.append(nii)

        t1 = time.time()
        segs = run_segmentations(niftis, args.workers, args.keep_logs)
        print(f"  segmented {sum(v is not None for v in segs.values())}/{len(niftis)} "
              f"in {time.time()-t1:.0f}s", flush=True)

        for nii in niftis:
            segp = segs.get(nii)
            key = nii.name.replace(".nii.gz", "")
            if segp is None:
                print(f"  {key}: SEGMENTATION FAILED", flush=True)
                continue
            vol, info = loaded[nii]
            lab = read_labels(segp)
            if lab.shape != vol.shape:
                print(f"  {key}: seg shape {lab.shape} != vol {vol.shape}", flush=True)
                continue
            ok, why = landmarks_ok(vol, lab, name2lab)
            if not ok:
                print(f"  {key}: LANDMARK CHECK FAILED ({why})", flush=True)
                continue
            sp = info["spacing"]
            vt = [c for c in (centroid(lab == name2lab[v]) for v in VERTEBRAE) if c is not None]
            tr = lab == name2lab["trachea"]
            zs = np.where(tr.any(axis=(1, 2)))[0]
            if len(vt) < 4 or not len(zs):
                print(f"  {key}: landmarks missing (T={len(vt)}, trachea={len(zs)})", flush=True)
                continue
            # carina proxy: in-plane centre of the most inferior trachea slices
            cl = centroid(tr[zs[0]:zs[0] + 5])
            zc = 0.5 * (min(c[0] for c in vt) + max(c[0] for c in vt)) * sp[0]
            crop = resample_to_grid(
                vol, sp,
                zc - GRID[0] * SPACING[0] / 2,
                cl[1] * sp[1] - GRID[1] * SPACING[1] / 2,
                cl[2] * sp[2] - GRID[2] * SPACING[2] / 2)

            x = hu_to_unit(torch.from_numpy(crop)[None, None]).to(device)
            z = model.fsq.dequantize(model.compress(x)[0])[0].float().cpu().numpy()
            np.save(out / "latents" / f"{key}.npy", z.astype(np.float16))
            rec = {"key": key, "series_uid": str(info["series_uid"]),
                   "shape": list(vol.shape), "spacing": [float(s) for s in sp],
                   "n_vertebrae": len(vt), "check": why, **image_features(crop)}
            with feat_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {key}: latent {tuple(z.shape)}  soft {rec['soft_frac']*100:5.1f}%  "
                  f"fat {rec['fat_frac']*100:5.1f}%  [{why}]", flush=True)

        shutil.rmtree(work, ignore_errors=True)

    n = len(list((out / "latents").glob("*.npy")))
    print(f"\ndone: {n} cases -> {out}")


if __name__ == "__main__":
    main()
