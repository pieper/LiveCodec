"""Convert staged DICOM to the npy cache (deleting each series' DICOM after a
successful conversion, to fit corpus + cache on one volume), then validate
every cache file by actually reading from a memory-map — the public-API check
that positively identifies truncated files. Deletion happens ONLY on that
specific mmap/read failure, never on incidental errors."""

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from livecodec.train2d import cache_volumes  # noqa: E402

dirs = sorted({p.parent for p in Path("data/dicom").rglob("*.dcm")})
print(f"{len(dirs)} series to convert", flush=True)
for i, d in enumerate(dirs):
    cache_volumes([d], Path("data/npy"))
    shutil.rmtree(d)
    if (i + 1) % 100 == 0:
        print(f"converted {i + 1}/{len(dirs)}", flush=True)

removed = 0
files = sorted(Path("data/npy").glob("*.npy"))
for p in files:
    try:
        a = np.load(p, mmap_mode="r")
        _ = a.shape
        _ = int(a.ravel()[0]) + int(a.ravel()[-1])  # touch first + last page
    except (ValueError, OSError) as e:
        print(f"truncated cache file {p.name}: {e}", flush=True)
        p.unlink()
        removed += 1
print(f"validated {len(files) - removed} volumes; removed {removed}", flush=True)
