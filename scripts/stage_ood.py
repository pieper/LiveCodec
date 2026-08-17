"""Stage the five demo out-of-distribution series from IDC into data/ood/<id>/.

The scan ids and their SeriesInstanceUIDs come from the published bundles: the
bucket's ood-scans.json only keeps a 40-char truncation of the directory name,
so the real UIDs were recovered from each scan's versions/<tag>/<id>/meta.json.
"""
import shutil, sys
from pathlib import Path
from idc_index import IDCClient

OOD = {
    "2c491b43ed8ce655": "1.3.6.1.4.1.14519.5.2.1.8778865714429543025390549910853963828",
    "b5c825e1f000de48": "1.3.6.1.4.1.14519.5.2.1.339317691405384426983700108067931171789",
    "13b2886c6cafa1e8": "1.3.6.1.4.1.14519.5.2.1.4320.7015.177447273293879900929015869634",
    "555fa7d41179aae8": "1.3.6.1.4.1.14519.5.2.1.243791570152436014482618136851436040592",
    "0f6ec1dac7a92105": "1.3.6.1.4.1.14519.5.2.1.80966168417945654246464894130387503224",
}

root = Path("data/ood"); root.mkdir(parents=True, exist_ok=True)
only = sys.argv[1:] or list(OOD)
c = IDCClient()
for sid in only:
    dest = root / sid
    if dest.exists() and any(dest.rglob("*.dcm")):
        print(f"{sid}: already staged ({len(list(dest.rglob('*.dcm')))} files)", flush=True)
        continue
    tmp = root / f".tmp-{sid}"
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    c.download_from_selection(seriesInstanceUID=OOD[sid], downloadDir=str(tmp))
    dcm = list(tmp.rglob("*.dcm"))
    if not dcm:
        print(f"{sid}: NO DICOM DOWNLOADED", flush=True); continue
    # idc-index nests under collection/patient/study/series; flatten to data/ood/<id>/
    dest.mkdir(parents=True, exist_ok=True)
    for p in dcm:
        p.rename(dest / p.name)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{sid}: staged {len(dcm)} instances", flush=True)
