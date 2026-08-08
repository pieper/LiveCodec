"""CLI: build cohort manifests from the IDC index and download series.

Usage:
  livecodec-cohort collections --keyword lymph
  livecodec-cohort manifest --collection nlst --modality CT --out data/nlst.csv --max-series 100
  livecodec-cohort download --manifest data/nlst.csv --n 1 --dest data/dicom
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _client():
    from idc_index import IDCClient

    return IDCClient.client() if hasattr(IDCClient, "client") else IDCClient()


def cmd_collections(args) -> None:
    client = _client()
    index = client.index
    cols = sorted(index["collection_id"].unique())
    kw = (args.keyword or "").lower()
    for c in cols:
        if kw in c.lower():
            n = int((index["collection_id"] == c).sum())
            print(f"{c}\t{n} series")


def cmd_manifest(args) -> None:
    client = _client()
    df = client.index
    df = df[df["collection_id"].str.contains(args.collection, case=False)]
    if args.modality and "Modality" in df.columns:
        df = df[df["Modality"] == args.modality]
    size_col = next((c for c in df.columns if "size" in c.lower()), None)
    if size_col:
        df = df.sort_values(size_col)
    if args.max_series:
        df = df.head(args.max_series)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    total = df[size_col].sum() if size_col else float("nan")
    print(f"wrote {len(df)} series to {out} (~{total:.0f} MB)")


def cmd_download(args) -> None:
    client = _client()
    df = pd.read_csv(args.manifest)
    uids = df["SeriesInstanceUID"].head(args.n).tolist()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"downloading {len(uids)} series to {dest}")
    client.download_from_selection(seriesInstanceUID=uids, downloadDir=str(dest))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collections", help="list IDC collections matching a keyword")
    p.add_argument("--keyword", default="")
    p.set_defaults(func=cmd_collections)

    p = sub.add_parser("manifest", help="write a series manifest CSV for a collection")
    p.add_argument("--collection", required=True)
    p.add_argument("--modality", default="CT")
    p.add_argument("--max-series", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("download", help="download the first N series of a manifest")
    p.add_argument("--manifest", required=True)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--dest", default="data/dicom")
    p.set_defaults(func=cmd_download)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
