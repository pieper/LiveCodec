"""Do the codec's spatial latents predict mediastinal nodal burden, and do they
beat a trivial measurement of the same crop?

Arms:
  metadata  acquisition + demographics only - the floor any claim must clear
  image     soft-tissue / fat fraction + HU histogram of the mediastinum crop
  latent    the frozen encoder's un-pooled latent field over the same crop
  meta+X    whether X adds anything to the floor

Statistical choices, each fixing a defect in the earlier probe:
  * mean-of-fold Spearman, never pooled out-of-fold. Pooling raw predictions
    from folds with different intercepts is biased - on this data a predictor
    with true rho +0.003 reported -0.127, and a true +0.303 reported +0.194.
  * >=20 repeated splits with a reported seed interval. The prior headline
    (+0.111 vs a floor of +0.166) sat entirely inside seed noise.
  * >=1000 permutations. The old default of 15 has a p-floor of 0.0625 and
    cannot produce significance. Permuting y leaves X[tr] untouched, so the
    per-fold scaler+PCA basis is cached and the null is nearly free.
  * The endpoint is DELTA over the metadata floor, not raw correlation.

  uv run --no-sync --with scikit-learn python scripts/probe_spatial.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def agg_latent(z: np.ndarray, how: str) -> np.ndarray:
    """z is (C, zz, yy, xx). Returns a 1-D feature vector."""
    C = z.shape[0]
    flat = z.reshape(C, -1)
    if how == "flat":
        return z.ravel()
    if how == "mean":
        return flat.mean(1)
    if how == "max":
        return flat.max(1)
    if how == "q90":
        return np.quantile(flat, 0.90, axis=1)
    if how == "q99":
        return np.quantile(flat, 0.99, axis=1)
    if how == "quantiles":
        return np.concatenate([np.quantile(flat, q, axis=1)
                               for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)])
    if how == "pool4":
        import torch
        t = torch.from_numpy(z.astype(np.float32))[None]
        return torch.nn.functional.adaptive_avg_pool3d(t, (4, 4, 4)).numpy().ravel()
    if how == "hist":
        return np.concatenate([np.histogram(flat[c], bins=16, range=(-1, 1))[0] / flat.shape[1]
                               for c in range(C)])
    raise ValueError(how)


def fold_rhos(X, y, seed, folds=5, n_comp=30, alpha=10.0):
    """Mean-of-fold Spearman. Preprocessing is fit inside each fold."""
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rhos = []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=min(n_comp, len(tr) - 1, X.shape[1])),
                             Ridge(alpha=alpha))
        pipe.fit(X[tr], y[tr])
        r = spearmanr(pipe.predict(X[te]), y[te]).statistic
        if np.isfinite(r):
            rhos.append(r)
    return float(np.mean(rhos)) if rhos else np.nan


def score(X, y, seeds=20, **kw):
    v = [fold_rhos(X, y, s, **kw) for s in range(seeds)]
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))


def perm_p(X, y, observed, perms=1000, seeds=5, **kw):
    rng = np.random.default_rng(0)
    null = []
    for _ in range(perms):
        ysh = rng.permutation(y)
        null.append(np.median([fold_rhos(X, ysh, s, **kw) for s in range(seeds)]))
    null = np.asarray(null)
    return (1 + int((null >= observed).sum())) / (1 + len(null)), float(null.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feats", default="features/lnq-spatial")
    ap.add_argument("--targets", default="features/lnq_targets.csv")
    ap.add_argument("--group", default="Fully Annotated")
    ap.add_argument("--target", default="total_mm3")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--aggs", default="flat,pool4,quantiles,max,hist")
    args = ap.parse_args()

    feats = Path(args.feats)
    img = {}
    with (feats / "image_features.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            img[r["key"]] = r
    tgt = {r["key"]: r for r in csv.DictReader(open(args.targets))}

    keys = sorted(k for k in img
                  if k in tgt and (feats / "latents" / f"{k}.npy").exists()
                  and tgt[k]["group"] == args.group)
    if len(keys) < 20:
        raise SystemExit(f"only {len(keys)} cases matched for group {args.group!r}")

    y = np.array([np.log10(float(tgt[k][args.target]) + 1.0) for k in keys])
    if args.target == "n_nodes":
        y = np.array([float(tgt[k]["n_nodes"]) for k in keys])

    Z = [np.load(feats / "latents" / f"{k}.npy").astype(np.float32) for k in keys]

    # ── arms ────────────────────────────────────────────────────────────────
    X_img = np.array([[img[k]["soft_frac"], img[k]["fat_frac"], img[k]["soft_fat_ratio"],
                       img[k]["mean_hu"], img[k]["sd_hu"], *img[k]["hist"]] for k in keys],
                     dtype=np.float32)
    X_meta = np.array([[float(tgt[k]["slice_thickness"]), float(tgt[k]["in_plane"]),
                        float(img[k]["shape"][0]),
                        float(img[k]["shape"][0]) * float(img[k]["spacing"][0])]
                       for k in keys], dtype=np.float32)

    print(f"group={args.group!r}  n={len(keys)}  target={args.target}  "
          f"({y.min():.2f}..{y.max():.2f})")
    print(f"latent grid {Z[0].shape}   perms={args.perms}  seeds={args.seeds}\n")
    print(f"{'arm':<34s} {'rho (median)':>14s} {'[5-95%]':>18s} {'p':>8s}")
    print("-" * 78)

    results = {}

    def run(name, X):
        r, lo, hi = score(X, y, seeds=args.seeds)
        p, nul = perm_p(X, y, r, perms=args.perms)
        results[name] = dict(rho=r, lo=lo, hi=hi, p=p, null=nul, dim=int(X.shape[1]))
        print(f"{name:<34s} {r:>+14.3f} {f'[{lo:+.3f},{hi:+.3f}]':>18s} {p:>8.4f}")
        return r

    r_meta = run("metadata floor", X_meta)
    r_img = run("image (soft/fat + HU hist)", X_img)
    run("metadata + image", np.hstack([X_meta, X_img]))
    for how in args.aggs.split(","):
        Xl = np.array([agg_latent(z, how) for z in Z], dtype=np.float32)
        r = run(f"latent [{how}]  {Xl.shape[1]}d", Xl)
        run(f"metadata + latent [{how}]", np.hstack([X_meta, Xl]))
        results[f"latent [{how}]"]["delta_over_meta"] = r - r_meta
        results[f"latent [{how}]"]["delta_over_image"] = r - r_img

    print("-" * 78)
    print(f"\nfloor (metadata) {r_meta:+.3f}   image baseline {r_img:+.3f}")
    print("A latent arm is interesting only if it clears BOTH.")
    print(f"\nNOTE: {len(args.aggs.split(','))} latent aggregations were tried; "
          "treat individual p-values accordingly.")

    out = feats / f"probe_{args.group.split()[0].lower()}_{args.target}.json"
    out.write_text(json.dumps(dict(group=args.group, target=args.target,
                                   n=len(keys), results=results), indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
