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


def block_projections(blocks, seeds=20, folds=5, n_comp=30):
    """Like `projections`, but each block is reduced SEPARATELY and the reduced
    blocks are concatenated.

    Running one PCA over [metadata | latent] would be meaningless: 4 metadata
    columns against 69,120 latent columns means the leading components are pure
    latent and the metadata is effectively discarded, so the combined arm would
    score like the latent arm rather than testing whether the latent ADDS to the
    floor. Reducing per block keeps every block represented.
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    out = []
    n = blocks[0].shape[0]
    for seed in range(seeds):
        for tr, te in KFold(folds, shuffle=True, random_state=seed).split(np.arange(n)):
            Ztr, Zte = [], []
            for B in blocks:
                sc = StandardScaler().fit(B[tr])
                Btr, Bte = sc.transform(B[tr]), sc.transform(B[te])
                k = min(n_comp, len(tr) - 1, B.shape[1])
                if B.shape[1] > k:
                    pca = PCA(n_components=k).fit(Btr)
                    Btr, Bte = pca.transform(Btr), pca.transform(Bte)
                Ztr.append(Btr)
                Zte.append(Bte)
            out.append((seed, tr, te, np.hstack(Ztr), np.hstack(Zte)))
    return out


def projections(X, seeds=20, folds=5, n_comp=30):
    """Per (seed, fold): fit scaler+PCA on the TRAINING rows only and return the
    projected train/test matrices.

    Permuting y never changes X[tr], so the projection is identical across every
    permutation. Caching it here is what makes a 1000-permutation null cheap: the
    inner loop becomes a ridge solve on a (n x 30) matrix instead of an SVD on
    (n x 69120). Fitting inside the fold is still what keeps it leak-free.
    """
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    out = []
    for seed in range(seeds):
        for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
            sc = StandardScaler().fit(X[tr])
            pca = PCA(n_components=min(n_comp, len(tr) - 1, X.shape[1])).fit(sc.transform(X[tr]))
            out.append((seed, tr, te,
                        pca.transform(sc.transform(X[tr])),
                        pca.transform(sc.transform(X[te]))))
    return out


def score_cached(proj, y, seeds, alpha=10.0):
    """Median over seeds of the mean-of-fold Spearman.

    Mean-of-fold, never pooled out-of-fold: pooling raw predictions from folds
    with different intercepts is biased under weak signal.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    per_seed = {}
    for seed, tr, te, Ztr, Zte in proj:
        r = Ridge(alpha=alpha).fit(Ztr, y[tr])
        rho = spearmanr(r.predict(Zte), y[te]).statistic
        if np.isfinite(rho):
            per_seed.setdefault(seed, []).append(rho)
    v = [float(np.mean(v)) for v in per_seed.values() if v]
    if not v:
        return np.nan, np.nan, np.nan
    return float(np.median(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))


def perm_p(proj, y, observed, perms=1000, seeds=20):
    rng = np.random.default_rng(0)
    null = np.empty(perms)
    for i in range(perms):
        null[i] = score_cached(proj, rng.permutation(y), seeds)[0]
    return (1 + int((null >= observed).sum())) / (1 + perms), float(np.nanmean(null))


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

    def run(name, *blocks):
        proj = (block_projections(list(blocks), seeds=args.seeds) if len(blocks) > 1
                else projections(blocks[0], seeds=args.seeds))
        X = np.hstack(blocks)
        r, lo, hi = score_cached(proj, y, args.seeds)
        p, nul = perm_p(proj, y, r, perms=args.perms, seeds=args.seeds)
        results[name] = dict(rho=r, lo=lo, hi=hi, p=p, null=nul, dim=int(X.shape[1]))
        print(f"{name:<34s} {r:>+14.3f} {f'[{lo:+.3f},{hi:+.3f}]':>18s} {p:>8.4f}", flush=True)
        return r

    r_meta = run("metadata floor", X_meta)
    r_img = run("image (soft/fat + HU hist)", X_img)
    run("metadata + image", X_meta, X_img)
    for how in args.aggs.split(","):
        Xl = np.array([agg_latent(z, how) for z in Z], dtype=np.float32)
        name = f"latent [{how}]  {Xl.shape[1]}d"
        r = run(name, Xl)
        rc = run(f"metadata + latent [{how}]", X_meta, Xl)
        results[name]["delta_over_meta"] = rc - r_meta
        results[name]["delta_over_image"] = r - r_img

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
