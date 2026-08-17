"""Phase 0 pre-flight gate: can the actual probe pipeline recover nodal burden
from un-pooled latents when acquisition protocol is randomised against it?

This runs BEFORE any GPU spend and gates the LNQ spatial-latent experiment.

Design. Synthetic burden is inserted into real CT blocks, and a nuisance state
(slice thickness, dose, kernel, contrast phase, shift) is drawn INDEPENDENTLY of
burden, so protocol carries no information about the target by construction. We
then run the real probe — un-pooled latents, PCA, ridge, leave-one-volume-out CV
— and ask whether it recovers burden anyway. Grouping by volume matters: the same
patient appears at every burden level, so an ungrouped split would leak identity.

Two earlier framings were wrong and are recorded so they are not repeated:
  * a paired difference ||z_perturbed - z_baseline|| is not available at test
    time; only absolute summaries of one patient's latent field are.
  * hand-picked scalars (norm quantiles, per-channel means) move with burden but
    move just as much with slice thickness. A learned projection is the thing
    under test, not a summary statistic.

Nodes are modelled as mediastinal FAT (-140..-30 HU) replaced by SOFT TISSUE
(~+45 HU) - a ~145 HU contrast - spread over several nodes, matching the cohort
median of 23.7 mL across ~7 nodes.

  uv run --no-sync python scripts/phase0_sweep.py --volumes 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAT_LO, FAT_HI = -140.0, -30.0
NODE_HU = 45.0
VOX_ML = 1e-3


def mediastinal_block(vol: np.ndarray, size=(64, 128, 128)) -> np.ndarray:
    """Axial block richest in fat near the volume centre. The npy cache carries
    no spacing/orientation, so this stands in for Phase 1's landmark-anchored
    crop - enough to characterise the encoder, not a substitute for it."""
    dz, dy, dx = size
    z0 = max(0, (vol.shape[0] - dz) // 2)
    cy, cx = vol.shape[1] // 2, vol.shape[2] // 2
    best, best_frac = None, -1.0
    for oy in (-24, 0, 24):
        for ox in (-24, 0, 24):
            y0 = int(np.clip(cy + oy - dy // 2, 0, vol.shape[1] - dy))
            x0 = int(np.clip(cx + ox - dx // 2, 0, vol.shape[2] - dx))
            b = np.ascontiguousarray(vol[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]).astype(np.float32)
            frac = float(((b > FAT_LO) & (b < FAT_HI)).mean())
            if frac > best_frac:
                best, best_frac = b, frac
    return best


def add_burden(block, target_mL, n_nodes, node_hu, rng) -> tuple[np.ndarray, float]:
    if target_mL <= 0:
        return block.copy(), 0.0
    fat = np.argwhere((block > FAT_LO) & (block < FAT_HI))
    if len(fat) == 0:
        return block.copy(), 0.0
    out = block.copy()
    r = (target_mL / n_nodes / VOX_ML * 3 / (4 * np.pi)) ** (1 / 3)
    zz, yy, xx = np.ogrid[:block.shape[0], :block.shape[1], :block.shape[2]]
    placed = 0
    for _ in range(n_nodes):
        c = fat[rng.integers(len(fat))]
        m = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) <= r * r
        out[m] = node_hu
        placed += int(m.sum())
    return out, placed * VOX_ML


def nz_thickness(a, k):
    """Thicker acquisition resampled back to the same grid - the agreed control.
    Removes geometry, not through-plane blur."""
    n = (a.shape[0] // k) * k
    t = a[:n].reshape(-1, k, *a.shape[1:]).mean(1)
    return torch.nn.functional.interpolate(torch.from_numpy(t)[None, None], size=a.shape,
                                           mode="trilinear", align_corners=False)[0, 0].numpy()


def apply_nuisance(a, name, rng):
    from scipy import ndimage
    if name == "none":            return a
    if name == "thickness x2":    return nz_thickness(a, 2)
    if name == "thickness x4":    return nz_thickness(a, 4)
    if name == "dose noise":      return a + rng.normal(0, 15.0, a.shape).astype(np.float32)
    if name == "kernel smooth":   return ndimage.gaussian_filter(a, sigma=(0, 1.0, 1.0))
    if name == "contrast +50HU":  return np.where(a > -200, a + 50.0, a).astype(np.float32)
    if name == "shift 4vox":      return np.roll(a, 4, axis=2)
    raise ValueError(name)


NUISANCES = ["none", "thickness x2", "thickness x4", "dose noise",
             "kernel smooth", "contrast +50HU", "shift 4vox"]


def grouped_probe(X, y, groups, n_comp=20, alpha=10.0):
    """Leave-one-group-out ridge on PCA. Returns mean-of-fold Spearman.

    Mean-of-fold, not pooled-OOF: pooling raw predictions from folds with
    different intercepts is biased (a null predictor reports a large negative
    correlation). Preprocessing is fit inside the fold.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    rhos = []
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if te.sum() < 3:
            continue
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=min(n_comp, int(tr.sum()) - 1, X.shape[1])),
                             Ridge(alpha=alpha))
        pipe.fit(X[tr], y[tr])
        r = spearmanr(pipe.predict(X[te]), y[te]).statistic
        if np.isfinite(r):
            rhos.append(r)
    return float(np.mean(rhos)), float(np.std(rhos)), len(rhos)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="results/big-400k-encoder.pt")
    ap.add_argument("--cache", default="data/npy")
    ap.add_argument("--volumes", type=int, default=8)
    ap.add_argument("--burdens", default="0,10,25,50,100,200")
    ap.add_argument("--nodes", type=int, default=7)
    ap.add_argument("--out", default="results/phase0-sweep.json")
    ap.add_argument("--feats", default="results/phase0-feats.npz")
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--reuse", action="store_true",
                    help="skip encoding, re-analyse the cached feature matrix")
    args = ap.parse_args()

    if args.reuse:
        d = np.load(args.feats, allow_pickle=True)
        analyse(d["X"], d["yb"], d["yn"], d["g"],
                json.loads(str(d["meta"])), args.out, args.perms)
        return

    from livecodec.model2d import hu_to_unit
    from livecodec.model3d import load_model
    from livecodec.train2d import pick_device

    torch.set_grad_enabled(False)
    device = pick_device()
    model = load_model(args.ckpt, device)
    model.eval()

    def latent(a):
        x = hu_to_unit(torch.from_numpy(np.ascontiguousarray(a))[None, None]).to(device)
        return model.fsq.dequantize(model.compress(x)[0])[0].float().cpu().numpy()

    vols = sorted(Path(args.cache).glob("*.npy"))[:args.volumes]
    burdens = [float(b) for b in args.burdens.split(",")]
    print(f"device={device.type}  volumes={len(vols)}  burdens={burdens}", flush=True)

    feats, y_burden, y_nuis, groups, meta = [], [], [], [], []
    for vi, vp in enumerate(vols):
        rng = np.random.default_rng(1000 + vi)
        block = mediastinal_block(np.load(vp, mmap_mode="r"))
        fat = float(((block > FAT_LO) & (block < FAT_HI)).sum()) * VOX_ML
        print(f"[{vi}] {vp.name}  fat {fat:5.0f} mL", flush=True)
        for b in burdens:
            # nuisance drawn INDEPENDENTLY of burden -> carries no information about y
            nname = NUISANCES[rng.integers(len(NUISANCES))]
            a, act = add_burden(block, b, args.nodes, NODE_HU, rng)
            a = apply_nuisance(a, nname, rng)
            z = latent(a)
            feats.append(z.ravel())
            y_burden.append(np.log10(act + 1.0))
            y_nuis.append(float(NUISANCES.index(nname)))
            groups.append(vi)
            meta.append(dict(vol=vi, burden=b, actual=act, nuisance=nname))
            print(f"    burden {b:6.1f} (act {act:6.1f})  nuisance {nname:<14s} "
                  f"latent {z.shape}", flush=True)

    X = np.asarray(feats, dtype=np.float32)
    yb, yn, g = np.asarray(y_burden), np.asarray(y_nuis), np.asarray(groups)
    np.savez_compressed(args.feats, X=X, yb=yb, yn=yn, g=g,
                        meta=json.dumps(meta))
    print(f"\ncached features -> {args.feats}")
    analyse(X, yb, yn, g, meta, args.out, args.perms)


def analyse(X, yb, yn, g, meta, out, perms):
    print(f"feature matrix {X.shape}  ({len(np.unique(g))} volumes, leave-one-volume-out)\n")

    r, s, k = grouped_probe(X, yb, g)
    print(f"GATE  latents -> log burden        rho {r:+.3f} +/- {s:.3f}  ({k} folds)")
    rn, sn, _ = grouped_probe(X, yn, g)
    print(f"      latents -> nuisance id       rho {rn:+.3f} +/- {sn:.3f}   (protocol IS readable)")

    # Permutation null: shuffle burden WITHIN each volume, so the null preserves
    # both the group structure and each volume's marginal burden distribution.
    # A single shuffle is not a control - with few samples per fold the spread is
    # wide enough that one draw says nothing.
    rng = np.random.default_rng(0)
    null = []
    for _ in range(perms):
        ysh = yb.copy()
        for gg in np.unique(g):
            idx = np.where(g == gg)[0]
            ysh[idx] = rng.permutation(ysh[idx])
        null.append(grouped_probe(X, ysh, g)[0])
    null = np.asarray(null)
    p = (1 + int((null >= r).sum())) / (1 + len(null))
    print(f"      permutation null ({perms})       mean {null.mean():+.3f} +/- {null.std():.3f}"
          f"   [{np.percentile(null,5):+.3f}, {np.percentile(null,95):+.3f}]")
    print(f"\n      observed {r:+.3f}   empirical p = {p:.4f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(dict(
        burden_rho=r, burden_sd=s, nuisance_rho=rn,
        null_mean=float(null.mean()), null_sd=float(null.std()), p=p,
        n_perms=perms, meta=meta), indent=1))
    print(f"\nwrote {out}")
    verdict = "PASS" if (p < 0.05 and r > 0) else "FAIL"
    print(f"\nGATE {verdict}: burden rho {r:+.3f} vs null {null.mean():+.3f}, p={p:.4f}")


if __name__ == "__main__":
    main()
