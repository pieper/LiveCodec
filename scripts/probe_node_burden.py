"""Do LiveCodec latents track annotated lymph-node burden?

Two framings:
  SUPERVISED   ridge regression latents -> log(annotated node volume),
               cross-validated Spearman rho. "Can we read burden off the latent?"
  UNSUPERVISED Mahalanobis distance of each case's latent from the cohort centre
               (fit on the LOWEST-burden quartile = 'near-normal'), correlated
               with burden. "Does latent abnormality track nodal disease?"
               This is the version that needs no labels at inference.

LNQ2023 annotations are partial (index lesion at baseline), so burden is a
lower bound and correlations should be attenuated, not absent.

Controls reported alongside: scan z-extent and coverage (a bigger scan can
contain more nodes), sex, and disease class (heme disease is bulky by nature —
that is mechanism, not artifact, but it should be visible).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HEME = {"Chronic Lymphocytic Leukemia (CLL)", "Hodgkin`s Lymphoma",
        "Diffuse Large B-cell Lymphoma", "Non-Hodgkin's Lymphoma",
        "Follicular Lymphoma", "Mantle Cell Lymphoma"}


def cv_spearman(X, y, n_comp=40, seed=0, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=min(n_comp, len(tr) - 1, X.shape[1]), random_state=seed),
                             Ridge(alpha=10.0))
        pipe.fit(X[tr], y[tr])
        pred[te] = pipe.predict(X[te])
    return spearmanr(pred, y).statistic, pred


def perm_spearman(X, y, n=15, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([cv_spearman(X, rng.permutation(y), seed=seed)[0] for _ in range(n)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="features/lnq.npz")
    ap.add_argument("--nodes", default="features/lnq_nodes.csv")
    ap.add_argument("--perms", type=int, default=15)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X = d["X"]
    meta = pd.DataFrame([json.loads(m) for m in d["meta"]]).reset_index().rename(columns={"index": "row"})
    nodes = pd.read_csv(args.nodes)

    from idc_index import IDCClient
    c = IDCClient.client() if hasattr(IDCClient, "client") else IDCClient()
    c.fetch_index("clinical_index")
    clin = c.get_clinical_table("mediastinal_lymph_node_seg_clinical")
    idx = c.index
    ct = idx[(idx.collection_id == "mediastinal_lymph_node_seg") & (idx.Modality == "CT")]
    clin = clin.merge(ct[["SeriesInstanceUID", "StudyInstanceUID", "BodyPartExamined"]],
                      left_on="seriesinstanceuid", right_on="SeriesInstanceUID", how="left")

    m = meta.merge(clin, left_on="series_uid", right_on="seriesinstanceuid", how="inner")
    # SEG -> CT by referenced series when present, else by study
    n = nodes.copy()
    m = m.merge(n[["ref_ct_series_uid", "study_uid", "volume_mm3", "voxels"]],
                left_on="series_uid", right_on="ref_ct_series_uid", how="left")
    miss = m["volume_mm3"].isna()
    if miss.any():
        by_study = n.set_index("study_uid")["volume_mm3"].to_dict()
        m.loc[miss, "volume_mm3"] = m.loc[miss, "StudyInstanceUID"].map(by_study)
    m = m[m["volume_mm3"].notna()].reset_index(drop=True)
    X = X[m["row"].values]
    y = np.log10(m["volume_mm3"].values.astype(float))
    print(f"joined {len(m)} cases with both latents and node volume")
    print(f"log10 volume: mean {y.mean():.2f} sd {y.std():.2f}\n")

    m["heme"] = m["primarycondition"].isin(HEME)
    zext = np.array([s[0] for s in m["shape"]], float)
    print("=== context ===")
    print(f"median volume, heme {10**np.median(y[m.heme.values]):.0f} mm^3 vs "
          f"solid {10**np.median(y[~m.heme.values]):.0f} mm^3")
    print(f"confound check, log-volume vs z-extent:  rho {spearmanr(zext, y).statistic:+.3f}")
    cov_rank = m["BodyPartExamined"].astype("category").cat.codes.values
    print(f"confound check, log-volume vs coverage:  rho {spearmanr(cov_rank, y).statistic:+.3f}")

    print("\n=== SUPERVISED: latents -> log node volume ===")
    rho, pred = cv_spearman(X, y)
    null = perm_spearman(X, y, args.perms)
    print(f"latents             CV Spearman rho {rho:+.3f} | null {null.mean():+.3f}±{null.std():.3f} "
          f"| z={(rho-null.mean())/(null.std()+1e-9):+.1f}")
    Xz = np.column_stack([zext, (m['patientsex'] == 'F').astype(float),
                          pd.get_dummies(m['BodyPartExamined'].fillna('?')).values.astype(float)])
    rho_m, _ = cv_spearman(Xz, y)
    print(f"metadata-only floor CV Spearman rho {rho_m:+.3f}   (z-extent + sex + coverage)")

    print("\n=== UNSUPERVISED: latent abnormality (no burden labels used) ===")
    lo = y <= np.quantile(y, 0.25)                    # 'near-normal' reference set
    sc = StandardScaler().fit(X[lo])
    p = PCA(n_components=30, random_state=0).fit(sc.transform(X[lo]))
    Z = p.transform(sc.transform(X))
    cov = LedoitWolf().fit(Z[lo])
    dist = cov.mahalanobis(Z)
    rho_u = spearmanr(dist, y).statistic
    print(f"Mahalanobis distance vs log volume: rho {rho_u:+.3f}  (n={len(y)}, "
          f"reference = lowest-burden quartile)")
    hi = y >= np.quantile(y, 0.75)
    sel = lo | hi
    print(f"bulky (top quartile) vs minimal (bottom): AUC {roc_auc_score(hi[sel], dist[sel]):.3f}")

    print("\n=== SUPERVISED binary: bulky vs minimal burden ===")
    rho_b, pred_b = cv_spearman(X[sel], y[sel])
    print(f"latents -> bulky/minimal AUC {roc_auc_score(hi[sel], pred_b):.3f}  (n={sel.sum()})")

    print("\n=== within-stratum (mechanism check) ===")
    for name, mask in (("heme only", m.heme.values), ("solid only", ~m.heme.values),
                       ("CHEST only", (m.BodyPartExamined == 'CHEST').values)):
        if mask.sum() >= 60:
            r, _ = cv_spearman(X[mask], y[mask])
            print(f"  {name:12s} n={mask.sum():3d}  CV Spearman rho {r:+.3f}")


if __name__ == "__main__":
    main()
