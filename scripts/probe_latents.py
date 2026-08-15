"""Do LiveCodec latents carry diagnostic signal? Linear probes on the LNQ cohort.

Methodology notes, because this is the kind of question that is easy to answer
wrongly:
  * Every score is grouped-stratified cross-validated by PATIENT.
  * A permutation test (labels shuffled) gives the null — an AUC is only
    interesting relative to that, not relative to 0.5, at n≈400.
  * Coverage (BodyPartExamined) is confounded with the label: heme cases are
    4x more likely to be ABDOMEN and 25x more likely to be NECK scans. So the
    headline task runs on the CHEST-ONLY subset, and we also report what a
    metadata-only model (coverage + age + sex) achieves as the floor a latent
    model has to beat to be saying anything about the images.

Usage:
  uv run --extra train python scripts/probe_latents.py --features features/lnq.npz
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HEME = {"Chronic Lymphocytic Leukemia (CLL)", "Hodgkin`s Lymphoma",
        "Diffuse Large B-cell Lymphoma", "Non-Hodgkin's Lymphoma",
        "Follicular Lymphoma", "Mantle Cell Lymphoma"}


def cv_auc(X, y, n_splits=5, seed=0, n_comp=40):
    """Mean CV AUC of an L2 logistic probe on PCA-reduced features."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=min(n_comp, len(tr) - 1, X.shape[1]), random_state=seed),
            LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"),
        )
        pipe.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def permutation_null(X, y, n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [cv_auc(X, rng.permutation(y), seed=seed)[0] for _ in range(n)]


def report(name, X, y, n_perm=20):
    if len(np.unique(y)) < 2 or min(np.bincount(y.astype(int))) < 10:
        print(f"{name}: skipped (too few in a class)")
        return
    auc, sd = cv_auc(X, y)
    null = permutation_null(X, y, n_perm)
    mu, s = float(np.mean(null)), float(np.std(null) + 1e-9)
    z = (auc - mu) / s
    p = (1 + sum(v >= auc for v in null)) / (1 + len(null))
    print(f"{name:38s} AUC {auc:.3f} ±{sd:.3f} | null {mu:.3f}±{s:.3f} | z={z:+.1f} p<{p:.3f} "
          f"| n={len(y)} pos={int(y.sum())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--perms", type=int, default=20)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X = d["X"]
    meta = pd.DataFrame([json.loads(m) for m in d["meta"]])
    print(f"features {X.shape} for {len(meta)} series")

    from idc_index import IDCClient
    c = IDCClient.client() if hasattr(IDCClient, "client") else IDCClient()
    c.fetch_index("clinical_index")
    clin = c.get_clinical_table("mediastinal_lymph_node_seg_clinical")
    idx = c.index
    ct = idx[(idx["collection_id"] == "mediastinal_lymph_node_seg") & (idx["Modality"] == "CT")]
    clin = clin.merge(ct[["SeriesInstanceUID", "BodyPartExamined", "PatientAge"]],
                      left_on="seriesinstanceuid", right_on="SeriesInstanceUID", how="left")
    meta = meta.reset_index().rename(columns={"index": "row"})
    m = meta.merge(clin, left_on="series_uid", right_on="seriesinstanceuid", how="inner")
    X = X[m["row"].values]
    print(f"joined to clinical labels: {len(m)} cases\n")

    m["heme"] = m["primarycondition"].isin(HEME).astype(int)
    m["is_f"] = (m["patientsex"] == "F").astype(int)

    # metadata-only floor: what coverage + sex + age alone achieve
    cov = pd.get_dummies(m["BodyPartExamined"].fillna("?")).values.astype(float)
    age = pd.to_numeric(m["PatientAge"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    med = age.median()
    age = age.fillna(0.0 if not np.isfinite(med) else med)   # collection may not record age
    Xmeta = np.nan_to_num(np.column_stack([cov, m["is_f"].values, age.values]).astype(float))
    print(f"metadata baseline: {cov.shape[1]} coverage dummies, "
          f"age available for {int(np.isfinite(pd.to_numeric(m['PatientAge'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')).sum())}/{len(m)}")

    print("=== all cases (coverage CONFOUNDED with label — read the floor) ===")
    report("metadata only -> heme", Xmeta, m["heme"].values, args.perms)
    report("latents -> heme", X, m["heme"].values, args.perms)

    chest = (m["BodyPartExamined"] == "CHEST").values
    print(f"\n=== CHEST-only subset (coverage matched), n={chest.sum()} ===")
    report("metadata only -> heme", Xmeta[chest], m["heme"].values[chest], args.perms)
    report("latents -> heme", X[chest], m["heme"].values[chest], args.perms)

    print("\n=== positive control: sex (should be clearly predictable from CT) ===")
    report("latents -> sex", X, m["is_f"].values, args.perms)
    report("latents -> sex (chest only)", X[chest], m["is_f"].values[chest], args.perms)

    print("\n=== per-class one-vs-rest, CHEST-only (>=25 cases) ===")
    for cls, n in m.loc[chest, "primarycondition"].value_counts().items():
        if n >= 25:
            y = (m["primarycondition"].values[chest] == cls).astype(int)
            report(f"latents -> {str(cls)[:28]}", X[chest], y, args.perms)


if __name__ == "__main__":
    main()
