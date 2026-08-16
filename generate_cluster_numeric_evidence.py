#!/usr/bin/env python3
"""Generate numeric unsupervised cluster evidence from TCGA UNI2-h features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min


def load_h5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h:
        features = h["features"]
        if len(features.shape) == 3 and features.shape[0] == 1:
            x = features[0][:]
        else:
            x = features[:]
        coord_key = "coords_patching" if "coords_patching" in h else "coords"
        coords = h[coord_key][:]
    return x.astype(np.float32), coords.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_csv", type=Path, default=Path("results/audit_first_stage/tcga_public_feature_matched_246_cohort.csv"))
    parser.add_argument("--feature_dir", type=Path, default=Path("tcga_stad_uni2h/TCGA-STAD/features"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/cpu_supplement/unsupervised_cluster_numeric_evidence"))
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--sample_per_slide", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort_csv)
    rng = np.random.default_rng(args.seed)

    sampled_x = []
    sampled_meta = []
    for _, row in cohort.iterrows():
        path = args.feature_dir / f"{row['slide_id']}.h5"
        x, coords = load_h5(path)
        n = x.shape[0]
        take = min(args.sample_per_slide, n)
        idx = rng.choice(n, size=take, replace=False)
        sampled_x.append(x[idx])
        for local_i in idx:
            sampled_meta.append(
                {
                    "patient_id": row["patient_id"],
                    "slide_id": row["slide_id"],
                    "subtype": row["subtype"],
                    "M1_label": int(row["M1_label"]),
                    "patch_index": int(local_i),
                    "coord_x": int(coords[local_i, 0]),
                    "coord_y": int(coords[local_i, 1]),
                }
            )
    sample_matrix = np.concatenate(sampled_x, axis=0)
    sampled_meta_df = pd.DataFrame(sampled_meta)

    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        random_state=args.seed,
        batch_size=4096,
        n_init="auto",
        max_iter=200,
    )
    kmeans.fit(sample_matrix)
    np.save(args.out_dir / "cluster_centers.npy", kmeans.cluster_centers_)

    nearest_idx, nearest_dist = pairwise_distances_argmin_min(kmeans.cluster_centers_, sample_matrix)
    rep = sampled_meta_df.iloc[nearest_idx].copy()
    rep.insert(0, "cluster", np.arange(args.n_clusters))
    rep["distance_to_center"] = nearest_dist
    rep.to_csv(args.out_dir / "cluster_representative_patch_coords.csv", index=False)

    sampled_meta_df["cluster"] = kmeans.predict(sample_matrix)
    sampled_meta_df.to_csv(args.out_dir / "sampled_patch_cluster_assignments.csv", index=False)
    subtype_dist = (
        sampled_meta_df.groupby(["cluster", "subtype"]).size().reset_index(name="n_sampled_patches")
    )
    subtype_dist.to_csv(args.out_dir / "cluster_distribution_by_subtype_sampled.csv", index=False)

    patient_rows = []
    for _, row in cohort.iterrows():
        path = args.feature_dir / f"{row['slide_id']}.h5"
        x, _coords = load_h5(path)
        counts = np.zeros(args.n_clusters, dtype=np.int64)
        for start in range(0, x.shape[0], 4096):
            labels = kmeans.predict(x[start : start + 4096])
            counts += np.bincount(labels, minlength=args.n_clusters)
        out = {
            "patient_id": row["patient_id"],
            "slide_id": row["slide_id"],
            "subtype": row["subtype"],
            "M1_label": int(row["M1_label"]),
            "n_patches": int(x.shape[0]),
        }
        for c, value in enumerate(counts):
            out[f"cluster_{c}_count"] = int(value)
            out[f"cluster_{c}_fraction"] = float(value / max(x.shape[0], 1))
        patient_rows.append(out)
    pd.DataFrame(patient_rows).to_csv(args.out_dir / "patient_cluster_fractions.csv", index=False)

    summary = {
        "cohort_csv": str(args.cohort_csv),
        "feature_dir": str(args.feature_dir),
        "n_patients": int(cohort["patient_id"].nunique()),
        "n_clusters": int(args.n_clusters),
        "sample_per_slide": int(args.sample_per_slide),
        "n_sampled_patches": int(sample_matrix.shape[0]),
        "feature_dim": int(sample_matrix.shape[1]),
        "seed": int(args.seed),
        "important_limitation": "Numeric unsupervised clusters are not pathology names. TIL/tumor/stroma/necrosis labels require rendered representative patches and pathologist review.",
        "outputs": {
            "cluster_centers": str(args.out_dir / "cluster_centers.npy"),
            "representative_patch_coords": str(args.out_dir / "cluster_representative_patch_coords.csv"),
            "sampled_assignments": str(args.out_dir / "sampled_patch_cluster_assignments.csv"),
            "distribution_by_subtype": str(args.out_dir / "cluster_distribution_by_subtype_sampled.csv"),
            "patient_cluster_fractions": str(args.out_dir / "patient_cluster_fractions.csv"),
        },
    }
    (args.out_dir / "cluster_numeric_evidence_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
