#!/usr/bin/env python3
"""Run UNI2-h mean/max pooling baselines on the locked TCGA 246/231 split."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "locked_release_20260902" / "baselines"
FEATURE_DIR = ROOT / "tcga_stad_uni2h" / "TCGA-STAD" / "features"
M4_CLASSES = ["EBV", "MSI", "GS", "CIN"]
TASKS = {
    "M1_immune_sensitive": {"n_classes": 2, "label_col": "M1_label", "fold_registry": "fold_registry_M1_immune_sensitive.csv"},
    "M2_msi": {"n_classes": 2, "label_col": "MSI", "fold_registry": "fold_registry_M2_msi.csv"},
    "M3_ebv": {"n_classes": 2, "label_col": "EBV", "fold_registry": "fold_registry_M3_ebv.csv"},
    "M4_subtype4": {"n_classes": 4, "label_col": "M4_label", "fold_registry": "fold_registry_M4_subtype4.csv"},
}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def load_or_build_pooling_features(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_csv = OUT / "uni2h_mean_pooling_features_246.csv"
    max_csv = OUT / "uni2h_max_pooling_features_246.csv"
    if mean_csv.exists() and max_csv.exists():
        return pd.read_csv(mean_csv), pd.read_csv(max_csv)

    rows_mean = []
    rows_max = []
    for i, r in enumerate(manifest.itertuples(), 1):
        slide_id = r.slide_id
        path = FEATURE_DIR / f"{slide_id}.h5"
        with h5py.File(path, "r") as h:
            x = h["features"]
            arr = x[0] if len(x.shape) == 3 else x[:]
            mean_vec = np.asarray(arr).mean(axis=0)
            max_vec = np.asarray(arr).max(axis=0)
        base = {
            "patient_id": r.patient_id,
            "slide_id": slide_id,
            "n_patches": int(getattr(r, "n_patches", -1)),
        }
        rows_mean.append({**base, **{f"f{i}": float(v) for i, v in enumerate(mean_vec)}})
        rows_max.append({**base, **{f"f{i}": float(v) for i, v in enumerate(max_vec)}})
        if i % 25 == 0:
            print(f"pooled {i}/{len(manifest)} slides", flush=True)
    mean_df = pd.DataFrame(rows_mean)
    max_df = pd.DataFrame(rows_max)
    mean_df.to_csv(mean_csv, index=False)
    max_df.to_csv(max_csv, index=False)
    return mean_df, max_df


def metrics_binary(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    pred = (score >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else math.nan,
        "average_precision": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else math.nan,
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }


def metrics_multi(y: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    row_sum = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sum, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sum > 0)
    pred = proba.argmax(axis=1)
    return {
        "n": int(len(y)),
        "class_order": M4_CLASSES,
        "class_counts": {M4_CLASSES[i]: int((y == i).sum()) for i in range(4)},
        "macro_ovr_auroc": float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
        "macro_average_precision": float(average_precision_score(label_binarize(y, classes=[0, 1, 2, 3]), proba, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }


def run_one(pool_name: str, feat: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [c for c in feat.columns if c.startswith("f")]
    labels = manifest.set_index("patient_id")
    x_all = feat.set_index("patient_id")[feature_cols]
    output: dict[str, Any] = {}
    for task, cfg in TASKS.items():
        fr = pd.read_csv(ROOT / "results" / cfg["fold_registry"])
        proba_rows = []
        for rep in sorted(fr["repeat"].unique()):
            for fold in sorted(fr["fold"].unique()):
                fdf = fr[(fr["repeat"].eq(rep)) & (fr["fold"].eq(fold))]
                tr_pids = [p for p in fdf[fdf["split"].eq("train")]["patient_id"].tolist() if p in x_all.index]
                va_pids = [p for p in fdf[fdf["split"].eq("val")]["patient_id"].tolist() if p in x_all.index]
                if not va_pids:
                    continue
                y_tr = labels.loc[tr_pids, cfg["label_col"]].astype(int).to_numpy()
                y_va = labels.loc[va_pids, cfg["label_col"]].astype(int).to_numpy()
                pipe = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "lr",
                            LogisticRegression(
                                max_iter=3000,
                                class_weight="balanced",
                                C=0.5,
                            ),
                        ),
                    ]
                )
                pipe.fit(x_all.loc[tr_pids], y_tr)
                prob = pipe.predict_proba(x_all.loc[va_pids])
                classes = list(pipe.named_steps["lr"].classes_)
                full_prob = np.zeros((len(va_pids), cfg["n_classes"]))
                for ci, cls in enumerate(classes):
                    full_prob[:, int(cls)] = prob[:, ci]
                for pid, yy, pv in zip(va_pids, y_va, full_prob):
                    row = {"patient_id": pid, "repeat": int(rep), "fold": int(fold), "label": int(yy)}
                    for ci in range(cfg["n_classes"]):
                        row[f"prob_c{ci}"] = float(pv[ci])
                    proba_rows.append(row)
        fold_pred = pd.DataFrame(proba_rows)
        fold_pred.to_csv(OUT / f"{pool_name}_{task}_fold_predictions.csv", index=False)
        agg_dict = {"label": ("label", "first"), "n_validation_predictions": ("fold", "count")}
        for ci in range(cfg["n_classes"]):
            agg_dict[f"prob_c{ci}"] = (f"prob_c{ci}", "mean")
        oof = fold_pred.groupby("patient_id").agg(**agg_dict).reset_index()
        oof.to_csv(OUT / f"{pool_name}_{task}_oof.csv", index=False)
        if cfg["n_classes"] == 2:
            output[task] = metrics_binary(oof["label"].to_numpy(), oof["prob_c1"].to_numpy())
        else:
            output[task] = metrics_multi(oof["label"].to_numpy(), oof[[f"prob_c{i}" for i in range(4)]].to_numpy())
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(ROOT / "results" / "locked_release_20260902" / "patient_manifest.csv")
    mean_df, max_df = load_or_build_pooling_features(manifest)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_set": "TCGA 246 train / 231 center-isolated OOF, same fold registries as M1-M4",
        "feature_source": str(FEATURE_DIR),
        "model": "SimpleImputer + StandardScaler + penalized LogisticRegression(class_weight='balanced', C=0.5)",
        "baselines": {
            "uni2h_mean_pooling": run_one("uni2h_mean_pooling", mean_df, manifest),
            "uni2h_max_pooling": run_one("uni2h_max_pooling", max_df, manifest),
        },
    }
    write_json(OUT / "baseline_metrics.json", payload)
    print(json.dumps({"out_dir": str(OUT), "metrics": str(OUT / "baseline_metrics.json")}, indent=2))


if __name__ == "__main__":
    main()
