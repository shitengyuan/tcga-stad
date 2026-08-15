#!/usr/bin/env python3
"""
Generate first-stage audit materials from the current TCGA-STAD project outputs.

This script is intentionally read-only with respect to model/training artifacts:
it rebuilds audit tables and metrics from existing OOF predictions, saved model
metadata, CPTAC predictions/features, and locally available label tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
OUT = RESULTS / "audit_first_stage"
CPTAC_BASE = BASE.parent / "dataset" / "cptac-stad-histopathology"
CPTAC_LABELS = CPTAC_BASE / "labels" / "cptac_stad_2026_tcga_subtype_labels_qc_pass.csv"
CPTAC_FEATURE_DIR = RESULTS / "external_cptac_features_20x256"
CPTAC_INFER_DIR = RESULTS / "external_cptac_feature_infer_20x256_4gpu"

M4_CLASSES = ["EBV", "MSI", "GS", "CIN"]
M4_TO_ID = {"STAD_EBV": 0, "EBV": 0, "STAD_MSI": 1, "MSI": 1, "MSI-H": 1, "STAD_GS": 2, "GS": 2, "STAD_CIN": 3, "CIN": 3}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(cmd: list[str], cwd: Path = BASE) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def site_from_patient(pid: str) -> str:
    parts = pid.split("-")
    return parts[1] if len(parts) > 1 else "UNK"


def build_tcga_cohort() -> pd.DataFrame:
    m1 = read_csv(RESULTS / "oof_preds_M1_immune_sensitive.csv")
    m2 = read_csv(RESULTS / "oof_preds_M2_msi.csv")
    m3 = read_csv(RESULTS / "oof_preds_M3_ebv.csv")
    m4 = read_csv(RESULTS / "oof_preds_M4_subtype4.csv")
    m5 = read_csv(RESULTS / "oof_preds_M5_clinical.csv")

    df = m1[["patient_id", "subtype", "label"]].rename(columns={"label": "M1_label"})
    df["site"] = df["patient_id"].map(site_from_patient)
    df["MSI"] = df["subtype"].eq("STAD_MSI").astype(int)
    df["EBV"] = df["subtype"].eq("STAD_EBV").astype(int)
    df["M4_subtype"] = df["subtype"].str.replace("STAD_", "", regex=False)
    df["M4_label"] = df["subtype"].map(M4_TO_ID).astype("Int64")
    df["POLE"] = pd.NA
    df["slide_id"] = pd.NA
    df["source"] = "OOF reconstruction from results/oof_preds_M1-M5.csv"
    df["include_status"] = "included_in_289_oof"
    df["include_exclude_reason"] = "has OOF predictions for M1-M5; original clinical.csv unavailable in current checkout"

    for name, src in [
        ("M1_immune_sensitive", m1),
        ("M2_msi", m2),
        ("M3_ebv", m3),
        ("M4_subtype4", m4),
        ("M5_clinical", m5),
    ]:
        cols = [c for c in src.columns if c.startswith("prob_c")]
        add = src[["patient_id", *cols]].copy()
        add = add.rename(columns={c: f"{name}_{c}" for c in cols})
        df = df.merge(add, on="patient_id", how="left")

    return df.sort_values("patient_id").reset_index(drop=True)


def reconstruct_folds(cohort: pd.DataFrame, n_splits: int = 5, n_repeats: int = 3, seed: int = 42, min_site_for_val: int = 8) -> tuple[pd.DataFrame, dict]:
    pids = cohort["patient_id"].tolist()
    y = cohort["M1_label"].astype(int).to_numpy()
    sites = cohort["site"].to_numpy()
    site_counts = Counter(sites)
    small_sites = {s for s, c in site_counts.items() if c < min_site_for_val}
    cv_mask = np.array([s not in small_sites for s in sites])
    cv_idx = np.where(cv_mask)[0]
    small_train = np.where(~cv_mask)[0]

    rows: list[dict[str, Any]] = []
    val_seen_by_rep: dict[int, set[str]] = defaultdict(set)
    for rep in range(n_repeats):
        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for fold, (tr_cv, va_cv) in enumerate(kf.split(np.zeros(len(cv_idx)), y[cv_idx], sites[cv_idx])):
            train_idx = np.concatenate([cv_idx[tr_cv], small_train])
            val_idx = cv_idx[va_cv]
            for i in train_idx:
                rows.append({"repeat": rep, "fold": fold, "split": "train", "patient_id": pids[i], "site": sites[i], "label": int(y[i]), "small_site_train_only": bool(i in set(small_train))})
            for i in val_idx:
                rows.append({"repeat": rep, "fold": fold, "split": "val", "patient_id": pids[i], "site": sites[i], "label": int(y[i]), "small_site_train_only": False})
                val_seen_by_rep[rep].add(pids[i])

    folds = pd.DataFrame(rows).sort_values(["repeat", "fold", "split", "site", "patient_id"])
    val = folds[folds["split"] == "val"]
    patient_val_counts = val.groupby(["repeat", "patient_id"]).size()
    site_val_counts = val.groupby(["repeat", "site"])["fold"].nunique()
    fold_summary = {
        "status": "reconstructed_from_current_289_oof_patients",
        "warning": "Original fold assignment files were not saved; this uses the original seed and code policy but cannot prove exact training-time ordering without clinical.csv.",
        "n_patients": int(len(cohort)),
        "n_sites": int(cohort["site"].nunique()),
        "n_repeats": n_repeats,
        "n_folds": n_splits,
        "seed": seed,
        "min_site_for_val": min_site_for_val,
        "small_sites": sorted(small_sites),
        "n_small_site_patients_train_only_per_reconstructed_repeat": int((~cv_mask).sum()),
        "patients_missing_reconstructed_val": sorted(set(pids) - set(val["patient_id"])),
        "same_patient_crosses_val_folds_within_repeat": bool((patient_val_counts > 1).any()),
        "same_site_crosses_val_folds_within_repeat": bool((site_val_counts > 1).any()),
        "note_on_oof": "The 289-row OOF CSVs contain only patients with OOF predictions. Patients from small sites that were always train-only in original code would not appear in OOF if clinical.csv contained them.",
    }
    return folds, fold_summary


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5, n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    rng = np.random.default_rng(seed)
    aucs = []
    aps = []
    briers = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], score[idx]))
        aps.append(average_precision_score(y[idx], score[idx]))
        briers.append(brier_score_loss(y[idx], score[idx]))

    frac_pos, mean_pred = calibration_curve(y, score, n_bins=10, strategy="quantile")
    return {
        "n": int(n),
        "n_pos": int(y.sum()),
        "n_neg": int(n - y.sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc_average_precision": float(average_precision_score(y, score)),
        "bootstrap_95ci": {
            "n_boot": n_boot,
            "roc_auc": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "pr_auc_average_precision": [float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))],
            "brier": [float(np.percentile(briers, 2.5)), float(np.percentile(briers, 97.5))],
        },
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "ppv": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "brier": float(brier_score_loss(y, score)),
        "calibration": {
            "n_bins": int(len(frac_pos)),
            "mean_predicted_value": [float(x) for x in mean_pred],
            "fraction_of_positives": [float(x) for x in frac_pos],
            "calibration_intercept_slope": calibration_intercept_slope(y, score),
        },
    }


def calibration_intercept_slope(y: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    eps = 1e-6
    p = np.clip(score.astype(float), eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    try:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(penalty=None, solver="lbfgs")
        lr.fit(logit_p.reshape(-1, 1), y.astype(int))
        return {"intercept": float(lr.intercept_[0]), "slope": float(lr.coef_[0, 0])}
    except Exception:
        return {"intercept": None, "slope": None}


def m4_metrics(df: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    y = df["label"].astype(int).to_numpy()
    proba_raw = df[[f"prob_c{i}" for i in range(4)]].astype(float).to_numpy()
    row_sums = proba_raw.sum(axis=1, keepdims=True)
    proba = proba_raw / np.clip(row_sums, 1e-12, None)
    pred = proba.argmax(axis=1)
    out = {
        "n": int(len(y)),
        "classes": M4_CLASSES,
        "class_mapping": {str(i): c for i, c in enumerate(M4_CLASSES)},
        "label_dist": {M4_CLASSES[int(k)]: int(v) for k, v in Counter(y).items()},
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).astype(int).tolist(),
        "roc_auc_ovr_macro": float(roc_auc_score(y, proba, multi_class="ovr", average="macro")),
        "brier_multiclass_mean_squared": float(np.mean(np.sum((proba - np.eye(4)[y]) ** 2, axis=1))),
        "log_loss": float(log_loss(y, proba, labels=[0, 1, 2, 3])),
        "probability_note": "M4 CSV probabilities are rounded; metrics normalize rows before multiclass AUC/log-loss.",
        "raw_probability_row_sum_min": float(row_sums.min()),
        "raw_probability_row_sum_max": float(row_sums.max()),
    }
    rng = np.random.default_rng(seed)
    aucs, f1s, accs, briers = [], [], [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        yp = proba[idx].argmax(axis=1)
        try:
            aucs.append(roc_auc_score(y[idx], proba[idx], multi_class="ovr", average="macro", labels=[0, 1, 2, 3]))
        except ValueError:
            pass
        f1s.append(f1_score(y[idx], yp, average="macro", zero_division=0))
        accs.append(accuracy_score(y[idx], yp))
        briers.append(float(np.mean(np.sum((proba[idx] - np.eye(4)[y[idx]]) ** 2, axis=1))))
    out["bootstrap_95ci"] = {
        "n_boot": n_boot,
        "roc_auc_ovr_macro": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else [None, None],
        "macro_f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
        "accuracy": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "brier_multiclass_mean_squared": [float(np.percentile(briers, 2.5)), float(np.percentile(briers, 97.5))],
    }
    return out


def recompute_oof_metrics(n_boot: int, seed: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    binary = {
        "M1_immune_sensitive": RESULTS / "oof_preds_M1_immune_sensitive.csv",
        "M2_msi": RESULTS / "oof_preds_M2_msi.csv",
        "M3_ebv": RESULTS / "oof_preds_M3_ebv.csv",
        "M5_clinical": RESULTS / "oof_preds_M5_clinical.csv",
    }
    for i, (name, path) in enumerate(binary.items()):
        df = read_csv(path)
        m = binary_metrics(df["label"].to_numpy(), df["prob_c1"].to_numpy(), threshold=0.5, n_boot=n_boot, seed=seed + i)
        m["task"] = name
        m["source_csv"] = str(path)
        metrics[name] = m
        write_json(OUT / f"metrics_{name}_oof_recomputed.json", m)

    m4 = read_csv(RESULTS / "oof_preds_M4_subtype4.csv")
    m = m4_metrics(m4, n_boot=n_boot, seed=seed + 4)
    m["task"] = "M4_subtype4"
    m["source_csv"] = str(RESULTS / "oof_preds_M4_subtype4.csv")
    metrics["M4_subtype4"] = m
    write_json(OUT / "metrics_M4_subtype4_oof_recomputed.json", m)
    write_json(OUT / "metrics_M1-M5_oof_recomputed.json", metrics)
    return metrics


def write_m4_audit() -> dict[str, Any]:
    m4 = read_csv(RESULTS / "oof_preds_M4_subtype4.csv")
    m4 = m4.copy()
    m4["expected_label_from_subtype"] = m4["subtype"].map(M4_TO_ID).astype("Int64")
    m4["pred_class"] = m4[[f"prob_c{i}" for i in range(4)]].astype(float).to_numpy().argmax(axis=1)
    m4["pred_subtype"] = [M4_CLASSES[i] for i in m4["pred_class"]]
    m4["label_matches_mapping"] = m4["label"].astype("Int64").eq(m4["expected_label_from_subtype"])
    m4["prob_c0_EBV"] = m4["prob_c0"]
    m4["prob_c1_MSI"] = m4["prob_c1"]
    m4["prob_c2_GS"] = m4["prob_c2"]
    m4["prob_c3_CIN"] = m4["prob_c3"]
    cols = [
        "patient_id", "subtype", "label", "expected_label_from_subtype", "label_matches_mapping",
        "prob_c0_EBV", "prob_c1_MSI", "prob_c2_GS", "prob_c3_CIN", "pred_class", "pred_subtype",
    ]
    m4[cols].to_csv(OUT / "m4_class_mapping_audit.csv", index=False)

    cases = m4[m4["patient_id"].isin(["TCGA-VQ-AA69", "TCGA-VQ-AA6K", "TCGA-BR-7703"])][cols]
    audit = {
        "class_order_confirmed_from_src_feature_loader_and_train_multitask": M4_CLASSES,
        "mapping": {str(i): c for i, c in enumerate(M4_CLASSES)},
        "all_rows_match_subtype_mapping": bool(m4["label_matches_mapping"].all()),
        "n_mismatched_rows": int((~m4["label_matches_mapping"]).sum()),
        "case_check": cases.to_dict(orient="records"),
        "important_note": "M4 prob_c1 is MSI, not EBV. Any report interpreting AA69 prob_c1=0.9038 as EBV is inconsistent with the training class mapping.",
    }
    write_json(OUT / "m4_class_mapping_audit.json", audit)
    return audit


def write_agent_leakage_audit() -> dict[str, Any]:
    panel_path = RESULTS / "agent_panel_judgments.json"
    data = json.loads(panel_path.read_text(encoding="utf-8")) if panel_path.exists() else []
    findings = []
    for row in data:
        report = row.get("report", "")
        findings.append(
            {
                "patient_id": row.get("patient_id"),
                "final_pred": row.get("final_pred"),
                "m1_prob": row.get("m1_prob"),
                "confidence": row.get("agent_selection", {}).get("confidence"),
                "visual_module": row.get("visual_evidence"),
                "report_mentions_molecular_or_morphology_without_visual_support": any(x in report for x in ["EBV相关特征", "CIN亚型", "STAD_MSI", "肠型", "印戒"]),
                "report": report,
            }
        )
    audit = {
        "status": "development_demo_only",
        "reason": "run_agent_panel.py provided clinical subtype to the report prompt and current panel visual evidence is disabled.",
        "no_leakage_policy_for_next_run": [
            "remove clinical subtype, MSI, EBV, TCGA molecular subtype and labels from Agent input",
            "keep only patient_id, de-identified clinical variables allowed by analysis plan, and model probabilities",
            "do not emit TIL/MSI/CIN morphology statements until thumbnails/attention/cluster overlays are real inputs",
            "final prediction remains M1 fixed-threshold output; Agent may only report confidence and suggested confirmatory tests",
        ],
        "current_panel_cases": findings,
    }
    write_json(OUT / "agent_leakage_audit_and_no_leakage_plan.json", audit)
    return audit


def cptac_full_inference(device: str, force: bool) -> None:
    if not CPTAC_FEATURE_DIR.exists():
        return
    out_patient = OUT / "cptac_193_feature_inference" / "external_feature_patient_predictions.csv"
    if out_patient.exists() and not force:
        return
    cmd = [
        os.environ.get("PY", "/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python"),
        str(BASE / "eval_cptac_features.py"),
        "--feature_dir",
        str(CPTAC_FEATURE_DIR),
        "--model_dir",
        str(BASE / "models"),
        "--out_dir",
        str(OUT / "cptac_193_feature_inference"),
        "--device",
        device,
        "--pattern",
        "*",
    ]
    subprocess.check_call(cmd, cwd=str(BASE))


def build_cptac_outputs(n_boot: int, seed: int) -> dict[str, Any]:
    infer_dir = OUT / "cptac_193_feature_inference"
    if not infer_dir.exists():
        infer_dir = CPTAC_INFER_DIR
    pred_slide = read_csv(infer_dir / "external_feature_slide_predictions.csv")
    pred_patient = read_csv(infer_dir / "external_feature_patient_predictions.csv")
    pred_patient.to_csv(OUT / "cptac_patient_predictions_fixed_threshold.csv", index=False)
    pred_slide.to_csv(OUT / "cptac_slide_predictions_fixed_threshold.csv", index=False)

    labels = read_csv(CPTAC_LABELS) if CPTAC_LABELS.exists() else pd.DataFrame()
    manifest_path = CPTAC_BASE / "cptac_stad_50patients_all_svs.csv"
    manifest = read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()

    svs_files = sorted(CPTAC_BASE.glob("*.svs"))
    if svs_files:
        slide_status = pd.DataFrame(
            [
                {
                    "patient_id": "-".join(p.stem.split("-")[:2]),
                    "slide_id": p.stem,
                    "basename": p.name,
                    "path": str(p),
                    "size": p.stat().st_size,
                    "mtime": p.stat().st_mtime,
                }
                for p in svs_files
            ]
        )
        slide_status["manifest_source"] = "actual_svs_directory_scan"
    elif not manifest.empty:
        manifest["slide_id"] = manifest["basename"].str.replace(".svs", "", regex=False)
        slide_status = manifest[["patient_id", "slide_id", "basename", "path", "size", "mtime"]].copy()
        slide_status["manifest_source"] = str(manifest_path)
    else:
        slide_status = pd.DataFrame()

    if not slide_status.empty:
        feature_manifest = CPTAC_FEATURE_DIR / "feature_manifest.csv"
        if feature_manifest.exists():
            feat_df = pd.read_csv(feature_manifest)
            feature_stems = set(feat_df["slide_id"].astype(str))
        else:
            feature_stems = {p.stem for p in CPTAC_FEATURE_DIR.glob("*.pt")}
        feature_error_path = CPTAC_FEATURE_DIR / "feature_errors.json"
        feature_errors = {}
        if feature_error_path.exists():
            try:
                for row in json.loads(feature_error_path.read_text(encoding="utf-8")):
                    feature_errors[str(row.get("slide_id"))] = row.get("error", "")
            except Exception:
                feature_errors = {}
        pred_stems = set(pred_slide["slide_id"].astype(str))
        slide_status["feature_status"] = slide_status["slide_id"].map(lambda s: "feature_present" if s in feature_stems else "feature_missing")
        slide_status["prediction_status"] = slide_status["slide_id"].map(lambda s: "predicted" if s in pred_stems else "not_predicted")
        slide_status["feature_error"] = slide_status["slide_id"].map(lambda s: feature_errors.get(str(s), ""))

        def exclusion_reason(row: pd.Series) -> str:
            if row["prediction_status"] == "predicted":
                return ""
            if row["feature_error"]:
                if "OpenSlideUnsupportedFormatError" in str(row["feature_error"]):
                    return "openslide_unsupported_or_missing_image"
                return "feature_extraction_error"
            if row["feature_status"] == "feature_missing":
                return "feature_missing"
            return "feature_present_but_not_in_current_inference_output"

        slide_status["exclusion_reason"] = slide_status.apply(exclusion_reason, axis=1)
        slide_status.to_csv(OUT / "cptac_slide_cohort_and_exclusions.csv", index=False)

    patient = pred_patient.copy()
    if not labels.empty:
        labels = labels.copy()
        labels["subtype4_label_class"] = labels["subtype4"].map({"EBV": 0, "MSI": 1, "GS": 2, "CIN": 3})
        patient = patient.merge(labels, on="patient_id", how="left", suffixes=("", "_label"))
    patient["aggregation_rule"] = "mean of slide-level probabilities per patient; fixed threshold 0.5 for M1/M2/M3"
    patient.to_csv(OUT / "cptac_patient_cohort_labels_predictions_fixed_threshold.csv", index=False)

    metrics: dict[str, Any] = {
        "prediction_dir": str(infer_dir),
        "feature_dir": str(CPTAC_FEATURE_DIR),
        "n_feature_files": len(list(CPTAC_FEATURE_DIR.glob("*.pt"))) if CPTAC_FEATURE_DIR.exists() else 0,
        "n_predicted_slides": int(len(pred_slide)),
        "n_predicted_patients": int(pred_patient["patient_id"].nunique()),
        "threshold_policy": "fixed threshold 0.5; no CPTAC threshold tuning",
        "aggregation_rule": "patient probability = arithmetic mean of slide probabilities",
    }
    if not labels.empty:
        eval_df = patient.dropna(subset=["immune_sensitive", "msi", "ebv", "subtype4_label_class"]).copy()
        metrics["n_labeled_evaluable_patients_all_tasks"] = int(len(eval_df))
        tasks = {
            "M1_immune_sensitive": ("immune_sensitive", "immune_sensitive_prob"),
            "M2_msi": ("msi", "msi_prob"),
            "M3_ebv": ("ebv", "ebv_prob"),
        }
        metrics["tasks"] = {}
        for i, (name, (label, score)) in enumerate(tasks.items()):
            task_df = patient.dropna(subset=[label, score]).copy()
            task_df[f"{name}_fixed_pred"] = (task_df[score].astype(float) >= 0.5).astype(int)
            metrics["tasks"][name] = binary_metrics(
                task_df[label].astype(int).to_numpy(),
                task_df[score].astype(float).to_numpy(),
                threshold=0.5,
                n_boot=n_boot,
                seed=seed + 100 + i,
            )
        m4df = patient.dropna(subset=["subtype4_label_class"]).copy()
        if len(m4df):
            m4tmp = pd.DataFrame(
                {
                    "label": m4df["subtype4_label_class"].astype(int),
                    "prob_c0": m4df["M4_subtype4_prob_c0"],
                    "prob_c1": m4df["M4_subtype4_prob_c1"],
                    "prob_c2": m4df["M4_subtype4_prob_c2"],
                    "prob_c3": m4df["M4_subtype4_prob_c3"],
                }
            )
            metrics["tasks"]["M4_subtype4"] = m4_metrics(m4tmp, n_boot=n_boot, seed=seed + 104)
    write_json(OUT / "cptac_fixed_threshold_metrics_bootstrap.json", metrics)
    return metrics


def write_label_source_docs(cohort: pd.DataFrame) -> dict[str, Any]:
    label_doc = {
        "tcga_internal_289": {
            "status": "partially_reconstructed_from_oof",
            "available_source": "results/oof_preds_M1-M5.csv",
            "missing_source_file": "clinical.csv is not present in current checkout",
            "mapping_used_in_code": {
                "immune_sensitive/M1": "STAD_MSI or STAD_EBV -> 1; STAD_GS or STAD_CIN -> 0; POLE/NO_SUBTYPE/unknown excluded",
                "MSI/M2": "STAD_MSI -> 1; STAD_EBV/STAD_GS/STAD_CIN -> 0",
                "EBV/M3": "STAD_EBV -> 1; STAD_MSI/STAD_GS/STAD_CIN -> 0",
                "M4_subtype4": {"EBV": 0, "MSI": 1, "GS": 2, "CIN": 3},
            },
            "unknown_conflict_policy_from_code": "FeatureLoader drops values outside the explicit subtype mapping for each task.",
            "pole_status": "not recoverable from current OOF files; column set to NA in final_289_patient_cohort.csv",
        },
        "cptac_external": {
            "label_csv": str(CPTAC_LABELS),
            "source_fields": [
                "subtype4",
                "genomic_subtype_raw",
                "immune_sensitive",
                "msi",
                "ebv",
                "ebv_status",
                "ebv_counts",
                "msi_status",
                "msi_score",
                "source_file",
                "source_sheet",
            ],
            "source_file_recorded_in_labels": "1-s2.0-S2666379126001734-mmc2.xlsx / Derived_Features",
            "mapping": {"EBV": {"subtype4": 0, "immune_sensitive": 1, "ebv": 1, "msi": 0}, "MSI": {"subtype4": 1, "immune_sensitive": 1, "msi": 1, "ebv": 0}, "GS": {"subtype4": 2, "immune_sensitive": 0}, "CIN": {"subtype4": 3, "immune_sensitive": 0}},
            "unknown_conflict_policy": "Use passed_QC==1 tumor rows from local label builder output; rows without labels are retained in cohort tables but excluded from supervised metrics.",
            "label_file_sha256": sha256(CPTAC_LABELS),
        },
    }
    write_json(OUT / "label_sources_and_mapping.json", label_doc)
    return label_doc


def write_repro_files(args: argparse.Namespace) -> None:
    commit = run_text(["git", "rev-parse", "HEAD"]) or "unknown"
    status = run_text(["git", "status", "--short"]) or ""
    tracked = run_text(["git", "ls-files"]) or ""
    env = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "project_root": str(BASE),
        "git_commit": commit,
        "git_status_short": status,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "random_seed": args.seed,
        "n_boot": args.n_boot,
        "tracked_files_snapshot": tracked.splitlines(),
        "model_hashes": {p.name: sha256(p) for p in sorted((BASE / "models").glob("*.pt"))},
        "uni2h_weights_sha256": sha256(Path("/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/uni2-h-weights/pytorch_model.bin")),
        "notes": [
            "Use conda env gastric_msi_pathai for project scripts.",
            "Internal clinical.csv and TCGA h5 feature directory are absent from this checkout.",
            "M1-M4 models/*.pt are the last fold models saved by src/train_multitask.py, not per-fold weights or proven full-data refit weights.",
            "Use build_tcga_label_table.py when clinical.csv is restored to regenerate raw-to-model TCGA label tables.",
        ],
    }
    try:
        import sklearn
        import torch

        env["packages"] = {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
        }
    except Exception as e:
        env["package_probe_error"] = repr(e)
    write_json(OUT / "reproducibility_manifest.json", env)

    env_yml = f"""name: gastric_msi_pathai
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pytorch=2.4
  - torchvision
  - pytorch-cuda=12.1
  - numpy
  - pandas
  - scikit-learn
  - scipy
  - pillow
  - tqdm
  - h5py
  - openslide
  - pcre
  - lifelines
  - requests
  - pip
  - pip:
      - timm
"""
    (OUT / "environment_minimal.yml").write_text(env_yml, encoding="utf-8")


def write_readme(metrics: dict[str, Any], cptac_metrics: dict[str, Any], fold_summary: dict[str, Any], m4_audit: dict[str, Any]) -> None:
    lines = [
        "# First-stage Audit Package",
        "",
        f"Generated from commit `{run_text(['git', 'rev-parse', 'HEAD']) or 'unknown'}`.",
        "",
        "## Files",
        "- `final_289_patient_cohort.csv`: reconstructed TCGA 289-patient OOF cohort.",
        "- `cv_folds_reconstructed.csv`: reconstructed folds using original seed/policy.",
        "- `cv_fold_reconstruction_summary.json`: leakage checks for patient/site folds.",
        "- `metrics_M1-M5_oof_recomputed.json`: OOF metrics recomputed from the 289 rows.",
        "- `m4_class_mapping_audit.*`: confirms M4 class order.",
        "- `label_sources_and_mapping.json`: label source and mapping notes.",
        "- `agent_leakage_audit_and_no_leakage_plan.json`: current Agent caveats.",
        "- `cptac_*`: external CPTAC cohort, predictions, exclusions and fixed-threshold metrics.",
        "- `environment_minimal.yml` and `reproducibility_manifest.json`: minimal reproducibility metadata.",
        "- `../../build_tcga_label_table.py`: reusable script to regenerate TCGA label/cohort tables after `clinical.csv` is restored.",
        "",
        "## Key Caveats",
        "- `clinical.csv` and internal TCGA feature h5 files are absent from this checkout, so `slide_id`, `POLE`, and original TCGA raw label fields cannot be fully restored here.",
        "- CV folds are reconstructed from current 289 OOF patients. The original fold files were not saved.",
        "- M1-M4 `models/*.pt` are saved by the training script as the last fold model per task, not per-fold weights and not proven full-data retraining weights.",
        "- Existing Agent panel output is development-only because the prompt included true subtype and the visual module was disabled.",
        "",
        "## M4 Mapping",
        f"Class order: `{m4_audit['mapping']}`. Therefore `prob_c1` means MSI.",
        "",
        "## Fold Check",
        f"Same patient crosses validation folds within a repeat: `{fold_summary['same_patient_crosses_val_folds_within_repeat']}`.",
        f"Same site crosses validation folds within a repeat: `{fold_summary['same_site_crosses_val_folds_within_repeat']}`.",
        "",
        "## Recompute Commands",
        "```bash",
        "cd /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/tcga-stad",
        "/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python audit_first_stage.py --device cpu",
        "",
        "# After restoring clinical.csv, regenerate the raw TCGA label/cohort table:",
        "/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python build_tcga_label_table.py \\",
        "  --clinical_csv clinical.csv \\",
        "  --feature_dir tcga_stad_uni2h/TCGA-STAD/features",
        "```",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip_cptac_inference", action="store_true")
    p.add_argument("--force_cptac_inference", action="store_true")
    args = p.parse_args()

    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    cohort = build_tcga_cohort()
    folds, fold_summary = reconstruct_folds(cohort, seed=args.seed)
    val_assign = folds[folds["split"] == "val"][["repeat", "fold", "patient_id"]].copy()
    first_val = val_assign.sort_values(["repeat", "fold"]).drop_duplicates("patient_id")
    first_val = first_val.rename(columns={"repeat": "reconstructed_oof_repeat", "fold": "reconstructed_oof_fold"})
    cohort = cohort.merge(first_val, on="patient_id", how="left")
    cohort["fold"] = cohort["reconstructed_oof_fold"]
    cohort.to_csv(OUT / "final_289_patient_cohort.csv", index=False)
    folds.to_csv(OUT / "cv_folds_reconstructed.csv", index=False)
    write_json(OUT / "cv_fold_reconstruction_summary.json", fold_summary)

    flow = {
        "status": "partially_reconstructed",
        "reason": "clinical.csv is absent; only the final 289 OOF cohort can be enumerated exactly.",
        "steps": [
            {"step": "OOF patients with M1-M5 predictions", "n": int(len(cohort))},
            {"step": "M1 positive immune-sensitive", "n": int(cohort["M1_label"].sum())},
            {"step": "M1 negative non-sensitive", "n": int((1 - cohort["M1_label"].astype(int)).sum())},
            {"step": "M4 subtype EBV", "n": int(cohort["M4_subtype"].eq("EBV").sum())},
            {"step": "M4 subtype MSI", "n": int(cohort["M4_subtype"].eq("MSI").sum())},
            {"step": "M4 subtype GS", "n": int(cohort["M4_subtype"].eq("GS").sum())},
            {"step": "M4 subtype CIN", "n": int(cohort["M4_subtype"].eq("CIN").sum())},
        ],
        "cannot_reconstruct_from_raw_to_289_until_files_restored": ["clinical.csv", "tcga_stad_uni2h/TCGA-STAD/features"],
    }
    write_json(OUT / "case_flow_statistics.json", flow)

    metrics = recompute_oof_metrics(args.n_boot, args.seed)
    m4_audit = write_m4_audit()
    write_agent_leakage_audit()
    write_label_source_docs(cohort)
    write_repro_files(args)

    if not args.skip_cptac_inference:
        cptac_full_inference(args.device, args.force_cptac_inference)
    cptac_metrics = build_cptac_outputs(args.n_boot, args.seed)
    write_readme(metrics, cptac_metrics, fold_summary, m4_audit)
    print(f"Wrote first-stage audit package: {OUT}")


if __name__ == "__main__":
    main()
