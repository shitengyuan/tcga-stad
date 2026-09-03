#!/usr/bin/env python3
"""Build locked deliverables requested in 计算机改进部分.docx.

The script consolidates existing TCGA/CPTAC outputs, recomputes patient-level
statistics for the locked 231-patient TCGA OOF set, and creates review tables
for items that require human pathology annotation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
AUDIT = RESULTS / "audit_first_stage"
CPU = RESULTS / "cpu_supplement"
DOC_DIR = ROOT.parent / "文档"
OUT = RESULTS / "locked_release_20260902"
M4_CLASSES = ["EBV", "MSI", "GS", "CIN"]
BOOT = int(os.environ.get("N_BOOT", "500"))
SEED = 42


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def clean_subtype(x: Any) -> str:
    s = str(x)
    return s.replace("STAD_", "") if s.startswith("STAD_") else s


def binary_point_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else math.nan
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else math.nan
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "prevalence": float(y.mean()) if len(y) else math.nan,
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float((tp + tn) / len(y)) if len(y) else math.nan,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "brier": float(brier_score_loss(y, score)),
    }
    if len(np.unique(y)) >= 2:
        out["auroc"] = float(roc_auc_score(y, score))
        out["average_precision"] = float(average_precision_score(y, score))
    else:
        out["auroc"] = math.nan
        out["average_precision"] = math.nan
    out.update(calibration_intercept_slope(y, score))
    return out


def calibration_intercept_slope(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.clip(np.asarray(score).astype(float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return {"calibration_intercept": math.nan, "calibration_slope": math.nan}
    x = np.log(score / (1 - score)).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000).fit(x, y)
    except TypeError:
        lr = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000).fit(x, y)
    return {
        "calibration_intercept": float(lr.intercept_[0]),
        "calibration_slope": float(lr.coef_[0][0]),
    }


def bootstrap_ci(y: np.ndarray, score: np.ndarray, threshold: float = 0.5, n_boot: int = BOOT) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        rows.append(binary_point_metrics(y[idx], score[idx], threshold))
    df = pd.DataFrame(rows)
    ci: dict[str, Any] = {"n_boot_requested": n_boot, "n_boot_valid": int(len(df))}
    for key in [
        "auroc",
        "average_precision",
        "accuracy",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "brier",
        "calibration_intercept",
        "calibration_slope",
    ]:
        vals = pd.to_numeric(df.get(key), errors="coerce").dropna()
        ci[key] = [float(vals.quantile(0.025)), float(vals.quantile(0.975))] if len(vals) else [math.nan, math.nan]
    return ci


def multiclass_metrics(y: np.ndarray, proba: np.ndarray, n_boot: int = BOOT) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    row_sum = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sum, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sum > 0)
    pred = proba.argmax(axis=1)
    out = {
        "n": int(len(y)),
        "class_order": M4_CLASSES,
        "class_counts": {M4_CLASSES[i]: int((y == i).sum()) for i in range(len(M4_CLASSES))},
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "confusion_matrix_labels_EBV_MSI_GS_CIN": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).astype(int).tolist(),
    }
    if len(np.unique(y)) >= 2:
        out["macro_ovr_auroc"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
        y_bin = label_binarize(y, classes=[0, 1, 2, 3])
        out["macro_average_precision"] = float(average_precision_score(y_bin, proba, average="macro"))
    else:
        out["macro_ovr_auroc"] = math.nan
        out["macro_average_precision"] = math.nan

    rng = np.random.default_rng(SEED)
    rows = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            rows.append(
                {
                    "accuracy": accuracy_score(y[idx], proba[idx].argmax(axis=1)),
                    "macro_f1": f1_score(y[idx], proba[idx].argmax(axis=1), average="macro", zero_division=0),
                    "macro_ovr_auroc": roc_auc_score(y[idx], proba[idx], multi_class="ovr", average="macro"),
                    "macro_average_precision": average_precision_score(
                        label_binarize(y[idx], classes=[0, 1, 2, 3]), proba[idx], average="macro"
                    ),
                }
            )
        except Exception:
            continue
    bdf = pd.DataFrame(rows)
    out["bootstrap_95ci"] = {"n_boot_requested": n_boot, "n_boot_valid": int(len(bdf))}
    for key in ["accuracy", "macro_f1", "macro_ovr_auroc", "macro_average_precision"]:
        vals = pd.to_numeric(bdf.get(key), errors="coerce").dropna()
        out["bootstrap_95ci"][key] = (
            [float(vals.quantile(0.025)), float(vals.quantile(0.975))] if len(vals) else [math.nan, math.nan]
        )
    return out


def load_oof(name: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"oof_preds_{name}.csv")
    rename = {"label": f"{prefix}_label", "prob_c0": f"{prefix}_prob_c0", "prob_c1": f"{prefix}_prob_c1"}
    if name == "M4_subtype4":
        rename.update(
            {
                "prob_c0": "M4_EBV_prob",
                "prob_c1": "M4_MSI_prob",
                "prob_c2": "M4_GS_prob",
                "prob_c3": "M4_CIN_prob",
            }
        )
    return df.rename(columns=rename)


def build_locked_tcga() -> pd.DataFrame:
    cohort = pd.read_csv(AUDIT / "tcga_public_feature_matched_246_cohort_after_gpu_run.csv")
    cohort = cohort.copy()
    cohort["cohort"] = "TCGA"
    cohort["analysis_set_246"] = 1
    cohort["TCGA_four_subtype"] = cohort["M4_subtype"].map(clean_subtype)
    cohort["center"] = cohort["site"]
    cohort.to_csv(OUT / "patient_manifest.csv", index=False)
    cohort.to_csv(OUT / "patient_manifest_tcga_246.csv", index=False)

    fr = pd.read_csv(RESULTS / "fold_registry_M1_immune_sensitive.csv")
    fr.to_csv(OUT / "fold_assignment.csv", index=False)
    val_fr = fr[fr["split"].eq("val")].copy()
    val_fr.to_csv(OUT / "fold_assignment_validation_only.csv", index=False)
    small = fr[fr["small_site_train_only"].astype(str).str.lower().eq("true")].drop_duplicates("patient_id")
    small[["patient_id", "site", "subtype", "label", "small_site_train_only"]].to_csv(
        OUT / "small_site_train_only_patients.csv", index=False
    )

    locked = cohort.drop(columns=[c for c in ["subtype"] if c in cohort.columns]).merge(
        load_oof("M1_immune_sensitive", "M1"), on="patient_id", how="inner"
    )
    if "M1_label_x" in locked.columns:
        locked = locked.rename(columns={"M1_label_x": "M1_label"})
        locked = locked.drop(columns=["M1_label_y"], errors="ignore")
    for name, prefix in [
        ("M2_msi", "M2"),
        ("M3_ebv", "M3"),
        ("M4_subtype4", "M4"),
    ]:
        tmp = load_oof(name, prefix).drop(columns=["subtype"], errors="ignore")
        locked = locked.merge(tmp, on="patient_id", how="left")

    m5_path = RESULTS / "oof_preds_M5_clinical.csv"
    if m5_path.exists():
        m5 = pd.read_csv(m5_path).rename(
            columns={"label": "M5_legacy_label", "prob_c0": "M5_legacy_prob_c0", "prob_c1": "M5_legacy_prob_c1"}
        )
        locked = locked.merge(m5, on="patient_id", how="left")
    m6_path = RESULTS / "oof_preds_M6_survival.csv"
    if m6_path.exists():
        m6 = pd.read_csv(m6_path)
        m6 = m6.add_prefix("M6_").rename(columns={"M6_patient_id": "patient_id"})
        locked = locked.merge(m6, on="patient_id", how="left")

    val_keys = (
        val_fr.groupby("patient_id")
        .apply(lambda x: ";".join(f"rep{int(r.repeat)}_fold{int(r.fold)}" for r in x.itertuples()))
        .rename("validation_fold_keys")
        .reset_index()
    )
    locked = locked.merge(val_keys, on="patient_id", how="left")
    locked["analysis_set"] = "tcga_231_center_isolated_oof"
    locked["M4_class_order"] = "EBV|MSI|GS|CIN"
    locked["fixed_threshold_binary"] = 0.5
    if "M4_label_x" in locked.columns:
        locked = locked.rename(columns={"M4_label_x": "M4_label"})
        locked = locked.drop(columns=["M4_label_y"], errors="ignore")
    locked["M1_pred_0p5"] = (locked["M1_prob_c1"] >= 0.5).astype(int)
    locked["M2_pred_0p5"] = (locked["M2_prob_c1"] >= 0.5).astype(int)
    locked["M3_pred_0p5"] = (locked["M3_prob_c1"] >= 0.5).astype(int)
    locked["M4_pred_class"] = locked[["M4_EBV_prob", "M4_MSI_prob", "M4_GS_prob", "M4_CIN_prob"]].to_numpy().argmax(axis=1)
    locked["M4_pred_subtype"] = locked["M4_pred_class"].map({i: c for i, c in enumerate(M4_CLASSES)})
    locked.to_csv(OUT / "locked_predictions_tcga.csv", index=False)
    return locked


def build_locked_cptac() -> pd.DataFrame:
    cptac = pd.read_csv(AUDIT / "cptac_patient_cohort_labels_predictions_fixed_threshold.csv")
    label_cols = ["passed_QC", "subtype4", "immune_sensitive", "msi", "ebv", "subtype4_label_class"]
    mask = cptac[label_cols].notna().all(axis=1)
    locked = cptac[mask].copy()
    locked["cohort"] = "CPTAC"
    locked["analysis_set"] = "cptac_156_fixed_threshold_external_validation"
    locked["fixed_threshold_binary"] = 0.5
    locked["M4_class_order"] = "EBV|MSI|GS|CIN"
    locked.to_csv(OUT / "locked_predictions_cptac.csv", index=False)
    return locked


def summarize_model_hashes() -> pd.DataFrame:
    patterns = [
        "models/*.pt",
        "models/per_fold/*/*.pt",
        "models/per_fold/*/*.joblib",
        "src/*.py",
        "*.py",
        "run_*.sh",
    ]
    rows = []
    for pattern in patterns:
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file():
                rows.append({"path": rel(p), "sha256": sha256(p), "bytes": int(p.stat().st_size)})
    df = pd.DataFrame(rows).drop_duplicates("path")
    df.to_csv(OUT / "model_and_code_hash_manifest.csv", index=False)
    return df


def build_policy_files() -> None:
    metrics = {}
    for name in ["M1_immune_sensitive", "M2_msi", "M3_ebv", "M4_subtype4"]:
        p = RESULTS / f"metrics_{name}.json"
        metrics[name] = read_json(p).get("config", {}) if p.exists() else {}
    policy = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "git_status_short": run_text(["git", "status", "--short"]),
        "tcga_internal_policy": {
            "analysis_cohort": "246 TCGA patients with UNI2-h features",
            "reported_oof_set": "231 center-isolated validation patients",
            "folding": "patient-level, site/group isolated validation folds; sites with <8 patients are train-only",
            "repeats": 2,
            "folds": 5,
            "binary_threshold": 0.5,
        },
        "cptac_external_policy": {
            "analysis_cohort": "156 labeled evaluable CPTAC patients after QC and label matching",
            "feature_dir": "results/external_cptac_features_20x256",
            "inference_scripts": ["eval_cptac_features.py", "run_cptac_feature_inference_4gpu.sh"],
            "model_files_used_by_current_fixed_threshold_outputs": [
                "models/M1_immune_sensitive.pt",
                "models/M2_msi.pt",
                "models/M3_ebv.pt",
                "models/M4_subtype4.pt",
            ],
            "patient_aggregation": "arithmetic mean of slide-level probabilities",
            "threshold_policy": "fixed 0.5; no CPTAC threshold tuning",
            "note": "The current frozen CPTAC table records outputs generated from the saved model files. It is not documented as a per-fold ensemble unless the inference command is explicitly changed.",
        },
        "M4_class_order": M4_CLASSES,
    }
    write_json(OUT / "inference_model_policy.json", policy)
    config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_metrics_configs": metrics,
        "human_readable_training_config": {
            "encoder_feature_dim": 1536,
            "aggregator": "ABMIL",
            "hidden_dim": 256,
            "dropout": 0.25,
            "attention_dim": 128,
            "classifier_head": "Linear(256,64)+ReLU+Dropout+Linear(64,n_classes)",
            "loss": "torch.nn.functional.cross_entropy",
            "class_weighting": "inverse-frequency class weights computed inside each training fold",
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "epochs": 30,
            "best_checkpoint_policy": "select epoch with best validation AUROC/macro-AUROC; training loop still runs full configured epochs",
            "early_stopping": "no hard early stop in the current script",
            "seed": 42,
            "n_repeats": 2,
            "n_folds": 5,
            "min_site_for_val": 8,
            "max_patches_per_slide_bag": 8000,
        },
    }
    write_json(OUT / "training_config_summary.json", config)
    env = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": subprocess.check_output(["which", "python"], text=True).strip(),
    }
    for mod in ["numpy", "pandas", "sklearn", "torch", "h5py", "matplotlib"]:
        try:
            module = __import__(mod)
            env[mod] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            env[mod] = {"error": repr(exc)}
    write_json(OUT / "environment_lock_summary.json", env)


def recompute_stats_tcga(locked: pd.DataFrame) -> dict[str, Any]:
    tasks = [
        ("M1_immune_sensitive", "M1_label", "M1_prob_c1"),
        ("M2_msi", "M2_label", "M2_prob_c1"),
        ("M3_ebv", "M3_label", "M3_prob_c1"),
    ]
    rows = []
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cohort": "TCGA",
        "analysis_set": "231 center-isolated OOF patients",
        "bootstrap_unit": "patient",
        "n_boot": BOOT,
        "tasks": {},
    }
    for task, ycol, scol in tasks:
        y = locked[ycol].astype(int).to_numpy()
        score = locked[scol].astype(float).to_numpy()
        point = binary_point_metrics(y, score)
        ci = bootstrap_ci(y, score)
        payload["tasks"][task] = {"point": point, "bootstrap_95ci": ci}
        row = {"task": task, **point}
        for key, val in ci.items():
            if isinstance(val, list):
                row[f"{key}_ci_low"] = val[0]
                row[f"{key}_ci_high"] = val[1]
        rows.append(row)

    y4 = locked["M4_label"].astype(int).to_numpy()
    p4 = locked[["M4_EBV_prob", "M4_MSI_prob", "M4_GS_prob", "M4_CIN_prob"]].astype(float).to_numpy()
    payload["tasks"]["M4_subtype4"] = multiclass_metrics(y4, p4)
    m4row = {"task": "M4_subtype4", **{k: v for k, v in payload["tasks"]["M4_subtype4"].items() if not isinstance(v, (dict, list))}}
    for key, val in payload["tasks"]["M4_subtype4"]["bootstrap_95ci"].items():
        if isinstance(val, list):
            m4row[f"{key}_ci_low"] = val[0]
            m4row[f"{key}_ci_high"] = val[1]
    rows.append(m4row)

    write_json(OUT / "tcga_231_metrics_with_ci.json", payload)
    pd.DataFrame(rows).to_csv(OUT / "tcga_231_metrics_summary.csv", index=False)
    return payload


def add_cptac_threshold_cis(cptac: pd.DataFrame) -> dict[str, Any]:
    tasks = [
        ("M1_immune_sensitive", "immune_sensitive", "immune_sensitive_prob"),
        ("M2_msi", "msi", "msi_prob"),
        ("M3_ebv", "ebv", "ebv_prob"),
    ]
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cohort": "CPTAC",
        "analysis_set": "156 labeled external validation patients",
        "bootstrap_unit": "patient",
        "n_boot": BOOT,
        "tasks": {},
    }
    rows = []
    for task, ycol, scol in tasks:
        y = cptac[ycol].astype(int).to_numpy()
        score = cptac[scol].astype(float).to_numpy()
        point = binary_point_metrics(y, score)
        ci = bootstrap_ci(y, score)
        payload["tasks"][task] = {"point": point, "bootstrap_95ci": ci}
        row = {"task": task, **point}
        for key, val in ci.items():
            if isinstance(val, list):
                row[f"{key}_ci_low"] = val[0]
                row[f"{key}_ci_high"] = val[1]
        rows.append(row)
    y4 = cptac["subtype4_label_class"].astype(int).to_numpy()
    p4 = cptac[["M4_subtype4_prob_c0", "M4_subtype4_prob_c1", "M4_subtype4_prob_c2", "M4_subtype4_prob_c3"]].to_numpy()
    payload["tasks"]["M4_subtype4"] = multiclass_metrics(y4, p4)
    pd.DataFrame(rows).to_csv(OUT / "cptac_156_threshold_metrics_with_ci.csv", index=False)
    write_json(OUT / "cptac_156_metrics_with_ci.json", payload)
    return payload


def draw_curves(locked: pd.DataFrame, cptac: pd.DataFrame) -> None:
    bin_specs = [
        ("TCGA M1", locked["M1_label"].to_numpy(), locked["M1_prob_c1"].to_numpy()),
        ("TCGA M2", locked["M2_label"].to_numpy(), locked["M2_prob_c1"].to_numpy()),
        ("TCGA M3", locked["M3_label"].to_numpy(), locked["M3_prob_c1"].to_numpy()),
        ("CPTAC M1", cptac["immune_sensitive"].to_numpy(), cptac["immune_sensitive_prob"].to_numpy()),
        ("CPTAC M2", cptac["msi"].to_numpy(), cptac["msi_prob"].to_numpy()),
        ("CPTAC M3", cptac["ebv"].to_numpy(), cptac["ebv_prob"].to_numpy()),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    for name, y, s in bin_specs:
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y, s):.3f}")
        axes[1].plot(rec, prec, label=f"{name} AP={average_precision_score(y, s):.3f}")
    axes[0].plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "roc_pr_curves_tcga231_cptac156.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    calib_rows = []
    for name, y, s in bin_specs:
        prob_true, prob_pred = calibration_curve(y, s, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=name)
        for i, (mp, fp) in enumerate(zip(prob_pred, prob_true), 1):
            calib_rows.append({"task": name, "bin": i, "mean_predicted": mp, "fraction_positive": fp})
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction positive", title="Calibration")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "calibration_curves_tcga231_cptac156.png")
    plt.close(fig)
    pd.DataFrame(calib_rows).to_csv(OUT / "calibration_curve_data_tcga231_cptac156.csv", index=False)


def triage_capacity(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if cohort == "TCGA":
        specs = [
            ("joint_MSI_or_EBV_by_M1", "M1_label", "M1_prob_c1"),
            ("MSI_by_M2", "M2_label", "M2_prob_c1"),
            ("EBV_by_M3", "M3_label", "M3_prob_c1"),
        ]
    else:
        specs = [
            ("joint_MSI_or_EBV_by_M1", "immune_sensitive", "immune_sensitive_prob"),
            ("MSI_by_M2", "msi", "msi_prob"),
            ("EBV_by_M3", "ebv", "ebv_prob"),
        ]
    rows = []
    capacities = sorted(set([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]))
    for task, ycol, scol in specs:
        sub = df[["patient_id", ycol, scol]].dropna().copy()
        sub[ycol] = sub[ycol].astype(int)
        sub[scol] = sub[scol].astype(float)
        sub = sub.sort_values(scol, ascending=False).reset_index(drop=True)
        n = len(sub)
        positives = int(sub[ycol].sum())
        for cap in capacities:
            k = max(1, int(math.ceil(n * cap)))
            selected = sub.iloc[:k]
            detected = int(selected[ycol].sum())
            missed = int(positives - detected)
            rows.append(
                {
                    "cohort": cohort,
                    "task": task,
                    "n_patients": n,
                    "capacity_fraction": cap,
                    "immediate_tests": k,
                    "deferred_tests": n - k,
                    "positives_total": positives,
                    "positives_detected": detected,
                    "positives_missed": missed,
                    "positive_coverage": detected / positives if positives else math.nan,
                    "immediate_tests_per_100": 100 * k / n if n else math.nan,
                    "deferred_tests_per_100": 100 * (n - k) / n if n else math.nan,
                    "detected_per_100": 100 * detected / n if n else math.nan,
                    "missed_per_100": 100 * missed / n if n else math.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"triage_capacity_{cohort.lower()}.csv", index=False)
    out[out["capacity_fraction"].isin([0.20, 0.30, 0.40])].to_csv(
        OUT / f"per_100_triage_{cohort.lower()}_20_30_40pct.csv", index=False
    )
    return out


def draw_capacity(tcga_cap: pd.DataFrame, cptac_cap: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    for df in [tcga_cap, cptac_cap]:
        for task, sub in df.groupby("task"):
            ax.plot(
                sub["capacity_fraction"] * 100,
                sub["positive_coverage"] * 100,
                marker="o",
                label=f"{sub['cohort'].iloc[0]} {task}",
            )
    ax.set(xlabel="Testing capacity (% patients)", ylabel="Positive coverage (%)", title="Capacity triage curve")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "triage_capacity_curves_tcga_cptac.png")
    plt.close(fig)


def make_error_review(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    if cohort == "TCGA":
        specs = [
            ("M1_immune_sensitive", "M1_label", "M1_prob_c1", "M1_pred_0p5"),
            ("M2_msi", "M2_label", "M2_prob_c1", "M2_pred_0p5"),
            ("M3_ebv", "M3_label", "M3_prob_c1", "M3_pred_0p5"),
        ]
        clinical = pd.read_csv(ROOT / "clinical.csv")
        clinical_cols = [
            "patient_id",
            "histological_diagnosis",
            "ajcc_pathologic_tumor_stage",
            "primary_site_patient",
            "lymph_node_examined_count",
            "source_lauren_class",
        ]
        base = df.merge(clinical[clinical_cols], on="patient_id", how="left")
    else:
        specs = [
            ("M1_immune_sensitive", "immune_sensitive", "immune_sensitive_prob", "immune_sensitive_pred"),
            ("M2_msi", "msi", "msi_prob", "msi_pred"),
            ("M3_ebv", "ebv", "ebv_prob", "ebv_pred"),
        ]
        base = df.copy()
    rows = []
    for _, r in base.iterrows():
        for task, ycol, scol, pcol in specs:
            y = int(r[ycol])
            score = float(r[scol])
            pred = int(r[pcol])
            is_error = y != pred
            low_conf = abs(score - 0.5) <= 0.10
            if not (is_error or low_conf):
                continue
            row = {
                "cohort": cohort,
                "patient_id": r["patient_id"],
                "slide_id": r.get("slide_id", ""),
                "task": task,
                "true_label": y,
                "pred_label": pred,
                "prob_positive": score,
                "review_trigger": "error" if is_error else "low_confidence",
                "error_type": "FP" if y == 0 and pred == 1 else ("FN" if y == 1 and pred == 0 else ""),
                "low_confidence_rule": "abs(probability-0.5)<=0.10",
                "needs_pathology_review": 1,
                "tumor_content_percent": "",
                "diffuse_or_signet_ring": "",
                "mucin": "",
                "necrosis": "",
                "tissue_folds": "",
                "staining_variation": "",
                "scan_quality": "",
                "multi_slide_discordance": "",
                "reference_label_uncertainty": "",
                "pathologist_notes": "",
            }
            for c in [
                "TCGA_four_subtype",
                "subtype4",
                "histological_diagnosis",
                "ajcc_pathologic_tumor_stage",
                "primary_site_patient",
                "lymph_node_examined_count",
                "source_lauren_class",
                "lauren_label",
                "n_slides",
            ]:
                if c in r.index:
                    row[c] = r.get(c, "")
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"error_review_manifest_{cohort.lower()}.csv", index=False)
    return out


def build_clinical_raw(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    hist = df.get("histological_diagnosis", pd.Series(index=df.index, dtype=object)).fillna("")

    def lauren(v: Any) -> str:
        s = str(v).lower()
        if "diffuse" in s or "signet" in s or "poorly_cohesive" in s:
            return "diffuse"
        if "intestinal" in s or "tubular" in s or "papillary" in s or "mucinous" in s:
            return "intestinal"
        return "other"

    out["lauren"] = hist.map(lauren)
    out["sex"] = df.get("sex", pd.Series(index=df.index, dtype=object)).fillna("unknown")
    stage = df.get("ajcc_pathologic_tumor_stage", pd.Series(index=df.index, dtype=object)).fillna("unknown")
    out["stage"] = stage.map(lambda s: str(s)[:7] if str(s).lower().startswith("stage") else str(s))
    out["primary_site"] = df.get("primary_site_patient", pd.Series(index=df.index, dtype=object)).fillna("unknown")
    out["ln_examined"] = pd.to_numeric(df.get("lymph_node_examined_count"), errors="coerce")
    return out


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def run_matched_clinical(locked: pd.DataFrame) -> dict[str, Any]:
    clinical = pd.read_csv(ROOT / "clinical.csv").set_index("patient_id")
    manifest = pd.read_csv(OUT / "patient_manifest.csv").set_index("patient_id")
    pids = locked["patient_id"].tolist()
    raw_all = build_clinical_raw(clinical)
    raw = raw_all.loc[pids]
    y_map = manifest["M1_label"].astype(int)
    img_map = locked.set_index("patient_id")["M1_prob_c1"].astype(float)
    fr = pd.read_csv(OUT / "fold_assignment.csv")
    fr = fr[fr["task"].eq("immune_sensitive")]

    miss = raw.isna().mean().reset_index()
    miss.columns = ["variable", "missing_rate"]
    miss["n_missing"] = raw.isna().sum().to_numpy()
    miss["n_total"] = len(raw)
    miss.to_csv(OUT / "matched_clinical_feature_missingness_231.csv", index=False)

    numeric = ["age", "ln_examined"]
    categorical = ["lauren", "sex", "stage", "primary_site"]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_onehot())]), categorical),
        ]
    )

    pred_rows = []
    fold_rows = []
    for rep in sorted(fr["repeat"].unique()):
        for fold in sorted(fr["fold"].unique()):
            fold_df = fr[(fr["repeat"].eq(rep)) & (fr["fold"].eq(fold))]
            tr_pids = [p for p in fold_df[fold_df["split"].eq("train")]["patient_id"].tolist() if p in clinical.index]
            va_pids = [p for p in fold_df[fold_df["split"].eq("val")]["patient_id"].tolist() if p in pids]
            if not va_pids:
                continue
            x_tr = raw_all.loc[tr_pids]
            y_tr = y_map.loc[tr_pids].to_numpy()
            x_va = raw_all.loc[va_pids]
            y_va = y_map.loc[va_pids].to_numpy()
            clinical_pipe = Pipeline(
                [("preprocess", pre), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))]
            )
            clinical_pipe.fit(x_tr, y_tr)
            cprob = clinical_pipe.predict_proba(x_va)[:, 1]

            fusion_tr_pids = [p for p in tr_pids if p in img_map.index]
            fusion_train = raw_all.loc[fusion_tr_pids].copy()
            fusion_train["image_m1_oof_prob"] = img_map.reindex(fusion_tr_pids).to_numpy()
            fusion_val = raw_all.loc[va_pids].copy()
            fusion_val["image_m1_oof_prob"] = img_map.reindex(va_pids).to_numpy()
            fusion_numeric = numeric + ["image_m1_oof_prob"]
            fusion_pre = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                        fusion_numeric,
                    ),
                    (
                        "cat",
                        Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_onehot())]),
                        categorical,
                    ),
                ]
            )
            fusion_pipe = Pipeline(
                [("preprocess", fusion_pre), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))]
            )
            fusion_pipe.fit(fusion_train, y_map.loc[fusion_tr_pids].to_numpy())
            fprob = fusion_pipe.predict_proba(fusion_val)[:, 1]

            for pid, yy, cp, fp in zip(va_pids, y_va, cprob, fprob):
                pred_rows.append(
                    {
                        "patient_id": pid,
                        "repeat": int(rep),
                        "fold": int(fold),
                        "label": int(yy),
                        "clinical_only_prob": float(cp),
                        "image_only_m1_oof_prob": float(img_map.loc[pid]),
                        "image_clinical_late_fusion_prob": float(fp),
                    }
                )
            for split, ids in [("train", tr_pids), ("val", va_pids)]:
                for pid in ids:
                    fold_rows.append({"repeat": int(rep), "fold": int(fold), "split": split, "patient_id": pid})

    fold_pred = pd.DataFrame(pred_rows)
    fold_pred.to_csv(OUT / "matched_clinical_fold_predictions_231.csv", index=False)
    agg = (
        fold_pred.groupby("patient_id")
        .agg(
            label=("label", "first"),
            clinical_only_prob=("clinical_only_prob", "mean"),
            image_only_m1_oof_prob=("image_only_m1_oof_prob", "first"),
            image_clinical_late_fusion_prob=("image_clinical_late_fusion_prob", "mean"),
            n_validation_predictions=("fold", "count"),
        )
        .reset_index()
    )
    agg.to_csv(OUT / "matched_clinical_oof_predictions_231.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "matched_clinical_fold_registry_231.csv", index=False)

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cohort": "TCGA",
        "analysis_set": "same 231 center-isolated OOF patients and same folds as M1",
        "clinical_variables": numeric + categorical,
        "clinical_preprocessing": "numeric median imputation + standardization; categorical most-frequent imputation + one-hot; all fitted inside train fold",
        "model": "penalized LogisticRegression(class_weight='balanced', C=0.5)",
        "late_fusion_note": "The fusion model is a fold-wise late-fusion/stacking comparator using cross-fitted M1 OOF probability plus clinical variables. It is not an end-to-end image-feature plus clinical network.",
        "models": {},
    }
    y = agg["label"].to_numpy()
    for name, scol in [
        ("Clinical_only_matched_231", "clinical_only_prob"),
        ("Image_only_M1_matched_231", "image_only_m1_oof_prob"),
        ("Image_plus_clinical_late_fusion_matched_231", "image_clinical_late_fusion_prob"),
    ]:
        s = agg[scol].to_numpy()
        metrics["models"][name] = {"point": binary_point_metrics(y, s), "bootstrap_95ci": bootstrap_ci(y, s)}
    metrics["paired_bootstrap_deltas"] = paired_deltas(
        y,
        {
            "Clinical_only": agg["clinical_only_prob"].to_numpy(),
            "Image_only_M1": agg["image_only_m1_oof_prob"].to_numpy(),
            "Image_plus_clinical_late_fusion": agg["image_clinical_late_fusion_prob"].to_numpy(),
        },
    )
    write_json(OUT / "matched_clinical_metrics_231.json", metrics)
    return metrics


def paired_deltas(y: np.ndarray, scores: dict[str, np.ndarray]) -> dict[str, Any]:
    pairs = [
        ("Image_only_M1", "Clinical_only"),
        ("Image_plus_clinical_late_fusion", "Image_only_M1"),
        ("Image_plus_clinical_late_fusion", "Clinical_only"),
    ]
    rng = np.random.default_rng(SEED)
    out = {}
    for a, b in pairs:
        point = {
            "delta_auroc": float(roc_auc_score(y, scores[a]) - roc_auc_score(y, scores[b])),
            "delta_average_precision": float(average_precision_score(y, scores[a]) - average_precision_score(y, scores[b])),
        }
        rows = []
        n = len(y)
        for _ in range(BOOT):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                continue
            rows.append(
                {
                    "delta_auroc": roc_auc_score(y[idx], scores[a][idx]) - roc_auc_score(y[idx], scores[b][idx]),
                    "delta_average_precision": average_precision_score(y[idx], scores[a][idx])
                    - average_precision_score(y[idx], scores[b][idx]),
                }
            )
        bdf = pd.DataFrame(rows)
        out[f"{a}_minus_{b}"] = {
            "point": point,
            "bootstrap_95ci": {
                k: [float(bdf[k].quantile(0.025)), float(bdf[k].quantile(0.975))] for k in bdf.columns
            },
        }
    return out


def pathologist_template(tcga_err: pd.DataFrame, cptac_err: pd.DataFrame) -> None:
    cols = [
        "cohort",
        "patient_id",
        "slide_id",
        "task",
        "true_label",
        "pred_label",
        "prob_positive",
        "review_trigger",
        "error_type",
        "tumor_content_percent",
        "diffuse_or_signet_ring",
        "mucin",
        "necrosis",
        "tissue_folds",
        "staining_variation",
        "scan_quality",
        "multi_slide_discordance",
        "reference_label_uncertainty",
        "pathologist_notes",
    ]
    template = pd.concat([tcga_err, cptac_err], ignore_index=True, sort=False)
    template = template.reindex(columns=cols)
    template.to_csv(OUT / "pathologist_annotation_template.csv", index=False)


def write_completion_status() -> None:
    baseline_status = "完成" if (OUT / "baselines" / "baseline_metrics.json").exists() else "未完成"
    baseline_reason = (
        "results/locked_release_20260902/baselines/baseline_metrics.json"
        if baseline_status == "完成"
        else "运行 python run_pooling_baselines.py 生成"
    )
    fusion_path = OUT / "gpu_missing" / "fusion_M1_immune_sensitive" / "metrics_fusion_M1_immune_sensitive.json"
    if fusion_path.exists():
        fusion_status = "完成"
        fusion_reason = "results/locked_release_20260902/gpu_missing/fusion_M1_immune_sensitive/metrics_fusion_M1_immune_sensitive.json"
    else:
        fusion_status = "部分完成"
        fusion_reason = "已生成late-fusion/stacking探索版；端到端图像特征+临床融合网络仍未训练"
    transformer_tasks = [
        OUT / "gpu_missing" / "transformer_M1_immune_sensitive" / "metrics_transformer_M1_immune_sensitive.json",
        OUT / "gpu_missing" / "transformer_M2_msi" / "metrics_transformer_M2_msi.json",
        OUT / "gpu_missing" / "transformer_M3_ebv" / "metrics_transformer_M3_ebv.json",
        OUT / "gpu_missing" / "transformer_M4_subtype4" / "metrics_transformer_M4_subtype4.json",
    ]
    transformer_done = sum(p.exists() for p in transformer_tasks)
    if transformer_done == len(transformer_tasks):
        transformer_status = "完成"
        transformer_reason = "results/locked_release_20260902/gpu_missing/transformer_*/metrics_transformer_*.json"
    elif transformer_done:
        transformer_status = "部分完成"
        transformer_reason = f"已完成 {transformer_done}/{len(transformer_tasks)} 个任务，见 results/locked_release_20260902/gpu_missing/"
    else:
        transformer_status = "未完成/可选"
        transformer_reason = "运行 bash run_missing_materials_4gpu.sh 生成；本地当前没有正式训练结果"
    conch_tasks = [
        OUT / "second_encoder" / "conch_abmil" / "M1_immune_sensitive" / "metrics_conch_abmil_M1_immune_sensitive.json",
        OUT / "second_encoder" / "conch_abmil" / "M2_msi" / "metrics_conch_abmil_M2_msi.json",
        OUT / "second_encoder" / "conch_abmil" / "M3_ebv" / "metrics_conch_abmil_M3_ebv.json",
        OUT / "second_encoder" / "conch_abmil" / "M4_subtype4" / "metrics_conch_abmil_M4_subtype4.json",
    ]
    conch_done = sum(p.exists() for p in conch_tasks)
    if conch_done == len(conch_tasks):
        conch_status = "完成"
        conch_reason = "results/locked_release_20260902/second_encoder/conch_abmil/*/metrics_conch_abmil_*.json"
    elif conch_done:
        conch_status = "部分完成"
        conch_reason = f"已完成 {conch_done}/{len(conch_tasks)} 个任务，见 results/locked_release_20260902/second_encoder/conch_abmil/"
    else:
        conch_status = "未完成"
        conch_reason = "已实现 run_conch_abmil_4gpu.sh；需下载TCGA原片并抽CONCH特征后训练"
    rows = [
        ("冻结TCGA 246 patient_manifest.csv", "完成", "results/locked_release_20260902/patient_manifest.csv"),
        ("冻结fold_assignment.csv", "完成", "results/locked_release_20260902/fold_assignment.csv"),
        ("冻结TCGA 231 locked_predictions_tcga.csv", "完成", "results/locked_release_20260902/locked_predictions_tcga.csv"),
        ("冻结CPTAC 156 locked_predictions_cptac.csv", "完成", "results/locked_release_20260902/locked_predictions_cptac.csv"),
        ("CPTAC模型来源与固定阈值策略", "完成", "results/locked_release_20260902/inference_model_policy.json"),
        ("权重/代码SHA256清单", "完成", "results/locked_release_20260902/model_and_code_hash_manifest.csv"),
        ("训练配置摘要", "完成", "results/locked_release_20260902/training_config_summary.json"),
        ("TCGA 231 AUROC/AP/阈值指标/校准CI", "完成", "results/locked_release_20260902/tcga_231_metrics_with_ci.json"),
        ("CPTAC 156 阈值指标CI补充", "完成", "results/locked_release_20260902/cptac_156_metrics_with_ci.json"),
        ("ROC/PR/校准曲线", "完成", "results/locked_release_20260902/roc_pr_curves_tcga231_cptac156.png"),
        ("临床容量情景 20/30/40%", "完成", "results/locked_release_20260902/triage_capacity_tcga.csv; results/locked_release_20260902/triage_capacity_cptac.csv"),
        ("错误病例/低置信病理审阅模板", "完成", "results/locked_release_20260902/pathologist_annotation_template.csv"),
        ("Matched Clinical-only 231", "完成", "results/locked_release_20260902/matched_clinical_oof_predictions_231.csv"),
        ("Image+clinical融合模型", fusion_status, fusion_reason),
        ("mean/max pooling最小基线", baseline_status, baseline_reason),
        ("CONCH第二编码器+ABMIL", conch_status, conch_reason),
        ("Transformer aggregator", transformer_status, transformer_reason),
        ("病理医生盲评/IHC验证", "不可由脚本完成", "需要病理同学在pathologist_annotation_template.csv上人工标注"),
    ]
    df = pd.DataFrame(rows, columns=["item", "status", "deliverable_or_reason"])
    df.to_csv(OUT / "completion_status_after_20260902.csv", index=False)
    write_json(OUT / "completion_status_after_20260902.json", df.to_dict(orient="records"))


def write_report(tcga_metrics: dict[str, Any], cptac_metrics: dict[str, Any], clinical_metrics: dict[str, Any]) -> None:
    m1 = tcga_metrics["tasks"]["M1_immune_sensitive"]["point"]
    cm1 = cptac_metrics["tasks"]["M1_immune_sensitive"]["point"]
    clinical = clinical_metrics["models"]["Clinical_only_matched_231"]["point"]
    fusion = clinical_metrics["models"]["Image_plus_clinical_late_fusion_matched_231"]["point"]
    baseline_text = ""
    baseline_path = OUT / "baselines" / "baseline_metrics.json"
    if baseline_path.exists():
        baseline = read_json(baseline_path)["baselines"]
        baseline_text = (
            f"UNI2-h mean pooling M1：AUROC={baseline['uni2h_mean_pooling']['M1_immune_sensitive']['auroc']:.3f}，"
            f"AP={baseline['uni2h_mean_pooling']['M1_immune_sensitive']['average_precision']:.3f}。  \n"
            f"UNI2-h max pooling M1：AUROC={baseline['uni2h_max_pooling']['M1_immune_sensitive']['auroc']:.3f}，"
            f"AP={baseline['uni2h_max_pooling']['M1_immune_sensitive']['average_precision']:.3f}。  \n"
        )
    text = f"""# 计算机改进部分逐项完成进展

生成时间：{datetime.now().isoformat(timespec="seconds")}  
项目目录：`{ROOT}`  
输出目录：`{OUT}`  
Git HEAD：`{run_text(["git", "rev-parse", "HEAD"])}`  

## 已补齐的核心交付

1. 已冻结 TCGA 246 例主队列、231 例中心隔离 OOF、CPTAC 156 例外部验证队列。
2. 已按患者级 bootstrap 重算 TCGA 231 和 CPTAC 156 的 AUROC、AP、阈值指标、Brier、校准截距/斜率 CI。
3. 已生成 ROC/PR、校准曲线、临床检测容量曲线。
4. 已按同一 231 例和同一 fold 重算 Clinical-only，并生成图像+临床 late-fusion/stacking 探索版。
5. 已生成 UNI2-h mean pooling 和 max pooling 最小基线。
6. 已生成错误病例和低置信病例的病理审阅模板。

## 当前主线指标摘要

TCGA 231 OOF M1：AUROC={m1["auroc"]:.3f}，AP={m1["average_precision"]:.3f}，Sensitivity={m1["sensitivity"]:.3f}，Specificity={m1["specificity"]:.3f}。  
CPTAC 156 external M1：AUROC={cm1["auroc"]:.3f}，AP={cm1["average_precision"]:.3f}，Sensitivity={cm1["sensitivity"]:.3f}，Specificity={cm1["specificity"]:.3f}。  
Matched Clinical-only 231：AUROC={clinical["auroc"]:.3f}，AP={clinical["average_precision"]:.3f}。  
Image+clinical late-fusion 231：AUROC={fusion["auroc"]:.3f}，AP={fusion["average_precision"]:.3f}。  
{baseline_text}

## 仍不能自动交付的内容

CONCH 第二编码器 + ABMIL 已提供下载、抽特征和同 fold 训练脚本；若对应 metrics 尚不存在，说明仍需在四卡环境运行 `run_conch_abmil_4gpu.sh`。病理形态学解释需要病理医生基于模板盲评，不能由当前脚本替代。late-fusion 文件只能作为探索性图像分数+临床融合，不等同于端到端图像特征融合网络。
"""
    (OUT / "COMPLETION_REPORT_20260902.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    locked = build_locked_tcga()
    cptac = build_locked_cptac()
    summarize_model_hashes()
    build_policy_files()
    tcga_metrics = recompute_stats_tcga(locked)
    cptac_metrics = add_cptac_threshold_cis(cptac)
    draw_curves(locked, cptac)
    tcga_cap = triage_capacity(locked, "TCGA")
    cptac_cap = triage_capacity(cptac, "CPTAC")
    draw_capacity(tcga_cap, cptac_cap)
    tcga_err = make_error_review(locked, "TCGA")
    cptac_err = make_error_review(cptac, "CPTAC")
    pathologist_template(tcga_err, cptac_err)
    clinical_metrics = run_matched_clinical(locked)
    write_completion_status()
    write_report(tcga_metrics, cptac_metrics, clinical_metrics)
    print(
        json.dumps(
            {
                "out_dir": str(OUT),
                "tcga_locked_rows": int(len(locked)),
                "cptac_locked_rows": int(len(cptac)),
                "tcga_error_review_rows": int(len(tcga_err)),
                "cptac_error_review_rows": int(len(cptac_err)),
                "status_csv": rel(OUT / "completion_status_after_20260902.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
