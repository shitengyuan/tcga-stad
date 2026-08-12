#!/usr/bin/env python3
"""
Generate CPU-only supplementary audit materials.

Inputs are existing OOF/external prediction CSVs. No WSI, h5 feature loading,
GPU inference, or LLM calls are performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
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
    precision_score,
    recall_score,
    roc_auc_score,
)


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
AUDIT = RESULTS / "audit_first_stage"
OUT = RESULTS / "cpu_supplement"
M4_CLASSES = ["EBV", "MSI", "GS", "CIN"]


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


def run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=str(BASE), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, score))


def fixed_binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y = y.astype(int)
    score = score.astype(float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "threshold": float(threshold),
        "roc_auc": safe_auc(y, score),
        "pr_auc_average_precision": safe_ap(y, score),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else None,
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "ppv": float(precision_score(y, pred, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else None,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, score)),
        "confusion_matrix_labels_0_1": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_delta_auc(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    y = y.astype(int)
    deltas = []
    auc_a = []
    auc_b = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aa = roc_auc_score(y[idx], score_a[idx])
        bb = roc_auc_score(y[idx], score_b[idx])
        auc_a.append(aa)
        auc_b.append(bb)
        deltas.append(aa - bb)
    deltas = np.asarray(deltas)
    return {
        "n_boot": int(n_boot),
        "auc_a_mean": float(np.mean(auc_a)),
        "auc_b_mean": float(np.mean(auc_b)),
        "delta_mean": float(np.mean(deltas)),
        "delta_95ci": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "two_sided_bootstrap_p": float(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))),
        "interpretation": "paired patient-level bootstrap for AUC difference; positive delta favors model_a",
    }


def decision_curve(y: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    y = y.astype(int)
    n = len(y)
    prevalence = y.mean()
    for t in thresholds:
        pred = score >= t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        nb = tp / n - fp / n * (t / (1 - t))
        treat_all = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append(
            {
                "threshold": float(t),
                "net_benefit_model": float(nb),
                "net_benefit_treat_all": float(treat_all),
                "net_benefit_treat_none": 0.0,
                "tp": tp,
                "fp": fp,
                "n": int(n),
            }
        )
    return pd.DataFrame(rows)


def make_error_lists() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = [
        ("M1_immune_sensitive", RESULTS / "oof_preds_M1_immune_sensitive.csv", "prob_c1"),
        ("M2_msi", RESULTS / "oof_preds_M2_msi.csv", "prob_c1"),
        ("M3_ebv", RESULTS / "oof_preds_M3_ebv.csv", "prob_c1"),
        ("M5_clinical", RESULTS / "oof_preds_M5_clinical.csv", "prob_c1"),
    ]
    summary: dict[str, Any] = {}
    for name, path, prob_col in specs:
        df = pd.read_csv(path)
        df["score"] = df[prob_col].astype(float)
        df["pred"] = (df["score"] >= 0.5).astype(int)
        df["error_type"] = np.where(
            (df["label"].astype(int) == 1) & (df["pred"] == 0),
            "FN",
            np.where((df["label"].astype(int) == 0) & (df["pred"] == 1), "FP", np.where(df["label"].astype(int) == 1, "TP", "TN")),
        )
        df["margin_to_0p5"] = (df["score"] - 0.5).abs()
        out = df.sort_values(["error_type", "margin_to_0p5", "patient_id"])
        out.to_csv(OUT / f"errors_{name}_oof.csv", index=False)
        summary[name] = {
            "source": str(path),
            "counts": out["error_type"].value_counts().to_dict(),
            "n_errors": int(out["error_type"].isin(["FP", "FN"]).sum()),
        }

    m4 = pd.read_csv(RESULTS / "oof_preds_M4_subtype4.csv")
    probs = m4[[f"prob_c{i}" for i in range(4)]].astype(float).to_numpy()
    m4["pred_class"] = probs.argmax(axis=1)
    m4["pred_subtype"] = [M4_CLASSES[i] for i in m4["pred_class"]]
    m4["true_subtype"] = [M4_CLASSES[int(i)] for i in m4["label"].astype(int)]
    m4["correct"] = m4["pred_class"].eq(m4["label"].astype(int))
    m4["error_type"] = np.where(m4["correct"], "correct", m4["true_subtype"] + "_to_" + m4["pred_subtype"])
    m4.to_csv(OUT / "errors_M4_subtype4_oof.csv", index=False)
    summary["M4_subtype4"] = {
        "source": str(RESULTS / "oof_preds_M4_subtype4.csv"),
        "class_order": M4_CLASSES,
        "counts": m4["error_type"].value_counts().to_dict(),
        "n_errors": int((~m4["correct"]).sum()),
    }
    write_json(OUT / "error_case_summary.json", summary)
    return summary


def subgroup_metrics() -> dict[str, Any]:
    cohort = pd.read_csv(AUDIT / "final_289_patient_cohort.csv")
    out_rows = []
    binary_specs = [
        ("M1_immune_sensitive", "M1_label", "M1_immune_sensitive_prob_c1"),
        ("M2_msi", "MSI", "M2_msi_prob_c1"),
        ("M3_ebv", "EBV", "M3_ebv_prob_c1"),
        ("M5_clinical", "M1_label", "M5_clinical_prob_c1"),
    ]
    for group_col in ["site", "M4_subtype"]:
        for group, g in cohort.groupby(group_col, dropna=False):
            for model, label_col, score_col in binary_specs:
                m = fixed_binary_metrics(g[label_col].to_numpy(), g[score_col].to_numpy())
                row = {"group_by": group_col, "group": group, "model": model, **m}
                out_rows.append(row)
    subgroup = pd.DataFrame(out_rows)
    subgroup.to_csv(OUT / "subgroup_metrics_by_site_and_subtype.csv", index=False)

    summary = {
        "source": str(AUDIT / "final_289_patient_cohort.csv"),
        "group_by": ["site", "M4_subtype"],
        "warning": "Only site and molecular subtype are available without clinical.csv; Lauren/site/stage/quality subgroup analyses remain missing.",
        "n_rows": int(len(subgroup)),
    }
    write_json(OUT / "subgroup_metrics_summary.json", summary)
    return summary


def calibration_and_dca() -> dict[str, Any]:
    cohort = pd.read_csv(AUDIT / "final_289_patient_cohort.csv")
    specs = [
        ("M1_immune_sensitive", "M1_label", "M1_immune_sensitive_prob_c1"),
        ("M2_msi", "MSI", "M2_msi_prob_c1"),
        ("M3_ebv", "EBV", "M3_ebv_prob_c1"),
        ("M5_clinical", "M1_label", "M5_clinical_prob_c1"),
    ]
    cal_rows = []
    dca_rows = []
    for model, label_col, score_col in specs:
        y = cohort[label_col].astype(int).to_numpy()
        score = cohort[score_col].astype(float).to_numpy()
        frac_pos, mean_pred = calibration_curve(y, score, n_bins=10, strategy="quantile")
        for i, (mp, fp) in enumerate(zip(mean_pred, frac_pos)):
            cal_rows.append({"model": model, "bin": i, "mean_predicted": float(mp), "fraction_positive": float(fp), "n_total": int(len(y))})
        dca = decision_curve(y, score, np.round(np.arange(0.01, 1.00, 0.01), 2))
        dca.insert(0, "model", model)
        dca_rows.append(dca)
    pd.DataFrame(cal_rows).to_csv(OUT / "calibration_curve_data_oof.csv", index=False)
    pd.concat(dca_rows, ignore_index=True).to_csv(OUT / "decision_curve_data_oof.csv", index=False)
    summary = {"calibration_csv": "calibration_curve_data_oof.csv", "decision_curve_csv": "decision_curve_data_oof.csv", "thresholds": "0.01..0.99"}
    write_json(OUT / "calibration_dca_summary.json", summary)
    return summary


def model_comparison() -> dict[str, Any]:
    cohort = pd.read_csv(AUDIT / "final_289_patient_cohort.csv")
    y = cohort["M1_label"].astype(int).to_numpy()
    m1 = cohort["M1_immune_sensitive_prob_c1"].astype(float).to_numpy()
    m5 = cohort["M5_clinical_prob_c1"].astype(float).to_numpy()
    result = {
        "model_a": "M1_immune_sensitive",
        "model_b": "M5_clinical",
        "endpoint": "immune_sensitive",
        "n": int(len(y)),
        "model_a_auc": float(roc_auc_score(y, m1)),
        "model_b_auc": float(roc_auc_score(y, m5)),
        "paired_bootstrap_auc_difference": bootstrap_delta_auc(y, m1, m5, n_boot=5000, seed=4242),
        "note": "This is CPU-only paired bootstrap. DeLong testing is not included to avoid adding another dependency/implementation risk.",
    }
    write_json(OUT / "paired_model_comparison_M1_vs_M5.json", result)
    return result


def agent_no_leakage_package() -> dict[str, Any]:
    cohort = pd.read_csv(AUDIT / "final_289_patient_cohort.csv")
    allowed = cohort[
        [
            "patient_id",
            "site",
            "M1_immune_sensitive_prob_c0",
            "M1_immune_sensitive_prob_c1",
            "M2_msi_prob_c0",
            "M2_msi_prob_c1",
            "M3_ebv_prob_c0",
            "M3_ebv_prob_c1",
            "M4_subtype4_prob_c0",
            "M4_subtype4_prob_c1",
            "M4_subtype4_prob_c2",
            "M4_subtype4_prob_c3",
            "M5_clinical_prob_c0",
            "M5_clinical_prob_c1",
        ]
    ].copy()
    allowed["m4_class_order"] = "prob_c0=EBV;prob_c1=MSI;prob_c2=GS;prob_c3=CIN"
    allowed["visual_module_status"] = "disabled_no_visual_evidence_available"
    allowed["agent_allowed_action"] = "explain model agreement/conflict only; do not change M1 final prediction"
    allowed["final_pred_fixed_from_M1"] = np.where(allowed["M1_immune_sensitive_prob_c1"] >= 0.5, "sensitive", "nonsensitive")
    allowed.to_csv(OUT / "agent_no_leakage_inputs_289.csv", index=False)

    schema = {
        "type": "object",
        "required": ["patient_id", "final_pred", "confidence", "trusted_models", "conflicts", "ihc_suggest", "report"],
        "properties": {
            "patient_id": {"type": "string"},
            "final_pred": {"enum": ["sensitive", "nonsensitive"]},
            "confidence": {"enum": ["high", "medium", "low"]},
            "trusted_models": {"type": "array", "items": {"type": "string"}},
            "conflicts": {"type": "array", "items": {"type": "string"}},
            "ihc_suggest": {"enum": ["MMR", "EBER-ISH", "both", "none"]},
            "report": {"type": "string", "description": "No morphology claims unless real visual evidence is supplied."},
        },
    }
    write_json(OUT / "agent_no_leakage_output_schema.json", schema)

    prompt = """你是胃肠病理AI助手。你只能读取模型概率和非分子临床字段。
禁止使用真实MSI、EBV、TCGA subtype、label、POLE结果作为输入。
当前没有视觉模块和图像证据，因此禁止输出TIL、CIN、MSI形态学结论。
最终预测必须等于M1固定阈值结果，Agent不得修改M1预测。
任务：总结M1/M2/M3/M4/M5之间的一致性或冲突，给出high/medium/low置信度，并建议MMR IHC、EBER-ISH、both或none。
只输出符合JSON Schema的JSON。"""
    (OUT / "agent_no_leakage_prompt.txt").write_text(prompt, encoding="utf-8")

    summary = {
        "inputs_csv": "agent_no_leakage_inputs_289.csv",
        "schema_json": "agent_no_leakage_output_schema.json",
        "prompt_txt": "agent_no_leakage_prompt.txt",
        "excluded_fields": ["label", "MSI", "EBV", "subtype", "M4_subtype", "M4_label", "POLE"],
        "visual_module": "disabled; no morphology statements allowed",
    }
    write_json(OUT / "agent_no_leakage_package_summary.json", summary)
    return summary


def model_registry() -> dict[str, Any]:
    registry = {
        "git_commit": run_text(["git", "rev-parse", "HEAD"]),
        "models": [],
        "important_note": "M1-M4 .pt files are saved by current training code as the last fold model for each task; per-fold weights and full-data refit weights are not available.",
    }
    model_info = {
        "M1_immune_sensitive.pt": {"task": "immune_sensitive", "n_classes": 2, "status": "available_last_fold_checkpoint"},
        "M2_msi.pt": {"task": "msi", "n_classes": 2, "status": "available_last_fold_checkpoint"},
        "M3_ebv.pt": {"task": "ebv", "n_classes": 2, "status": "available_last_fold_checkpoint"},
        "M4_subtype4.pt": {"task": "subtype4", "n_classes": 4, "status": "available_last_fold_checkpoint", "class_order": M4_CLASSES},
    }
    for name, meta in model_info.items():
        p = BASE / "models" / name
        registry["models"].append({"file": str(p), "sha256": sha256(p), **meta})
    registry["missing_model_artifacts"] = [
        "M1-M4 per-fold weights",
        "M1-M4 best epoch files",
        "M5 fitted sklearn pipeline",
        "M6 survival model weights / OOF risk table",
        "training logs and hyperparameter search records",
    ]
    write_json(OUT / "model_registry_current.json", registry)
    return registry


def cptac_error_lists() -> dict[str, Any]:
    path = AUDIT / "cptac_patient_cohort_labels_predictions_fixed_threshold.csv"
    if not path.exists():
        return {"error": f"missing {path}"}
    df = pd.read_csv(path)
    specs = [
        ("M1_immune_sensitive", "immune_sensitive", "immune_sensitive_prob"),
        ("M2_msi", "msi", "msi_prob"),
        ("M3_ebv", "ebv", "ebv_prob"),
    ]
    summary = {}
    for model, label_col, score_col in specs:
        sub = df.dropna(subset=[label_col, score_col]).copy()
        sub["pred"] = (sub[score_col].astype(float) >= 0.5).astype(int)
        sub["error_type"] = np.where(
            (sub[label_col].astype(int) == 1) & (sub["pred"] == 0),
            "FN",
            np.where((sub[label_col].astype(int) == 0) & (sub["pred"] == 1), "FP", np.where(sub[label_col].astype(int) == 1, "TP", "TN")),
        )
        sub.to_csv(OUT / f"cptac_errors_{model}.csv", index=False)
        summary[model] = {"n": int(len(sub)), "counts": sub["error_type"].value_counts().to_dict()}
    slide_excl = pd.read_csv(AUDIT / "cptac_slide_cohort_and_exclusions.csv")
    summary["slide_exclusions"] = slide_excl["exclusion_reason"].fillna("included").value_counts().to_dict()
    write_json(OUT / "cptac_error_summary.json", summary)
    return summary


def write_readme(outputs: dict[str, Any]) -> None:
    text = f"""# CPU Supplement Materials

Generated from existing CSV/JSON outputs only. No GPU, WSI loading, feature extraction, or LLM calls were used.

## Newly Added

- `errors_*_oof.csv`: TCGA OOF TP/TN/FP/FN and M4 misclassification lists.
- `error_case_summary.json`: error counts by model.
- `subgroup_metrics_by_site_and_subtype.csv`: subgroup metrics by TCGA site and molecular subtype.
- `calibration_curve_data_oof.csv`: calibration curve raw data.
- `decision_curve_data_oof.csv`: decision curve raw data.
- `paired_model_comparison_M1_vs_M5.json`: paired bootstrap comparison of M1 vs M5 AUC.
- `agent_no_leakage_inputs_289.csv`: Agent input table with true labels/subtypes removed.
- `agent_no_leakage_output_schema.json` and `agent_no_leakage_prompt.txt`: no-leakage Agent contract.
- `model_registry_current.json`: current model artifact registry.
- `cptac_errors_*.csv` and `cptac_error_summary.json`: CPTAC fixed-threshold error lists.

## Still Missing

- `clinical.csv`, TCGA h5 features, original fold files, per-fold weights, training logs.
- WSI thumbnails, attention heatmaps, cluster overlays, representative patches.
- Formal no-leakage LLM rerun outputs. This package only prepares safe inputs/prompt/schema.

## Re-run

```bash
cd /share/home/shitengyuan_lustre/medical/tcga-stad
/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python generate_cpu_supplement.py
```
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=OUT)
    args = p.parse_args()
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    outputs = {
        "errors": make_error_lists(),
        "subgroups": subgroup_metrics(),
        "calibration_dca": calibration_and_dca(),
        "m1_vs_m5": model_comparison(),
        "agent_no_leakage": agent_no_leakage_package(),
        "model_registry": model_registry(),
        "cptac_errors": cptac_error_lists(),
    }
    write_json(OUT / "cpu_supplement_manifest.json", outputs)
    write_readme(outputs)
    print(f"Wrote CPU supplement materials: {OUT}")


if __name__ == "__main__":
    main()
