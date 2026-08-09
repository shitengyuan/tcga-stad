#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
except Exception:
    accuracy_score = None


TASKS = [
    ("immune_sensitive", "immune_sensitive_prob", "immune_sensitive_pred", "Immune-sensitive"),
    ("msi", "msi_prob", "msi_pred", "MSI"),
    ("ebv", "ebv_prob", "ebv_pred", "EBV"),
]
SUBTYPE_NAMES = ["EBV", "MSI", "GS", "CIN"]
SUBTYPE_TO_ID = {name: i for i, name in enumerate(SUBTYPE_NAMES)}


def qstats(x: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return {
        "n": int(len(x)),
        "mean": float(x.mean()) if len(x) else None,
        "std": float(x.std()) if len(x) > 1 else None,
        "min": float(x.min()) if len(x) else None,
        "q25": float(x.quantile(0.25)) if len(x) else None,
        "median": float(x.median()) if len(x) else None,
        "q75": float(x.quantile(0.75)) if len(x) else None,
        "max": float(x.max()) if len(x) else None,
    }


def to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.upper()
    pos = {"1", "TRUE", "YES", "Y", "POS", "POSITIVE", "MSI", "MSI-H", "MSI_H", "EBV", "EBV+", "IMMUNE_SENSITIVE"}
    neg = {"0", "FALSE", "NO", "N", "NEG", "NEGATIVE", "MSS", "EBV-", "NON_SENSITIVE", "NONSENSITIVE"}
    return text.map(lambda v: 1 if v in pos else (0 if v in neg else np.nan))


def binary_metrics(df: pd.DataFrame, label_col: str, score_col: str, pred_col: str) -> dict[str, Any]:
    y = to_binary(df[label_col])
    s = pd.to_numeric(df[score_col], errors="coerce")
    p = pd.to_numeric(df[pred_col], errors="coerce")
    mask = y.notna() & s.notna() & p.notna()
    yv = y[mask].astype(int).to_numpy()
    sv = s[mask].to_numpy()
    pv = p[mask].astype(int).to_numpy()
    out: dict[str, Any] = {"n": int(len(yv)), "n_pos": int(yv.sum()) if len(yv) else 0}
    if len(yv) == 0:
        out["error"] = "no labeled samples"
        return out
    cm = confusion_matrix(yv, pv, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out.update(
        {
            "accuracy": float(accuracy_score(yv, pv)),
            "balanced_accuracy": float(balanced_accuracy_score(yv, pv)) if len(np.unique(yv)) == 2 else None,
            "f1": float(f1_score(yv, pv, zero_division=0)),
            "precision": float(precision_score(yv, pv, zero_division=0)),
            "recall_sensitivity": float(recall_score(yv, pv, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
            "confusion_matrix": cm.tolist(),
        }
    )
    if len(np.unique(yv)) == 2:
        out["auc"] = float(roc_auc_score(yv, sv))
        out["ap"] = float(average_precision_score(yv, sv))
    else:
        out["auc"] = None
        out["ap"] = None
        out["warning"] = "need both classes for AUC/AP"
    return out


def subtype_label_to_id(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.upper()
    mapping = {
        "EBV": 0,
        "STAD_EBV": 0,
        "MSI": 1,
        "MSI-H": 1,
        "MSI_H": 1,
        "STAD_MSI": 1,
        "GS": 2,
        "STAD_GS": 2,
        "CIN": 3,
        "STAD_CIN": 3,
    }
    return text.map(lambda v: mapping.get(v, np.nan))


def load_inputs(pred_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    patient_csv = pred_dir / "external_feature_patient_predictions.csv"
    slide_csv = pred_dir / "external_feature_slide_predictions.csv"
    error_json = pred_dir / "external_feature_errors.json"
    if not patient_csv.exists():
        raise FileNotFoundError(patient_csv)
    if not slide_csv.exists():
        raise FileNotFoundError(slide_csv)
    patient_df = pd.read_csv(patient_csv)
    slide_df = pd.read_csv(slide_csv)
    errors = []
    if error_json.exists():
        errors = json.loads(error_json.read_text())
    return patient_df, slide_df, errors


def write_summary(patient_df: pd.DataFrame, slide_df: pd.DataFrame, errors: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_patients": int(patient_df["patient_id"].nunique()),
        "n_slides": int(slide_df["slide_id"].nunique()),
        "n_errors": int(len(errors)),
        "patient_level": {},
        "slide_level": {},
        "coverage": {
            "n_slides_per_patient": qstats(patient_df["n_slides"]) if "n_slides" in patient_df.columns else {},
            "n_patches_total_per_patient": qstats(patient_df["n_patches_total"]) if "n_patches_total" in patient_df.columns else {},
            "n_patches_per_slide": qstats(slide_df["n_patches"]) if "n_patches" in slide_df.columns else {},
        },
    }
    for level_name, df in [("patient_level", patient_df), ("slide_level", slide_df)]:
        for key, prob_col, pred_col, label in TASKS:
            summary[level_name][key] = {
                "label": label,
                "prob": qstats(df[prob_col]),
                "pred_counts": df[pred_col].value_counts(dropna=False).sort_index().astype(int).to_dict(),
            }
        summary[level_name]["subtype4"] = {
            "pred_counts": df["subtype4_pred"].value_counts(dropna=False).to_dict(),
        }
    (out_dir / "cptac_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for key, prob_col, pred_col, label in TASKS:
        s = summary["patient_level"][key]
        rows.append(
            {
                "task": key,
                "label": label,
                "n_patients": summary["n_patients"],
                "n_pred_positive": int(patient_df[pred_col].sum()),
                "positive_rate": float(patient_df[pred_col].mean()),
                "mean_prob": s["prob"]["mean"],
                "median_prob": s["prob"]["median"],
                "q25_prob": s["prob"]["q25"],
                "q75_prob": s["prob"]["q75"],
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "cptac_patient_task_summary.csv", index=False)
    return summary


def add_label_metrics(patient_df: pd.DataFrame, labels_csv: Path, out_dir: Path) -> dict[str, Any]:
    if accuracy_score is None:
        return {"error": "sklearn is not available"}
    labels = pd.read_csv(labels_csv)
    if "patient_id" not in labels.columns and "case_submitter_id" in labels.columns:
        labels = labels.rename(columns={"case_submitter_id": "patient_id"})
    if "patient_id" not in labels.columns:
        return {"error": "labels_csv has no patient_id column"}
    df = patient_df.merge(labels, on="patient_id", how="inner", suffixes=("", "_label"))
    metrics: dict[str, Any] = {"n_pred_patients": int(patient_df["patient_id"].nunique()), "n_labeled_rows": int(len(df)), "tasks": {}}
    for key, score_col, pred_col, label in TASKS:
        if key in df.columns:
            metrics["tasks"][key] = binary_metrics(df, key, score_col, pred_col)
    if "subtype4" in df.columns and "subtype4_pred_class" in df.columns:
        y = subtype_label_to_id(df["subtype4"])
        p = pd.to_numeric(df["subtype4_pred_class"], errors="coerce")
        mask = y.notna() & p.notna()
        yv = y[mask].astype(int).to_numpy()
        pv = p[mask].astype(int).to_numpy()
        if len(yv):
            metrics["tasks"]["subtype4"] = {
                "n": int(len(yv)),
                "accuracy": float(accuracy_score(yv, pv)),
                "macro_f1": float(f1_score(yv, pv, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(yv, pv, labels=[0, 1, 2, 3]).tolist(),
                "classes": SUBTYPE_NAMES,
            }
        else:
            metrics["tasks"]["subtype4"] = {"n": 0, "error": "no labeled samples"}
    (out_dir / "cptac_supervised_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df.to_csv(out_dir / "cptac_patient_predictions_with_labels.csv", index=False)
    return metrics


def save_probability_histograms(patient_df: pd.DataFrame, slide_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey="row")
    for col_idx, (key, prob_col, pred_col, label) in enumerate(TASKS):
        for row_idx, (df, level) in enumerate([(patient_df, "Patient"), (slide_df, "Slide")]):
            ax = axes[row_idx, col_idx]
            ax.hist(df[prob_col].dropna(), bins=np.linspace(0, 1, 21), color="#4C78A8", alpha=0.85, edgecolor="white")
            ax.axvline(0.5, color="#E45756", linestyle="--", linewidth=1.5)
            ax.set_title(f"{level}: {label}")
            ax.set_xlabel("Predicted probability")
            ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_probability_histograms.png", dpi=180)
    plt.close(fig)


def save_patient_bars(patient_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (key, prob_col, pred_col, label) in zip(axes[:3], TASKS):
        counts = patient_df[pred_col].value_counts().reindex([0, 1], fill_value=0)
        ax.bar(["Negative", "Positive"], counts.values, color=["#72B7B2", "#E45756"])
        ax.set_title(label)
        ax.set_ylabel("Patients")
        for i, v in enumerate(counts.values):
            ax.text(i, v + 0.3, str(int(v)), ha="center", va="bottom")
    subtype_counts = patient_df["subtype4_pred"].value_counts().reindex(SUBTYPE_NAMES, fill_value=0)
    axes[3].bar(subtype_counts.index, subtype_counts.values, color=["#B279A2", "#F58518", "#54A24B", "#4C78A8"])
    axes[3].set_title("Subtype4")
    axes[3].set_ylabel("Patients")
    for i, v in enumerate(subtype_counts.values):
        axes[3].text(i, v + 0.3, str(int(v)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_patient_prediction_counts.png", dpi=180)
    plt.close(fig)


def save_heatmap(patient_df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["immune_sensitive_prob", "msi_prob", "ebv_prob", *[f"M4_subtype4_prob_c{i}" for i in range(4)]]
    labels = ["Immune", "MSI", "EBV", "M4_EBV", "M4_MSI", "M4_GS", "M4_CIN"]
    df = patient_df.sort_values(["immune_sensitive_prob", "msi_prob"], ascending=False).reset_index(drop=True)
    mat = df[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.18 * len(df))))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["patient_id"], fontsize=7)
    ax.set_title("Patient-level predicted probabilities")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_patient_probability_heatmap.png", dpi=180)
    plt.close(fig)


def save_slide_patient_consistency(patient_df: pd.DataFrame, slide_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (key, prob_col, pred_col, label) in zip(axes, TASKS):
        rows = []
        for pid, sub in slide_df.groupby("patient_id"):
            if len(sub) <= 1:
                continue
            rows.append({"patient_id": pid, "slide_std": float(sub[prob_col].std()), "patient_prob": float(patient_df.set_index("patient_id").loc[pid, prob_col])})
        d = pd.DataFrame(rows)
        if not d.empty:
            ax.scatter(d["patient_prob"], d["slide_std"], s=28, alpha=0.8, color="#4C78A8")
        ax.set_title(label)
        ax.set_xlabel("Patient mean probability")
        ax.set_ylabel("Within-patient slide std")
        ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_slide_patient_consistency.png", dpi=180)
    plt.close(fig)


def save_coverage(patient_df: pd.DataFrame, slide_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(patient_df["n_slides"], bins=range(1, int(patient_df["n_slides"].max()) + 3), color="#4C78A8", edgecolor="white")
    axes[0].set_title("Slides per patient")
    axes[0].set_xlabel("Slides")
    axes[0].set_ylabel("Patients")
    axes[1].hist(patient_df["n_patches_total"], bins=15, color="#F58518", edgecolor="white")
    axes[1].set_title("Patches per patient")
    axes[1].set_xlabel("Patches")
    axes[2].hist(slide_df["n_patches"], bins=20, color="#54A24B", edgecolor="white")
    axes[2].set_title("Patches per slide")
    axes[2].set_xlabel("Patches")
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_coverage_histograms.png", dpi=180)
    plt.close(fig)


def save_supervised_plots(patient_df: pd.DataFrame, labels_csv: Path, out_dir: Path) -> None:
    if accuracy_score is None:
        return
    labels = pd.read_csv(labels_csv)
    if "patient_id" not in labels.columns and "case_submitter_id" in labels.columns:
        labels = labels.rename(columns={"case_submitter_id": "patient_id"})
    if "patient_id" not in labels.columns:
        return
    df = patient_df.merge(labels, on="patient_id", how="inner")
    valid_tasks = []
    for key, score_col, pred_col, label in TASKS:
        if key not in df.columns:
            continue
        y = to_binary(df[key])
        s = pd.to_numeric(df[score_col], errors="coerce")
        mask = y.notna() & s.notna()
        if mask.sum() > 0 and len(np.unique(y[mask].astype(int))) == 2:
            valid_tasks.append((key, score_col, pred_col, label, y[mask].astype(int), s[mask]))
    if not valid_tasks:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for key, score_col, pred_col, label, y, s in valid_tasks:
        fpr, tpr, _ = roc_curve(y, s)
        precision, recall, _ = precision_recall_curve(y, s)
        axes[0].plot(fpr, tpr, label=f"{label} AUC={roc_auc_score(y, s):.3f}")
        axes[1].plot(recall, precision, label=f"{label} AP={average_precision_score(y, s):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_title("ROC")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[1].set_title("Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_supervised_roc_pr.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=Path, default=Path("results/external_cptac_feature_infer_20x256_4gpu"))
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--labels_csv", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (args.pred_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    patient_df, slide_df, errors = load_inputs(args.pred_dir)

    summary = write_summary(patient_df, slide_df, errors, out_dir)
    metrics = None
    if args.labels_csv:
        metrics = add_label_metrics(patient_df, args.labels_csv, out_dir)

    save_probability_histograms(patient_df, slide_df, out_dir)
    save_patient_bars(patient_df, out_dir)
    save_heatmap(patient_df, out_dir)
    save_slide_patient_consistency(patient_df, slide_df, out_dir)
    save_coverage(patient_df, slide_df, out_dir)
    if args.labels_csv:
        save_supervised_plots(patient_df, args.labels_csv, out_dir)

    print(f"Wrote summary: {out_dir / 'cptac_prediction_summary.json'}")
    print(f"Wrote task summary: {out_dir / 'cptac_patient_task_summary.csv'}")
    print(f"Wrote figures to: {out_dir}")
    if metrics is not None:
        print(f"Wrote supervised metrics: {out_dir / 'cptac_supervised_metrics.json'}")
        print(json.dumps(metrics, indent=2))
    print(json.dumps(summary["patient_level"], indent=2))


if __name__ == "__main__":
    main()
