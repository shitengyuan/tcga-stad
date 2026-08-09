#!/usr/bin/env python3
"""
Evaluate saved CPTAC/STAD slide features with the trained ABMIL panel.

Inputs:
  feature_dir: one feature file per slide (.h5/.hdf5/.pt/.pth/.npy/.npz)
  model_dir:   trained ABMIL checkpoints M1-M4
  labels_csv:  optional patient-level labels for metrics

Outputs:
  external_feature_slide_predictions.csv
  external_feature_patient_predictions.csv
  external_feature_metrics.json             (only when labels are provided)

Example:
  python eval_cptac_features.py \
    --feature_dir /path/to/cptac_features \
    --model_dir models \
    --out_dir results/external_cptac_feature_eval \
    --labels_csv /path/to/cptac_labels.csv \
    --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

try:
    import h5py
except ImportError:
    h5py = None


MODEL_FILES = {
    "M1_immune_sensitive": "M1_immune_sensitive.pt",
    "M2_msi": "M2_msi.pt",
    "M3_ebv": "M3_ebv.pt",
    "M4_subtype4": "M4_subtype4.pt",
}

SUBTYPE4_NAMES = ["EBV", "MSI", "GS", "CIN"]
SUBTYPE4_TO_ID = {
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


class ABMILClassifier(nn.Module):
    def __init__(self, in_dim=1536, hidden=256, n_classes=2, dropout=0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.att_V = nn.Linear(hidden, 128)
        self.att_U = nn.Linear(hidden, 128)
        self.att_w = nn.Linear(128, 1)
        self.clf = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.proj(x)
        v = torch.tanh(self.att_V(x))
        u = torch.sigmoid(self.att_U(x))
        a = F.softmax(self.att_w(v * u).squeeze(-1), dim=0)
        z = (x * a.unsqueeze(-1)).sum(0)
        return self.clf(z).unsqueeze(0), a


def patient_id_from_slide(slide_id: str) -> str:
    if slide_id.startswith("TCGA-"):
        return "-".join(slide_id.split("-")[:3])
    m = re.match(r"^(C3[NL]-\d+)", slide_id)
    if m:
        return m.group(1)
    parts = slide_id.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else slide_id


def list_feature_files(feature_dir: Path, pattern: str) -> list[Path]:
    files = sorted(p for p in feature_dir.rglob(pattern) if p.suffix.lower() in {".h5", ".hdf5", ".pt", ".pth", ".npy", ".npz"})
    return [p for p in files if p.is_file()]


def _first_existing_h5_dataset(h: Any, keys: list[str]):
    for key in keys:
        if key in h:
            return h[key][()]
    found = []
    h.visititems(lambda name, obj: found.append(name) if getattr(obj, "shape", None) is not None else None)
    for name in found:
        arr = h[name]
        if len(arr.shape) >= 2 and arr.shape[-1] == 1536:
            return arr[()]
    raise KeyError(f"no feature dataset found; checked keys={keys}, available={found[:20]}")


def load_features(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        if h5py is None:
            raise RuntimeError("h5py is required for .h5 feature files")
        with h5py.File(path, "r") as h:
            arr = _first_existing_h5_dataset(h, ["features", "feats", "embeddings", "x"])
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            arr = obj.numpy()
        elif isinstance(obj, dict):
            for key in ["features", "feats", "embeddings", "x"]:
                if key in obj:
                    val = obj[key]
                    arr = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.asarray(val)
                    break
            else:
                raise KeyError(f"{path} has no feature key; available={list(obj.keys())[:20]}")
        else:
            arr = np.asarray(obj)
    elif suffix == ".npz":
        obj = np.load(path)
        key = "features" if "features" in obj else obj.files[0]
        arr = obj[key]
    elif suffix == ".npy":
        arr = np.load(path)
    else:
        raise ValueError(f"unsupported feature file: {path}")

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{path} features must be 2D or (1,N,D), got shape={arr.shape}")
    if arr.shape[1] != 1536:
        raise ValueError(f"{path} feature dim must be 1536, got shape={arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{path} has zero patches")
    return arr


def maybe_subsample(features: np.ndarray, max_patches: int | None, seed: int, slide_id: str) -> np.ndarray:
    if not max_patches or features.shape[0] <= max_patches:
        return features
    stable = abs(hash(slide_id)) % (2**32)
    rng = np.random.default_rng(seed + stable)
    idx = rng.choice(features.shape[0], max_patches, replace=False)
    idx.sort()
    return features[idx]


def load_abmil_panel(model_dir: Path, device: torch.device):
    panel = {}
    for name, fname in MODEL_FILES.items():
        ckpt_path = model_dir / fname
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cfg = ckpt.get("config", {})
        model = ABMILClassifier(
            in_dim=int(cfg.get("in_dim", 1536)),
            hidden=int(cfg.get("hidden", 256)),
            n_classes=int(cfg.get("n_classes", 2)),
            dropout=float(cfg.get("dropout", 0.25)),
        )
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval().to(device)
        panel[name] = model
    return panel


@torch.inference_mode()
def predict_one(features: np.ndarray, panel, device: torch.device) -> dict[str, Any]:
    x = torch.from_numpy(features).to(device, non_blocking=True)
    out = {}
    for name, model in panel.items():
        logits, attn = model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        out[name] = probs
        if name == "M1_immune_sensitive":
            top = torch.topk(attn.detach().cpu(), k=min(20, attn.numel())).indices.numpy()
            out["M1_top_attention_idx"] = ";".join(map(str, top.tolist()))
    return out


def slide_row(feature_path: Path, n_patches: int, pred: dict[str, Any]) -> dict[str, Any]:
    slide_id = feature_path.stem
    row = {
        "slide_id": slide_id,
        "patient_id": patient_id_from_slide(slide_id),
        "feature_path": str(feature_path),
        "n_patches": int(n_patches),
    }
    for model_name in MODEL_FILES:
        probs = pred[model_name]
        for i, p in enumerate(probs):
            row[f"{model_name}_prob_c{i}"] = float(p)
        row[f"{model_name}_pred_class"] = int(np.argmax(probs))
    row["immune_sensitive_prob"] = row["M1_immune_sensitive_prob_c1"]
    row["immune_sensitive_pred"] = int(row["immune_sensitive_prob"] >= 0.5)
    row["msi_prob"] = row["M2_msi_prob_c1"]
    row["msi_pred"] = int(row["msi_prob"] >= 0.5)
    row["ebv_prob"] = row["M3_ebv_prob_c1"]
    row["ebv_pred"] = int(row["ebv_prob"] >= 0.5)
    row["subtype4_pred"] = SUBTYPE4_NAMES[row["M4_subtype4_pred_class"]]
    row["M1_top_attention_idx"] = pred.get("M1_top_attention_idx", "")
    return row


def aggregate_patient(slide_df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [c for c in slide_df.columns if "_prob_c" in c]
    keep = ["patient_id", *prob_cols, "immune_sensitive_prob", "msi_prob", "ebv_prob"]
    patient_df = slide_df[keep].groupby("patient_id", as_index=False).mean(numeric_only=True)
    patient_df["immune_sensitive_pred"] = (patient_df["immune_sensitive_prob"] >= 0.5).astype(int)
    patient_df["msi_pred"] = (patient_df["msi_prob"] >= 0.5).astype(int)
    patient_df["ebv_pred"] = (patient_df["ebv_prob"] >= 0.5).astype(int)
    m4_cols = [f"M4_subtype4_prob_c{i}" for i in range(4)]
    patient_df["subtype4_pred_class"] = patient_df[m4_cols].to_numpy().argmax(axis=1).astype(int)
    patient_df["subtype4_pred"] = [SUBTYPE4_NAMES[i] for i in patient_df["subtype4_pred_class"]]

    n_slides = slide_df.groupby("patient_id").size().rename("n_slides").reset_index()
    n_patches = slide_df.groupby("patient_id")["n_patches"].sum().rename("n_patches_total").reset_index()
    return patient_df.merge(n_slides, on="patient_id").merge(n_patches, on="patient_id")


def _to_binary(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.upper()
    pos = {"1", "TRUE", "YES", "Y", "POS", "POSITIVE", "MSI", "MSI-H", "MSI_H", "EBV", "EBV+", "IMMUNE_SENSITIVE"}
    neg = {"0", "FALSE", "NO", "N", "NEG", "NEGATIVE", "MSS", "EBV-", "NON_SENSITIVE", "NONSENSITIVE"}
    return text.map(lambda v: 1 if v in pos else (0 if v in neg else np.nan))


def add_derived_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    if "subtype" in labels.columns:
        st = labels["subtype"].astype(str).str.strip().str.upper()
        if "immune_sensitive" not in labels.columns:
            labels["immune_sensitive"] = st.map(lambda x: 1 if x in {"STAD_MSI", "MSI", "MSI-H", "MSI_H", "STAD_EBV", "EBV", "EBV+"} else (0 if x in {"STAD_GS", "GS", "STAD_CIN", "CIN"} else np.nan))
        if "msi" not in labels.columns:
            labels["msi"] = st.map(lambda x: 1 if x in {"STAD_MSI", "MSI", "MSI-H", "MSI_H"} else (0 if x in {"STAD_EBV", "EBV", "EBV+", "STAD_GS", "GS", "STAD_CIN", "CIN"} else np.nan))
        if "ebv" not in labels.columns:
            labels["ebv"] = st.map(lambda x: 1 if x in {"STAD_EBV", "EBV", "EBV+"} else (0 if x in {"STAD_MSI", "MSI", "MSI-H", "MSI_H", "STAD_GS", "GS", "STAD_CIN", "CIN"} else np.nan))
        if "subtype4" not in labels.columns:
            labels["subtype4"] = st.map(lambda x: SUBTYPE4_TO_ID.get(x, np.nan))
    return labels


def binary_metrics(df: pd.DataFrame, label_col: str, score_col: str, pred_col: str) -> dict[str, Any]:
    y = _to_binary(df[label_col])
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
    out.update(
        {
            "accuracy": float(accuracy_score(yv, pv)),
            "balanced_accuracy": float(balanced_accuracy_score(yv, pv)) if len(np.unique(yv)) == 2 else None,
            "f1": float(f1_score(yv, pv, zero_division=0)),
            "precision": float(precision_score(yv, pv, zero_division=0)),
            "recall_sensitivity": float(recall_score(yv, pv, zero_division=0)),
            "confusion_matrix": confusion_matrix(yv, pv, labels=[0, 1]).tolist(),
        }
    )
    tn, fp, fn, tp = confusion_matrix(yv, pv, labels=[0, 1]).ravel()
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else None
    if len(np.unique(yv)) == 2:
        out["auc"] = float(roc_auc_score(yv, sv))
        out["ap"] = float(average_precision_score(yv, sv))
    else:
        out["auc"] = None
        out["ap"] = None
        out["warning"] = "need both classes for AUC/AP"
    return out


def subtype4_metrics(df: pd.DataFrame) -> dict[str, Any]:
    y = df["subtype4"]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype(str).str.strip().str.upper().map(lambda x: SUBTYPE4_TO_ID.get(x, np.nan))
    y = pd.to_numeric(y, errors="coerce")
    p = pd.to_numeric(df["subtype4_pred_class"], errors="coerce")
    mask = y.notna() & p.notna()
    yv = y[mask].astype(int).to_numpy()
    pv = p[mask].astype(int).to_numpy()
    out: dict[str, Any] = {"n": int(len(yv)), "classes": SUBTYPE4_NAMES}
    if len(yv) == 0:
        out["error"] = "no labeled samples"
        return out
    out["accuracy"] = float(accuracy_score(yv, pv))
    out["macro_f1"] = float(f1_score(yv, pv, average="macro", zero_division=0))
    out["confusion_matrix"] = confusion_matrix(yv, pv, labels=[0, 1, 2, 3]).tolist()
    prob_cols = [f"M4_subtype4_prob_c{i}" for i in range(4)]
    if len(np.unique(yv)) >= 2:
        try:
            out["macro_auc_ovr"] = float(roc_auc_score(yv, df.loc[mask, prob_cols].to_numpy(), multi_class="ovr", average="macro"))
        except ValueError as e:
            out["macro_auc_ovr"] = None
            out["warning"] = str(e)
    return out


def compute_metrics(patient_df: pd.DataFrame, labels_csv: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = pd.read_csv(labels_csv)
    if "patient_id" not in labels.columns:
        raise ValueError("--labels_csv must contain patient_id")
    labels = add_derived_labels(labels)
    df = patient_df.merge(labels, on="patient_id", how="inner", suffixes=("", "_label"))
    metrics: dict[str, Any] = {
        "n_pred_patients": int(patient_df["patient_id"].nunique()),
        "n_labeled_patients": int(len(df)),
        "tasks": {},
    }
    task_map = {
        "M1_immune_sensitive": ("immune_sensitive", "immune_sensitive_prob", "immune_sensitive_pred"),
        "M2_msi": ("msi", "msi_prob", "msi_pred"),
        "M3_ebv": ("ebv", "ebv_prob", "ebv_pred"),
    }
    for task, (label_col, score_col, pred_col) in task_map.items():
        if label_col in df.columns:
            metrics["tasks"][task] = binary_metrics(df, label_col, score_col, pred_col)
    if "subtype4" in df.columns:
        metrics["tasks"]["M4_subtype4"] = subtype4_metrics(df)
    return df, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=Path, required=True, help="Directory containing per-slide feature files.")
    parser.add_argument("--model_dir", type=Path, default=Path("models"))
    parser.add_argument("--out_dir", type=Path, default=Path("results/external_cptac_feature_eval"))
    parser.add_argument("--labels_csv", type=Path, default=None, help="Optional labels CSV with patient_id and label columns.")
    parser.add_argument("--pattern", default="*", help="Feature filename glob; default recursively scans all supported files.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_patches", type=int, default=None, help="Optional deterministic subsampling for very large bags.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_slides", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = list_feature_files(args.feature_dir, args.pattern)
    if args.max_slides:
        files = files[: args.max_slides]
    if not files:
        raise SystemExit(f"No feature files found in {args.feature_dir} with pattern={args.pattern}")

    print(f"Found {len(files)} feature files. device={device}", flush=True)
    panel = load_abmil_panel(args.model_dir, device)

    rows = []
    errors = []
    for path in tqdm(files, desc="features"):
        try:
            feats = load_features(path)
            feats = maybe_subsample(feats, args.max_patches, args.seed, path.stem)
            pred = predict_one(feats, panel, device)
            rows.append(slide_row(path, feats.shape[0], pred))
        except Exception as e:
            errors.append({"feature_path": str(path), "slide_id": path.stem, "error": repr(e)})

    if rows:
        slide_df = pd.DataFrame(rows).sort_values(["patient_id", "slide_id"])
        patient_df = aggregate_patient(slide_df).sort_values("patient_id")
        slide_out = args.out_dir / "external_feature_slide_predictions.csv"
        patient_out = args.out_dir / "external_feature_patient_predictions.csv"
        slide_df.to_csv(slide_out, index=False)
        patient_df.to_csv(patient_out, index=False)
        print(f"Wrote slide predictions: {slide_out} rows={len(slide_df)}", flush=True)
        print(f"Wrote patient predictions: {patient_out} rows={len(patient_df)}", flush=True)

        if args.labels_csv:
            labeled_df, metrics = compute_metrics(patient_df, args.labels_csv)
            labeled_out = args.out_dir / "external_feature_patient_predictions_labeled.csv"
            metrics_out = args.out_dir / "external_feature_metrics.json"
            labeled_df.to_csv(labeled_out, index=False)
            metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"Wrote labeled predictions: {labeled_out} rows={len(labeled_df)}", flush=True)
            print(f"Wrote metrics: {metrics_out}", flush=True)
            print(json.dumps(metrics, indent=2), flush=True)

    err_out = args.out_dir / "external_feature_errors.json"
    err_out.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    if errors:
        print(f"Wrote errors: {err_out} n={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
