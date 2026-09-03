#!/usr/bin/env python3
"""GPU jobs for remaining computational deliverables.

Outputs are kept under results/locked_release_20260902/gpu_missing so the
current M1-M4 official ABMIL results are not overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize


ROOT = Path(__file__).resolve().parent
LOCKED = ROOT / "results" / "locked_release_20260902"
OUT = LOCKED / "gpu_missing"
FEATURE_DIR = ROOT / "tcga_stad_uni2h" / "TCGA-STAD" / "features"
TASKS = {
    "immune_sensitive": {
        "name": "M1_immune_sensitive",
        "label_col": "M1_label",
        "n_classes": 2,
        "fold_registry": ROOT / "results" / "fold_registry_M1_immune_sensitive.csv",
    },
    "msi": {
        "name": "M2_msi",
        "label_col": "MSI",
        "n_classes": 2,
        "fold_registry": ROOT / "results" / "fold_registry_M2_msi.csv",
    },
    "ebv": {
        "name": "M3_ebv",
        "label_col": "EBV",
        "n_classes": 2,
        "fold_registry": ROOT / "results" / "fold_registry_M3_ebv.csv",
    },
    "subtype4": {
        "name": "M4_subtype4",
        "label_col": "M4_label",
        "n_classes": 4,
        "fold_registry": ROOT / "results" / "fold_registry_M4_subtype4.csv",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


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


def fit_clinical_transform(train_pids: list[str], all_pids: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    clinical = pd.read_csv(ROOT / "clinical.csv").set_index("patient_id")
    raw = build_clinical_raw(clinical)
    numeric = ["age", "ln_examined"]
    categorical = ["lauren", "sex", "stage", "primary_site"]
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_onehot())]), categorical),
        ]
    )
    pre.fit(raw.loc[train_pids])
    arr = pre.transform(raw.loc[all_pids]).astype(np.float32)
    return {pid: arr[i] for i, pid in enumerate(all_pids)}, list(raw.columns)


def load_features(slide_id: str, max_patches: int, seed: int) -> np.ndarray:
    path = FEATURE_DIR / f"{slide_id}.h5"
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as h:
        ds = h["features"]
        n = ds.shape[1] if len(ds.shape) == 3 else ds.shape[0]
        if max_patches and n > max_patches:
            idx = rng.choice(n, max_patches, replace=False)
            idx.sort()
            x = ds[0, idx, :] if len(ds.shape) == 3 else ds[idx, :]
        else:
            x = ds[0] if len(ds.shape) == 3 else ds[:]
    return np.asarray(x, dtype=np.float32)


def build_dataset(task: str, max_patches: int, seed: int) -> dict[str, dict[str, Any]]:
    cfg = TASKS[task]
    manifest = pd.read_csv(LOCKED / "patient_manifest.csv")
    dataset = {}
    for r in manifest.itertuples():
        dataset[r.patient_id] = {
            "patient_id": r.patient_id,
            "slide_id": r.slide_id,
            "site": r.site,
            "subtype": getattr(r, "subtype", ""),
            "label": int(getattr(r, cfg["label_col"])),
            "features": load_features(r.slide_id, max_patches=max_patches, seed=seed),
        }
    return dataset


class GatedMILBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.att_v = nn.Linear(hidden, 128)
        self.att_u = nn.Linear(hidden, 128)
        self.att_w = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        a = F.softmax(self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1), dim=0)
        z = (h * a.unsqueeze(-1)).sum(0)
        return z, a


class FusionMIL(nn.Module):
    def __init__(self, clinical_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.mil = GatedMILBackbone(1536, hidden, dropout)
        self.clin = nn.Sequential(nn.Linear(clinical_dim, 64), nn.ReLU(), nn.Dropout(dropout))
        self.clf = nn.Sequential(nn.Linear(hidden + 64, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        z, _ = self.mil(x)
        if c is None:
            raise ValueError("FusionMIL requires clinical vector")
        cz = self.clin(c)
        return self.clf(torch.cat([z, cz], dim=0)).unsqueeze(0)


class TransformerMIL(nn.Module):
    def __init__(
        self,
        n_classes: int,
        hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1536, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.clf = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        h = self.proj(x).unsqueeze(0)
        cls = self.cls.expand(h.shape[0], -1, -1)
        z = self.encoder(torch.cat([cls, h], dim=1))[:, 0]
        return self.clf(z)


def score_fold(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    if n_classes == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    row_sum = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sum, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sum > 0)
    aucs = []
    for cls in range(n_classes):
        yt = (y_true == cls).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, proba[:, cls]))
    return float(np.mean(aucs)) if aucs else math.nan


def aggregate_metrics(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> dict[str, Any]:
    pred = proba.argmax(axis=1)
    if n_classes == 2:
        return {
            "n": int(len(y_true)),
            "n_pos": int(y_true.sum()),
            "auroc": float(roc_auc_score(y_true, proba[:, 1])),
            "average_precision": float(average_precision_score(y_true, proba[:, 1])),
            "accuracy": float((pred == y_true).mean()),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        }
    row_sum = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sum, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sum > 0)
    aucs = []
    aps = []
    for cls in range(n_classes):
        yt = (y_true == cls).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, proba[:, cls]))
        aps.append(average_precision_score(yt, proba[:, cls]))
    return {
        "n": int(len(y_true)),
        "class_order": ["EBV", "MSI", "GS", "CIN"],
        "auroc_macro_ovr": float(np.mean(aucs)) if aucs else math.nan,
        "average_precision_macro": float(np.mean(aps)) if aps else math.nan,
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def train_one_fold(
    model_kind: str,
    task: str,
    rep: int,
    fold: int,
    train_pids: list[str],
    val_pids: list[str],
    dataset: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], float, int, list[dict[str, Any]], nn.Module, dict[str, np.ndarray] | None]:
    cfg = TASKS[task]
    n_classes = cfg["n_classes"]
    clin_map = None
    if model_kind == "fusion":
        all_pids = sorted(set(train_pids + val_pids))
        clin_map, _ = fit_clinical_transform(train_pids, all_pids)
        model: nn.Module = FusionMIL(len(next(iter(clin_map.values()))), n_classes, args.hidden, args.dropout)
    else:
        model = TransformerMIL(n_classes, args.hidden, args.n_heads, args.n_layers, args.dropout)
    model.to(device)
    labels = [dataset[p]["label"] for p in train_pids]
    counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor(
        [total / (n_classes * max(counts.get(c, 1), 1)) for c in range(n_classes)],
        dtype=torch.float32,
        device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + rep * 100 + fold)
    best_score = -1.0
    best_epoch = args.epochs
    best_state = None
    best_preds: dict[str, np.ndarray] = {}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(train_pids)
        rng.shuffle(order)
        losses = []
        for pid in order:
            x = torch.from_numpy(dataset[pid]["features"]).to(device)
            y = torch.tensor([dataset[pid]["label"]], dtype=torch.long, device=device)
            c = torch.from_numpy(clin_map[pid]).to(device) if clin_map is not None else None
            logits = model(x, c)
            loss = F.cross_entropy(logits, y, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        preds = {}
        with torch.no_grad():
            for pid in val_pids:
                x = torch.from_numpy(dataset[pid]["features"]).to(device)
                c = torch.from_numpy(clin_map[pid]).to(device) if clin_map is not None else None
                preds[pid] = F.softmax(model(x, c), dim=1)[0].detach().cpu().numpy()
        y_val = np.array([dataset[p]["label"] for p in val_pids])
        p_val = np.array([preds[p] for p in val_pids])
        score = score_fold(y_val, p_val, n_classes)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_score": None if math.isnan(score) else score})
        if not math.isnan(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_preds = {k: v.copy() for k, v in preds.items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    elif not best_preds:
        best_preds = preds
    return best_preds, best_score, best_epoch, history, model, clin_map


def run(model_kind: str, task: str, args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    cfg = TASKS[task]
    run_name = f"{model_kind}_{cfg['name']}"
    out_dir = OUT / run_name
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "epoch_logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset(task, args.max_patches, args.seed)
    fr = pd.read_csv(cfg["fold_registry"])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Run this script inside the GPU-visible environment, or pass --device cpu explicitly for a slow CPU smoke run."
        )
    device = torch.device(args.device)
    all_oof: dict[str, list[np.ndarray]] = {}
    fold_registry = []
    fold_scores = []
    for rep in sorted(fr["repeat"].unique()):
        if int(rep) > args.n_repeats:
            continue
        for fold in sorted(fr["fold"].unique()):
            fold_df = fr[(fr["repeat"].eq(rep)) & (fr["fold"].eq(fold))]
            train_pids = [p for p in fold_df[fold_df["split"].eq("train")]["patient_id"].tolist() if p in dataset]
            val_pids = [p for p in fold_df[fold_df["split"].eq("val")]["patient_id"].tolist() if p in dataset]
            if not val_pids:
                continue
            print(f"{run_name} rep{rep} fold{fold}: train={len(train_pids)} val={len(val_pids)} device={device}", flush=True)
            preds, score, best_epoch, history, model, _ = train_one_fold(
                model_kind, task, int(rep), int(fold), train_pids, val_pids, dataset, args, device
            )
            fold_scores.append(score)
            for pid, p in preds.items():
                all_oof.setdefault(pid, []).append(p)
            epoch_log = log_dir / f"{run_name}_rep{int(rep)}_fold{int(fold)}_epochs.csv"
            with epoch_log.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_score"])
                writer.writeheader()
                writer.writerows(history)
            ckpt = ckpt_dir / f"{run_name}_rep{int(rep)}_fold{int(fold)}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kind": model_kind,
                    "task": task,
                    "task_name": cfg["name"],
                    "repeat": int(rep),
                    "fold": int(fold),
                    "best_epoch": int(best_epoch),
                    "best_score": None if math.isnan(score) else float(score),
                    "config": vars(args),
                },
                ckpt,
            )
            for split, ids in [("train", train_pids), ("val", val_pids)]:
                for pid in ids:
                    fold_registry.append(
                        {
                            "model_kind": model_kind,
                            "task": task,
                            "repeat": int(rep),
                            "fold": int(fold),
                            "split": split,
                            "patient_id": pid,
                            "site": dataset[pid]["site"],
                            "label": dataset[pid]["label"],
                            "checkpoint": str(ckpt.relative_to(ROOT)),
                            "epoch_log": str(epoch_log.relative_to(ROOT)),
                        }
                    )
    oof_pids = sorted(all_oof)
    y = np.array([dataset[p]["label"] for p in oof_pids])
    proba = np.array([np.mean(all_oof[p], axis=0) for p in oof_pids])
    metrics = aggregate_metrics(y, proba, cfg["n_classes"])
    metrics.update(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "model_kind": model_kind,
            "task": task,
            "task_name": cfg["name"],
            "analysis_set": "TCGA 246 train / 231 center-isolated OOF using existing fold registries",
            "fold_scores": [None if math.isnan(x) else float(x) for x in fold_scores],
            "fold_mean": float(np.nanmean(fold_scores)),
            "fold_std": float(np.nanstd(fold_scores)),
            "config": vars(args),
        }
    )
    rows = []
    for pid, yy, pv in zip(oof_pids, y, proba):
        row = {"patient_id": pid, "label": int(yy), "subtype": dataset[pid]["subtype"]}
        for ci in range(cfg["n_classes"]):
            row[f"prob_c{ci}"] = float(pv[ci])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"oof_preds_{run_name}.csv", index=False)
    pd.DataFrame(fold_registry).to_csv(out_dir / f"fold_registry_{run_name}.csv", index=False)
    write_json(out_dir / f"metrics_{run_name}.json", metrics)
    print(json.dumps({"run_name": run_name, "metrics": metrics}, ensure_ascii=False, indent=2), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_kind", choices=["transformer", "fusion"], required=True)
    parser.add_argument("--task", choices=list(TASKS), default="immune_sensitive")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_patches", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    args = parser.parse_args()
    if args.model_kind == "fusion" and args.task != "immune_sensitive":
        raise SystemExit("Current fusion deliverable is defined for M1 immune_sensitive only.")
    run(args.model_kind, args.task, args)


if __name__ == "__main__":
    main()
