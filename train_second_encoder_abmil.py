#!/usr/bin/env python3
"""Train ABMIL on second-encoder h5 features using the locked TCGA folds."""
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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parent
LOCKED = ROOT / "results" / "locked_release_20260902"
DEFAULT_FEATURE_DIR = ROOT / "results" / "second_encoder_features_20x256" / "CONCH" / "TCGA-STAD" / "features"
DEFAULT_OUT = LOCKED / "second_encoder" / "conch_abmil"
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
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True, default=str), encoding="utf-8")


def load_h5(path: Path, max_patches: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as h:
        ds = h["features"]
        n = ds.shape[1] if len(ds.shape) == 3 else ds.shape[0]
        if max_patches and n > max_patches:
            idx = rng.choice(n, max_patches, replace=False)
            idx.sort()
            arr = ds[0, idx, :] if len(ds.shape) == 3 else ds[idx, :]
        else:
            arr = ds[0] if len(ds.shape) == 3 else ds[:]
    return np.asarray(arr, dtype=np.float32)


def build_dataset(task: str, feature_dir: Path, max_patches: int, seed: int) -> dict[str, dict[str, Any]]:
    cfg = TASKS[task]
    manifest = pd.read_csv(LOCKED / "patient_manifest.csv")
    dataset = {}
    missing = []
    for r in manifest.itertuples():
        path = feature_dir / f"{r.slide_id}.h5"
        if not path.exists() or path.stat().st_size == 0:
            missing.append(r.slide_id)
            continue
        dataset[r.patient_id] = {
            "patient_id": r.patient_id,
            "slide_id": r.slide_id,
            "site": r.site,
            "subtype": getattr(r, "subtype", ""),
            "label": int(getattr(r, cfg["label_col"])),
            "features": load_h5(path, max_patches, seed),
        }
    if missing:
        print(f"Missing second-encoder features: {len(missing)} slides. First examples: {missing[:5]}", flush=True)
    if not dataset:
        raise RuntimeError(f"No usable feature h5 files under {feature_dir}")
    return dataset


class ABMILClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.att_v = nn.Linear(hidden, 128)
        self.att_u = nn.Linear(hidden, 128)
        self.att_w = nn.Linear(128, 1)
        self.clf = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.proj(x)
        a = F.softmax(self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1), dim=0)
        z = (h * a.unsqueeze(-1)).sum(0)
        return self.clf(z).unsqueeze(0), a


def macro_auc(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        return float(roc_auc_score(y_true, proba[:, 1])) if len(np.unique(y_true)) == 2 else math.nan
    row_sum = proba.sum(axis=1, keepdims=True)
    proba = np.divide(proba, row_sum, out=np.full_like(proba, 1.0 / proba.shape[1]), where=row_sum > 0)
    aucs = []
    for c in range(n_classes):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, proba[:, c]))
    return float(np.mean(aucs)) if aucs else math.nan


def metrics(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> dict[str, Any]:
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
    aps = []
    aucs = []
    for c in range(n_classes):
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, proba[:, c]))
        aps.append(average_precision_score(yt, proba[:, c]))
    return {
        "n": int(len(y_true)),
        "class_order": ["EBV", "MSI", "GS", "CIN"],
        "class_counts": {str(c): int((y_true == c).sum()) for c in range(n_classes)},
        "auroc_macro_ovr": float(np.mean(aucs)) if aucs else math.nan,
        "average_precision_macro": float(np.mean(aps)) if aps else math.nan,
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def train_fold(
    train_pids: list[str],
    val_pids: list[str],
    dataset: dict[str, dict[str, Any]],
    in_dim: int,
    n_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, np.ndarray], float, int, list[dict[str, Any]], ABMILClassifier]:
    model = ABMILClassifier(in_dim, args.hidden, n_classes, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    counts = Counter(dataset[p]["label"] for p in train_pids)
    total = len(train_pids)
    weights = torch.tensor([total / (n_classes * max(counts.get(c, 1), 1)) for c in range(n_classes)], dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    best_score = -1.0
    best_epoch = args.epochs
    best_state = None
    best_preds = {}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(train_pids)
        rng.shuffle(order)
        losses = []
        for pid in order:
            x = torch.from_numpy(dataset[pid]["features"]).to(device)
            y = torch.tensor([dataset[pid]["label"]], dtype=torch.long, device=device)
            logits, _ = model(x)
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
                logits, _ = model(x)
                preds[pid] = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
        y_val = np.array([dataset[p]["label"] for p in val_pids])
        p_val = np.array([preds[p] for p in val_pids])
        score = macro_auc(y_val, p_val, n_classes)
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
    return best_preds, best_score, best_epoch, history, model


def run_task(task: str, args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    cfg = TASKS[task]
    n_classes = cfg["n_classes"]
    dataset = build_dataset(task, args.feature_dir, args.max_patches, args.seed)
    first = next(iter(dataset.values()))["features"]
    in_dim = int(first.shape[1])
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Run in GPU-visible environment or pass --device cpu.")
    device = torch.device(args.device)
    out_dir = args.out_dir / cfg["name"]
    ckpt_dir = out_dir / "checkpoints"
    epoch_dir = out_dir / "epoch_logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    fr = pd.read_csv(cfg["fold_registry"])
    all_oof: dict[str, list[np.ndarray]] = {}
    fold_scores = []
    registry = []
    for rep in sorted(fr["repeat"].unique()):
        if int(rep) > args.n_repeats:
            continue
        for fold in sorted(fr["fold"].unique()):
            fdf = fr[(fr["repeat"].eq(rep)) & (fr["fold"].eq(fold))]
            train_pids = [p for p in fdf[fdf["split"].eq("train")]["patient_id"].tolist() if p in dataset]
            val_pids = [p for p in fdf[fdf["split"].eq("val")]["patient_id"].tolist() if p in dataset]
            if not val_pids:
                continue
            print(f"CONCH+ABMIL {task} rep{rep} fold{fold}: train={len(train_pids)} val={len(val_pids)} in_dim={in_dim} device={device}", flush=True)
            preds, score, best_epoch, history, model = train_fold(
                train_pids, val_pids, dataset, in_dim, n_classes, args, device, args.seed + int(rep) * 100 + int(fold)
            )
            fold_scores.append(score)
            for pid, prob in preds.items():
                all_oof.setdefault(pid, []).append(prob)
            epoch_log = epoch_dir / f"conch_abmil_{cfg['name']}_rep{int(rep)}_fold{int(fold)}_epochs.csv"
            with epoch_log.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_score"])
                writer.writeheader()
                writer.writerows(history)
            ckpt = ckpt_dir / f"conch_abmil_{cfg['name']}_rep{int(rep)}_fold{int(fold)}.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "encoder": args.encoder_name,
                    "task": task,
                    "name": cfg["name"],
                    "in_dim": in_dim,
                    "repeat": int(rep),
                    "fold": int(fold),
                    "best_epoch": int(best_epoch),
                    "best_score": None if math.isnan(best_score) else float(best_score),
                    "config": vars(args),
                },
                ckpt,
            )
            for split, ids in [("train", train_pids), ("val", val_pids)]:
                for pid in ids:
                    registry.append(
                        {
                            "encoder": args.encoder_name,
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
    rows = []
    for pid, yy, pv in zip(oof_pids, y, proba):
        row = {"patient_id": pid, "label": int(yy), "subtype": dataset[pid]["subtype"]}
        for c in range(n_classes):
            row[f"prob_c{c}"] = float(pv[c])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / f"oof_preds_conch_abmil_{cfg['name']}.csv", index=False)
    pd.DataFrame(registry).to_csv(out_dir / f"fold_registry_conch_abmil_{cfg['name']}.csv", index=False)
    payload = metrics(y, proba, n_classes)
    payload.update(
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "encoder": args.encoder_name,
            "feature_dir": str(args.feature_dir),
            "task": task,
            "task_name": cfg["name"],
            "analysis_set": "TCGA locked 246 train / 231 center-isolated OOF; same fold registries as UNI2-h ABMIL",
            "feature_dim": in_dim,
            "fold_scores": [None if math.isnan(x) else float(x) for x in fold_scores],
            "fold_mean": float(np.nanmean(fold_scores)),
            "fold_std": float(np.nanstd(fold_scores)),
            "config": vars(args),
        }
    )
    write_json(out_dir / f"metrics_conch_abmil_{cfg['name']}.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASKS) + ["all"], default="immune_sensitive")
    parser.add_argument("--feature_dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--encoder_name", default="CONCH")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_patches", type=int, default=8000)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(TASKS) if args.task == "all" else [args.task]
    summary = {task: run_task(task, args) for task in tasks}
    write_json(args.out_dir / "conch_abmil_summary.json", summary)


if __name__ == "__main__":
    main()
