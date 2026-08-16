"""
train_survival.py
═════════════════
M6 生存模型: 用 ABMIL 聚合的图像特征预测 OS 风险 (Cox)。

次要终点: 证明图像特征有预后价值 (不只是分子标签预测)。
为避免数据泄露, 不用分子特征, 只用图像特征 + 临床年龄/分期。

用法:
  python -m src.train_survival --device cpu
  python -m src.train_survival --device cuda --max_patches 8000
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.model_selection import GroupKFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from src.feature_loader import FeatureLoader
from src.abmil import set_seed

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("survival")

BASE = Path(__file__).resolve().parent.parent
FEAT_DIR = BASE / "tcga_stad_uni2h" / "TCGA-STAD" / "features"
CLIN_CSV = BASE / "clinical.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


class ABMILEncoder(nn.Module):
    """ABMIL 聚合 patch 特征 -> patient 向量 (无分类头, 用于 Cox 输入)。"""
    def __init__(self, in_dim=1536, hidden=256, dropout=0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.att_V = nn.Linear(hidden, 128)
        self.att_U = nn.Linear(hidden, 128)
        self.att_w = nn.Linear(128, 1)

    def forward(self, x):
        x = self.proj(x)
        v = torch.tanh(self.att_V(x)); u = torch.sigmoid(self.att_U(x))
        a = F.softmax(self.att_w(v * u).squeeze(-1), dim=0)
        return (x * a.unsqueeze(-1)).sum(0)  # (hidden,)


def train_abmil_encoder(dataset, train_pids, args, device):
    """用免疫敏感标签预训练 ABMIL encoder (迁移学习), 返回 encoder。"""
    from collections import Counter
    enc = ABMILEncoder(1536, args.hidden, args.dropout).to(device)
    # 用免疫敏感标签做辅助监督 (迁移)
    clf = nn.Linear(args.hidden, 2).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(clf.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    lbl_c = Counter(dataset[p]["label"] for p in train_pids if "label" in dataset[p])
    # 若无标签, 跳过预训练
    rng = np.random.default_rng(args.seed)
    epoch_history = []
    for epoch in range(args.epochs):
        order = [p for p in train_pids if "label" in dataset[p]]
        rng.shuffle(order)
        total_loss = 0.0
        for pid in order:
            d = dataset[pid]
            x = torch.from_numpy(d["features"]).to(device)
            y = torch.tensor([d["label"]], dtype=torch.long, device=device)
            z = enc(x)
            logit = clf(z.unsqueeze(0))
            loss = F.cross_entropy(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": float(total_loss / len(order)) if order else float("nan"),
        })
    return enc, epoch_history


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max_patches", type=int, default=4000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min_site_for_val", type=int, default=8)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    set_seed(args.seed)

    fl = FeatureLoader(str(FEAT_DIR), str(CLIN_CSV))
    # 用 immune_sensitive 任务的数据集 (有 label 用于预训练)
    dataset = fl.build_dataset(task="immune_sensitive", max_patches=args.max_patches, seed=args.seed)
    log.info(f"数据集: {len(dataset)} 患者")

    # 生存数据
    clin = pd.read_csv(CLIN_CSV).set_index("patient_id")
    surv = {}
    for pid in dataset:
        if pid not in clin.index: continue
        r = clin.loc[pid]
        os_m = pd.to_numeric(r.get("os_months"), errors="coerce")
        os_s = str(r.get("os_status",""))
        # 格式: "0:LIVING" / "1:DECEASED" 或 "LIVING"/"DECEASED"
        event = 1 if ("1" in os_s.split(":")[0] or "deceas" in os_s.lower() or "dead" in os_s.lower()) else 0
        if pd.notna(os_m) and os_m > 0:
            surv[pid] = {"os_months": float(os_m), "os_event": event,
                         "age": pd.to_numeric(r.get("age"), errors="coerce"),
                         "stage": str(r.get("ajcc_pathologic_tumor_stage",""))[:3]}
    log.info(f"有生存数据: {len(surv)} (事件数 {sum(s['os_event'] for s in surv.values())})")

    pids = [p for p in dataset if p in surv]
    sites = np.array([fl.get_site(p) for p in pids])
    from collections import Counter
    sc = Counter(sites)
    small = {s for s,c in sc.items() if c < args.min_site_for_val}
    cv_mask = np.array([s not in small for s in sites])

    cv_idx = np.where(cv_mask)[0]
    small_train = np.where(~cv_mask)[0]
    gkf = GroupKFold(n_splits=args.n_folds)

    oof_risk = np.full(len(pids), np.nan)
    oof_fold = np.full(len(pids), -1)
    fold_cidx = []
    fold_registry = []
    model_dir = BASE / "models" / "per_fold" / "M6_survival"
    model_dir.mkdir(parents=True, exist_ok=True)
    epoch_log_dir = RESULTS / "training_epoch_logs" / "M6_survival"
    epoch_log_dir.mkdir(parents=True, exist_ok=True)

    for fi, (tr_cv, va_cv) in enumerate(gkf.split(np.zeros(len(cv_idx)),
                                                   [surv[pids[c]]["os_event"] for c in cv_idx],
                                                   sites[cv_idx])):
        tr_full = np.concatenate([cv_idx[tr_cv], small_train])
        va_full = cv_idx[va_cv]
        tr_pids = [pids[i] for i in tr_full]
        va_pids = [pids[i] for i in va_full]

        # 1. 训练 ABMIL encoder (免疫敏感辅助监督)
        enc, epoch_history = train_abmil_encoder(dataset, tr_pids, args, device)
        enc.eval()
        epoch_log_path = epoch_log_dir / f"M6_survival_fold{fi+1}_epochs.csv"
        with epoch_log_path.open("w", newline="") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss"])
            w.writeheader()
            w.writerows(epoch_history)

        # 2. 提取 patient 向量
        def embed(pids_sub):
            vecs = []
            with torch.no_grad():
                for pid in pids_sub:
                    x = torch.from_numpy(dataset[pid]["features"]).to(device)
                    vecs.append(enc(x).cpu().numpy())
            return np.array(vecs)

        tr_vec = embed(tr_pids); va_vec = embed(va_pids)

        # 3. Cox 拟合 (图像特征 + age)
        tr_df = pd.DataFrame(tr_vec, columns=[f"v{i}" for i in range(tr_vec.shape[1])])
        tr_df["age"] = [surv[p]["age"] for p in tr_pids]
        tr_df["duration"] = [surv[p]["os_months"] for p in tr_pids]
        tr_df["event"] = [surv[p]["os_event"] for p in tr_pids]
        # PCA 降维 (Cox 对高维敏感)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(16, tr_vec.shape[0]-2, tr_vec.shape[1]))
        tr_pca = pca.fit_transform(tr_vec)
        va_pca = pca.transform(va_vec)
        tr_df = pd.DataFrame(tr_pca, columns=[f"pc{i}" for i in range(tr_pca.shape[1])])
        tr_df["age"] = [surv[p]["age"] for p in tr_pids]
        tr_df["duration"] = [surv[p]["os_months"] for p in tr_pids]
        tr_df["event"] = [surv[p]["os_event"] for p in tr_pids]
        tr_df = tr_df.dropna()

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(tr_df, duration_col="duration", event_col="event")
            va_df = pd.DataFrame(va_pca, columns=[f"pc{i}" for i in range(va_pca.shape[1])])
            va_df["age"] = [surv[p]["age"] for p in va_pids]
            risk = cph.predict_partial_hazard(va_df).values
            oof_risk[va_full] = risk
            oof_fold[va_full] = fi + 1
            yt = [surv[p]["os_event"] for p in va_pids]
            yt_dur = [surv[p]["os_months"] for p in va_pids]
            ci = concordance_index(yt_dur, -risk, yt)  # lifelines: 高risk应短生存
            fold_cidx.append(ci)
            encoder_path = model_dir / f"M6_survival_fold{fi+1}.pt"
            cox_path = model_dir / f"M6_survival_fold{fi+1}_pca_cox.joblib"
            torch.save({
                "encoder_state": enc.state_dict(),
                "config": {"in_dim": 1536, "hidden": args.hidden, "dropout": args.dropout},
                "task": "survival",
                "name": "M6_survival",
                "fold": fi + 1,
                "seed": args.seed,
                "train_patients": tr_pids,
                "val_patients": va_pids,
                "fold_cindex": float(ci),
            }, encoder_path)
            joblib.dump({
                "pca": pca,
                "cox": cph,
                "pca_columns": [f"pc{i}" for i in range(tr_pca.shape[1])],
                "uses_age": True,
                "fold": fi + 1,
                "seed": args.seed,
            }, cox_path)
            for split, split_idxs in (("train", tr_full), ("val", va_full)):
                for idx in split_idxs:
                    pid = pids[idx]
                    fold_registry.append({
                        "task": "survival",
                        "fold": fi + 1,
                        "split": split,
                        "patient_id": pid,
                        "site": sites[idx],
                        "os_months": surv[pid]["os_months"],
                        "os_event": surv[pid]["os_event"],
                        "seed": args.seed,
                        "fold_cindex": float(ci),
                        "encoder_checkpoint": str(encoder_path.relative_to(BASE)),
                        "cox_checkpoint": str(cox_path.relative_to(BASE)),
                        "epoch_log": str(epoch_log_path.relative_to(BASE)),
                    })
            log.info(f"  fold {fi+1}: C-index={ci:.3f} (train {len(tr_pids)} / val {len(va_pids)})")
        except Exception as e:
            log.warning(f"  fold {fi+1} 失败: {e}")

    mask = ~np.isnan(oof_risk)
    if mask.sum() > 0:
        yt_dur = np.array([surv[pids[i]]["os_months"] for i in range(len(pids)) if mask[i]])
        yt_evt = np.array([surv[pids[i]]["os_event"] for i in range(len(pids)) if mask[i]])
        oof_ci = concordance_index(yt_dur, -oof_risk[mask], yt_evt)
        n_events = int(yt_evt.sum())
    else:
        oof_ci = float('nan')
        n_events = 0

    log.info("="*60)
    log.info(f"M6 生存: per-fold C-index {np.mean(fold_cidx):.3f}±{np.std(fold_cidx):.3f}")
    log.info(f"OOF C-index = {oof_ci:.3f} (n={mask.sum()})")
    log.info("="*60)

    out = {"task":"survival","name":"M6_survival","n_classes":1,
           "n_samples": int(mask.sum()), "n_events": n_events,
           "fold_cindex":[float(c) for c in fold_cidx],
           "fold_mean": float(np.mean(fold_cidx)),"fold_std": float(np.std(fold_cidx)),
           "oof_cindex": float(oof_ci),
           "config": {"max_patches":args.max_patches,"epochs":args.epochs,
                      "n_folds": args.n_folds, "seed": args.seed,
                      "hidden": args.hidden, "dropout": args.dropout,
                      "lr": args.lr, "weight_decay": args.weight_decay,
                      "min_site_for_val": args.min_site_for_val},
           "per_fold_checkpoint_dir": "models/per_fold/M6_survival",
           "epoch_log_dir": "results/training_epoch_logs/M6_survival",
           "oof_predictions_csv": "results/oof_preds_M6_survival.csv",
           "fold_registry_csv": "results/fold_registry_M6_survival.csv"}
    with open(RESULTS/"metrics_M6_survival.json","w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    import csv
    with open(RESULTS/"oof_preds_M6_survival.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "site", "os_months", "os_event", "risk", "fold"])
        for i, pid in enumerate(pids):
            if mask[i]:
                w.writerow([pid, sites[i], surv[pid]["os_months"], surv[pid]["os_event"],
                            float(oof_risk[i]), int(oof_fold[i])])
    with open(RESULTS/"fold_registry_M6_survival.csv", "w", newline="") as f:
        fieldnames = ["task", "fold", "split", "patient_id", "site", "os_months",
                      "os_event", "seed", "fold_cindex", "encoder_checkpoint",
                      "cox_checkpoint", "epoch_log"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(fold_registry)
    log.info("保存: metrics_M6_survival.json, oof_preds_M6_survival.csv, fold_registry_M6_survival.csv")


if __name__ == "__main__":
    main()
