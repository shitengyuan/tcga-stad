"""
train_multitask.py
══════════════════
多任务 ABMIL 训练器 —— 模型面板 M1-M4 (图像任务)。

任务:
  M1 immune_sensitive : MSI∪EBV vs 非   (二分类)
  M2 msi              : MSI-H vs MSS    (二分类)
  M3 ebv              : EBV+ vs EBV-    (二分类)
  M4 subtype4         : EBV/MSI/GS/CIN  (四分类)

每个任务独立训练 ABMIL, site-stratified CV, 输出 OOF 预测 + 指标。
agent (run_agent_panel.py) 读取各模型的 OOF 预测做编排。

用法:
  python -m src.train_multitask --task immune_sensitive --device cuda
  python -m src.train_multitask --task msi --device cuda
  python -m src.train_multitask --task all --device cuda   # 跑全部4个
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.feature_loader import FeatureLoader
from src.abmil import set_seed

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("multitask")

BASE = Path(__file__).resolve().parent.parent
FEAT_DIR = BASE / "tcga_stad_uni2h" / "TCGA-STAD" / "features"
CLIN_CSV = BASE / "clinical.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

TASK_CONFIG = {
    "immune_sensitive": {"n_classes": 2, "name": "M1_immune_sensitive"},
    "msi":              {"n_classes": 2, "name": "M2_msi"},
    "ebv":              {"n_classes": 2, "name": "M3_ebv"},
    "subtype4":         {"n_classes": 4, "name": "M4_subtype4"},
}


class ABMILClassifier(nn.Module):
    """ABMIL, 支持二分类/多分类。"""
    def __init__(self, in_dim=1536, hidden=256, n_classes=2, dropout=0.25):
        super().__init__()
        self.n_classes = n_classes
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.att_V = nn.Linear(hidden, 128)
        self.att_U = nn.Linear(hidden, 128)
        self.att_w = nn.Linear(128, 1)
        self.clf = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(64, n_classes))

    def forward(self, x):
        x = self.proj(x)
        v = torch.tanh(self.att_V(x)); u = torch.sigmoid(self.att_U(x))
        a = F.softmax(self.att_w(v * u).squeeze(-1), dim=0)
        z = (x * a.unsqueeze(-1)).sum(0)
        return self.clf(z).unsqueeze(0), a  # (1, n_classes), (N,)


def train_one_fold(tr_pids, va_pids, dataset, args, n_classes, device):
    model = ABMILClassifier(1536, args.hidden, n_classes, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # 类别权重 (处理不平衡)
    from collections import Counter
    lbl_c = Counter(dataset[p]["label"] for p in tr_pids)
    total = sum(lbl_c.values())
    w = torch.tensor([total / (n_classes * max(lbl_c.get(c, 1), 1)) for c in range(n_classes)],
                     dtype=torch.float32, device=device)

    rng = np.random.default_rng(args.seed)
    best_score = -1.0
    best_epoch = None
    best_state = None
    val_preds = {}  # pid -> proba vector (二分类存正类概率, 多分类存各类概率)
    can_eval = len(set(dataset[p]["label"] for p in va_pids)) >= 2 and len(va_pids) >= 5
    epoch_history = []

    for epoch in range(args.epochs):
        model.train()
        order = list(tr_pids); rng.shuffle(order)
        total_loss = 0.0
        for pid in order:
            d = dataset[pid]
            x = torch.from_numpy(d["features"]).to(device)
            y = torch.tensor([d["label"]], dtype=torch.long, device=device)
            logit, _ = model(x)
            loss = F.cross_entropy(logit, y, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        model.eval()
        preds = {}
        with torch.no_grad():
            for pid in va_pids:
                d = dataset[pid]
                x = torch.from_numpy(d["features"]).to(device)
                logit, _ = model(x)
                preds[pid] = F.softmax(logit, dim=1)[0].cpu().numpy()
        val_preds = dict(preds)

        if can_eval:
            y_true = np.array([dataset[p]["label"] for p in va_pids])
            if n_classes == 2:
                y_score = np.array([preds[p][1] for p in va_pids])
                score = roc_auc_score(y_true, y_score)
            else:
                # 多分类: macro AUC (one-vs-rest)
                score = _macro_auc(y_true, np.array([preds[p] for p in va_pids]))
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                val_preds = dict(preds)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log.info(f"  epoch {epoch+1:2d}: loss={total_loss/len(order):.4f} val_score={score:.3f}")
        else:
            score = float("nan")
        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": float(total_loss / len(order)) if order else float("nan"),
            "val_score": None if np.isnan(score) else float(score),
        })
    if not can_eval:
        best_score = float('nan')
        best_epoch = args.epochs
    elif best_state is not None:
        model.load_state_dict(best_state)
    return val_preds, best_score, best_epoch, model, epoch_history


def _macro_auc(y_true, y_proba):
    """多分类 macro AUC (跳过缺失类)。"""
    classes = np.unique(y_true)
    if len(classes) < 2:
        return float('nan')
    aucs = []
    for c in classes:
        if c >= y_proba.shape[1]:
            continue
        yt = (y_true == c).astype(int)
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, y_proba[:, c]))
    return float(np.mean(aucs)) if aucs else float('nan')


def bootstrap_auc(y_true, y_score, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    n = len(y_true); aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def run_task(task, args):
    cfg = TASK_CONFIG[task]
    n_classes = cfg["n_classes"]
    log.info(f"\n{'='*60}\n训练 {cfg['name']} (task={task}, n_classes={n_classes})\n{'='*60}")

    set_seed(args.seed)
    fl = FeatureLoader(str(FEAT_DIR), str(CLIN_CSV))
    dataset = fl.build_dataset(task=task, max_patches=args.max_patches, seed=args.seed)
    log.info("数据集:\n" + fl.summary(dataset))

    pids = list(dataset.keys())
    y = np.array([dataset[p]["label"] for p in pids])
    sites = np.array([fl.get_site(p) for p in pids])

    # 小站点只进 train
    from collections import Counter
    site_counts = Counter(sites)
    small_sites = {s for s, c in site_counts.items() if c < args.min_site_for_val}
    cv_mask = np.array([sites[i] not in small_sites for i in range(len(pids))])
    log.info(f"小站点 {len(small_sites)} 个/{int((~cv_mask).sum())} 样本只进train; "
             f"CV 在 {int(cv_mask.sum())} 样本间划分")

    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    log.info(f"设备: {device}")

    all_oof = {}
    fold_scores = []
    fold_registry = []
    model_dir = BASE / "models"
    per_fold_dir = model_dir / "per_fold" / cfg["name"]
    per_fold_dir.mkdir(parents=True, exist_ok=True)
    epoch_log_dir = RESULTS / "training_epoch_logs" / cfg["name"]
    epoch_log_dir.mkdir(parents=True, exist_ok=True)
    kf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    cv_idx = np.where(cv_mask)[0]
    y_cv = y[cv_mask]; sites_cv = sites[cv_mask]
    small_train = np.where(~cv_mask)[0]

    for rep in range(args.n_repeats):
        seed_r = args.seed + rep
        kf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=seed_r)
        for fold_i, (tr_cv, va_cv) in enumerate(kf.split(np.zeros(len(cv_idx)), y_cv, sites_cv)):
            tr_full = np.concatenate([cv_idx[tr_cv], small_train])
            va_full = cv_idx[va_cv]
            tr_pids = [pids[i] for i in tr_full]
            va_pids = [pids[i] for i in va_full]
            log.info(f"=== {task} rep{rep+1} fold{fold_i+1}/{args.n_folds} "
                     f"(train {len(tr_pids)} / val {len(va_pids)}) ===")
            val_preds, fscore, best_epoch, fold_model, epoch_history = train_one_fold(
                tr_pids, va_pids, dataset, args, n_classes, device
            )
            epoch_log_path = epoch_log_dir / f"{cfg['name']}_rep{rep+1}_fold{fold_i+1}_epochs.csv"
            with epoch_log_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_score"])
                w.writeheader()
                w.writerows(epoch_history)
            fold_scores.append(fscore)
            for pid, prob in val_preds.items():
                all_oof.setdefault(pid, []).append(prob)
            log.info(f"  fold score = {fscore:.3f}" if not np.isnan(fscore) else "  fold score = N/A")
            checkpoint_path = per_fold_dir / f"{cfg['name']}_rep{rep+1}_fold{fold_i+1}.pt"
            torch.save({"model_state": fold_model.state_dict(),
                        "config": {"in_dim": 1536, "hidden": args.hidden,
                                   "n_classes": n_classes, "dropout": args.dropout},
                        "task": task, "name": cfg["name"],
                        "repeat": rep + 1, "fold": fold_i + 1,
                        "seed": seed_r, "best_epoch": best_epoch,
                        "best_score": None if np.isnan(fscore) else float(fscore),
                        "train_patients": tr_pids, "val_patients": va_pids},
                       checkpoint_path)
            for split, split_pids in (("train", tr_pids), ("val", va_pids)):
                for pid in split_pids:
                    fold_registry.append({
                        "task": task,
                        "repeat": rep + 1,
                        "fold": fold_i + 1,
                        "split": split,
                        "patient_id": pid,
                        "site": fl.get_site(pid),
                        "label": dataset[pid]["label"],
                        "subtype": dataset[pid]["subtype"],
                        "small_site_train_only": pid in {pids[i] for i in small_train},
                        "seed": seed_r,
                        "best_epoch": best_epoch,
                        "best_score": None if np.isnan(fscore) else float(fscore),
                        "checkpoint": str(checkpoint_path.relative_to(BASE)),
                        "epoch_log": str(epoch_log_path.relative_to(BASE)),
                    })
            # 保存最后一个 fold 的模型 (兼容旧推理脚本)
            last_model = fold_model

    # OOF 聚合
    oof_pids = sorted(all_oof.keys())
    y_true = np.array([dataset[p]["label"] for p in oof_pids])
    if n_classes == 2:
        y_score = np.array([np.mean([p[1] for p in all_oof[pid]]) for pid in oof_pids])
        oof_auc = roc_auc_score(y_true, y_score)
        oof_ap = average_precision_score(y_true, y_score)
        bm, clo, chi = bootstrap_auc(y_true, y_score, args.n_boot, args.seed)
    else:
        y_proba = np.array([np.mean(all_oof[pid], axis=0) for pid in oof_pids])
        oof_auc = _macro_auc(y_true, y_proba)
        oof_ap = None
        y_pred = y_proba.argmax(1)
        oof_f1 = f1_score(y_true, y_pred, average='macro')
        bm, clo, chi = float('nan'), float('nan'), float('nan')

    valid = [s for s in fold_scores if not np.isnan(s)]
    log.info("="*60)
    log.info(f"{cfg['name']}: per-fold {np.mean(valid):.3f}±{np.std(valid):.3f} (n={len(valid)})")
    if n_classes == 2:
        log.info(f"OOF AUC = {oof_auc:.3f} (AP={oof_ap:.3f})")
        log.info(f"Bootstrap AUC = {bm:.3f} [95% CI {clo:.3f}, {chi:.3f}]")
    else:
        log.info(f"OOF macro-AUC = {oof_auc:.3f} (macroF1={oof_f1:.3f})")
    log.info("="*60)

    # 保存
    out = {"task": task, "name": cfg["name"], "n_classes": n_classes,
           "n_samples": len(oof_pids), "label_dist": dict(Counter(y.tolist())),
           "fold_scores": [None if (isinstance(s,float) and np.isnan(s)) else float(s) for s in fold_scores],
           "fold_mean": float(np.mean(valid)), "fold_std": float(np.std(valid)),
           "oof_auc": float(oof_auc),
           "oof_ap": float(oof_ap) if oof_ap else None,
           "oof_macrof1": float(oof_f1) if n_classes>2 else None,
           "bootstrap_auc": bm, "bootstrap_ci": [clo, chi],
           "config": {"max_patches": args.max_patches, "epochs": args.epochs,
                      "n_repeats": args.n_repeats, "n_folds": args.n_folds,
                      "seed": args.seed, "hidden": args.hidden,
                      "dropout": args.dropout, "lr": args.lr,
                      "weight_decay": args.weight_decay,
                      "min_site_for_val": args.min_site_for_val},
           "per_fold_checkpoint_dir": str(per_fold_dir.relative_to(BASE))}
    out["epoch_log_dir"] = str(epoch_log_dir.relative_to(BASE))
    with open(RESULTS / f"metrics_{cfg['name']}.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)

    # 保存模型权重 (最后一个 fold)
    model_dir.mkdir(exist_ok=True)
    torch.save({"model_state": last_model.state_dict(),
                "config": {"in_dim": 1536, "hidden": args.hidden,
                           "n_classes": n_classes, "dropout": args.dropout},
                "task": task, "name": cfg["name"]},
               model_dir / f"{cfg['name']}.pt")
    log.info(f"模型已保存: models/{cfg['name']}.pt")

    with open(RESULTS / f"fold_registry_{cfg['name']}.csv", "w", newline="") as f:
        fieldnames = ["task", "repeat", "fold", "split", "patient_id", "site", "label",
                      "subtype", "small_site_train_only", "seed", "best_epoch",
                      "best_score", "checkpoint", "epoch_log"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(fold_registry)
    log.info(f"fold registry已保存: results/fold_registry_{cfg['name']}.csv")

    # OOF 预测 (供 agent 编排)
    with open(RESULTS / f"oof_preds_{cfg['name']}.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["patient_id", "label", "subtype"] + [f"prob_c{c}" for c in range(n_classes)]
        w.writerow(header)
        for pid in oof_pids:
            d = dataset[pid]
            proba = np.mean(all_oof[pid], axis=0)
            w.writerow([pid, d["label"], d["subtype"]] + [round(float(p),4) for p in proba])
    log.info(f"保存: metrics_{cfg['name']}.json, oof_preds_{cfg['name']}.csv")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="immune_sensitive",
                   choices=list(TASK_CONFIG.keys()) + ["all"])
    p.add_argument("--max_patches", type=int, default=8000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_boot", type=int, default=500)
    p.add_argument("--device", default="cpu")
    p.add_argument("--min_site_for_val", type=int, default=8)
    args = p.parse_args()

    tasks = list(TASK_CONFIG.keys()) if args.task == "all" else [args.task]
    summary = {}
    for t in tasks:
        summary[t] = run_task(t, args)
    log.info("\n最终汇总:")
    for t, s in summary.items():
        log.info(f"  {s['name']}: OOF AUC={s['oof_auc']:.3f}")


if __name__ == "__main__":
    main()
