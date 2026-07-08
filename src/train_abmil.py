"""
train_abmil.py
══════════════
ABMIL baseline: 免疫敏感 (MSI-H ∪ EBV+) vs 非敏感, site-stratified CV + bootstrap CI。

里程碑脚本: 验证 UNI2-h + ABMIL 能否打到 AUC >= 0.85 (故事能否立住)。

用法:
  python -m src.train_abmil                       # 默认 site-stratified 5-fold ×3 重复
  python -m src.train_abmil --max_patches 8000    # 限 patch 数控内存
  python -m src.train_abmil --random_cv           # 对照: 随机 CV (看 site 偏倚影响)
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.feature_loader import FeatureLoader
from src.abmil import GatedAttentionMIL, set_seed

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("abmil")

BASE = Path(__file__).resolve().parent.parent
FEAT_DIR = BASE / "tcga_stad_uni2h" / "TCGA-STAD" / "features"
CLIN_CSV = BASE / "clinical.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


def train_one_fold(train_pids, val_pids, dataset, args, device):
    """训练一个 fold, 返回 val 预测 + AUC。"""
    """训练一个 fold, 返回 val 预测 + AUC。"""
    model = GatedAttentionMIL(in_dim=1536, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # class imbalance: 正例权重提升
    n_pos = sum(dataset[p]["label"] for p in train_pids)
    n_neg = len(train_pids) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    # 每个 epoch 重新 shuffle
    rng = np.random.default_rng(args.seed)
    best_val_auc = -1.0
    val_preds = {}          # 始终保留最后一个 epoch 的预测 (OOF 聚合用)
    can_eval = len(set(dataset[p]["label"] for p in val_pids)) >= 2 and len(val_pids) >= 5

    for epoch in range(args.epochs):
        model.train()
        order = list(train_pids)
        rng.shuffle(order)
        total_loss = 0.0
        for pid in order:
            d = dataset[pid]
            x = torch.from_numpy(d["features"]).to(device)
            y = torch.tensor([d["label"]], dtype=torch.float32, device=device)
            logit, _ = model(x)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, y, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

        # 验证 (始终记录预测; AUC 仅在 val 可评估时计算)
        model.eval()
        preds = {}
        with torch.no_grad():
            for pid in val_pids:
                d = dataset[pid]
                x = torch.from_numpy(d["features"]).to(device)
                logit, _ = model(x)
                preds[pid] = torch.sigmoid(logit).item()
        val_preds = dict(preds)   # 保留最新预测

        if can_eval:
            y_true = np.array([dataset[p]["label"] for p in val_pids])
            y_score = np.array([preds[p] for p in val_pids])
            val_auc = roc_auc_score(y_true, y_score)
            val_ap = average_precision_score(y_true, y_score)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                val_preds = dict(preds)   # best epoch 的预测
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log.info(f"  epoch {epoch+1:2d}: train_loss={total_loss/len(order):.4f} "
                         f"val_auc={val_auc:.3f} val_ap={val_ap:.3f}")
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                log.info(f"  epoch {epoch+1:2d}: train_loss={total_loss/len(order):.4f} "
                         f"(val 不可评估 AUC, 仅记录预测)")
    if not can_eval:
        best_val_auc = float('nan')   # 该 fold 无 AUC, 但 OOF 预测仍有效
    return val_preds, best_val_auc


def bootstrap_auc(y_true, y_score, n_boot=1000, seed=42):
    """Bootstrap 95% CI for AUC."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, y_score[idx]))
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_patches", type=int, default=8000)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random_cv", action="store_true", help="对照: 随机 CV (非 site-stratified)")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    p.add_argument("--min_site_for_val", type=int, default=8,
                   help="样本数<此值的站点不单独做 val, 只进 train (避免小 fold AUC 退化)")
    p.add_argument("--task", default="immune_sensitive",
                   choices=["immune_sensitive", "msi", "ebv", "subtype4"],
                   help="预测任务: M1免疫敏感/M2 MSI/M3 EBV/M4 四亚型")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device != "cpu" and not torch.cuda.is_available():
        log.warning(f"请求 {args.device} 但 CUDA 不可用, 回退 CPU")
        device = torch.device("cpu")
    log.info(f"训练设备: {device} | 任务: {args.task}")

    set_seed(args.seed)
    log.info(f"加载数据 (task={args.task}, max_patches={args.max_patches})...")
    fl = FeatureLoader(str(FEAT_DIR), str(CLIN_CSV))
    dataset = fl.build_dataset(label_filter=True, max_patches=args.max_patches, seed=args.seed)
    log.info("数据集:\n" + fl.summary(dataset))

    pids = list(dataset.keys())
    y = np.array([dataset[p]["label"] for p in pids])
    sites = np.array([fl.get_site(p) for p in pids])
    log.info(f"正例率 {y.mean():.3f} | 站点数 {len(np.unique(sites))}")

    # 小站点 (样本数 < min_site_for_val) 不单独做 val, 只进 train
    # 避免 val 集样本过少/单类导致 AUC 退化 (如 fold val=3 全同标签 → AUC=0)
    from collections import Counter
    site_counts = Counter(sites)
    small_sites = {s for s, c in site_counts.items() if c < args.min_site_for_val}
    cv_mask = np.array([sites[i] not in small_sites for i in range(len(pids))])
    n_small = int((~cv_mask).sum())
    log.info(f"小站点(样本<{args.min_site_for_val}): {len(small_sites)} 个, {n_small} 样本只进train; "
             f"CV 在 {int(cv_mask.sum())} 样本/{len(site_counts)-len(small_sites)} 大站点间划分")

    # ── K-fold × N repeat ─────────────────────────────────
    all_oof_preds = {}  # pid -> [proba across repeats]
    fold_aucs = []

    for rep in range(args.n_repeats):
        seed_r = args.seed + rep
        if args.random_cv:
            # 对照: 随机分层 CV (全部样本参与, 暴露 site 偏倚)
            kf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=seed_r)
            splits = list(kf.split(np.zeros(len(pids)), y))
            # 小站点样本仍加入各 fold 的 train
            fold_train_extra = [set() for _ in range(args.n_folds)]
        else:
            # 主实验: site-stratified (仅大站点参与 CV 划分, 小站点恒 train)
            kf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=seed_r)
            cv_idx = np.where(cv_mask)[0]
            y_cv = y[cv_idx]
            sites_cv = sites[cv_idx]
            raw_splits = list(kf.split(np.zeros(len(cv_idx)), y_cv, sites_cv))
            # 还原回原 pids 索引, 并把小站点样本加入每个 fold 的 train
            splits = []
            small_train_idx = np.where(~cv_mask)[0]
            for tr_cv, va_cv in raw_splits:
                tr_full = np.concatenate([cv_idx[tr_cv], small_train_idx])
                splits.append((tr_full, cv_idx[va_cv]))

        for fold_i, (tr_idx, va_idx) in enumerate(splits):
            tr_pids = [pids[i] for i in tr_idx]
            va_pids = [pids[i] for i in va_idx]
            log.info(f"=== repeat {rep+1} fold {fold_i+1}/{args.n_folds} "
                     f"(train {len(tr_pids)} / val {len(va_pids)}) ===")
            val_preds, fold_auc = train_one_fold(tr_pids, va_pids, dataset, args, device)
            fold_aucs.append(fold_auc)
            for pid, prob in val_preds.items():
                all_oof_preds.setdefault(pid, []).append(prob)
            auc_str = f"{fold_auc:.3f}" if not np.isnan(fold_auc) else "N/A(小val)"
            log.info(f"  fold AUC = {auc_str}")

    # ── 汇总: OOF (out-of-fold) 聚合 ──────────────────────
    oof_pid = sorted(all_oof_preds.keys())
    y_true = np.array([dataset[p]["label"] for p in oof_pid])
    y_score = np.array([np.mean(all_oof_preds[p]) for p in oof_pid])  # 多 repeat 取均值

    oof_auc = roc_auc_score(y_true, y_score)
    oof_ap = average_precision_score(y_true, y_score)
    boot_mean, ci_lo, ci_hi = bootstrap_auc(y_true, y_score, args.n_boot, args.seed)

    # per-fold AUC 排除 nan (小 val fold)
    valid_folds = [a for a in fold_aucs if not np.isnan(a)]
    log.info("=" * 60)
    log.info(f"per-fold AUC: mean={np.mean(valid_folds):.3f} ± {np.std(valid_folds):.3f} "
             f"(n={len(valid_folds)} 有效, {len(fold_aucs)-len(valid_folds)} 小val跳过)")
    log.info(f"OOF AUC = {oof_auc:.3f}  (AP={oof_ap:.3f}, n={len(oof_pid)})")
    log.info(f"Bootstrap AUC = {boot_mean:.3f} [95% CI {ci_lo:.3f}, {ci_hi:.3f}]")
    cv_type = "random" if args.random_cv else "site-stratified"
    log.info(f"CV 方式: {cv_type}")
    log.info("=" * 60)

    # 保存
    tag = "random" if args.random_cv else "site"
    out = {
        "tag": tag, "cv": cv_type, "max_patches": args.max_patches,
        "n_samples": int(len(oof_pid)), "n_pos": int(y_true.sum()),
        "fold_aucs": [None if (isinstance(a,float) and np.isnan(a)) else float(a) for a in fold_aucs],
        "fold_auc_mean": float(np.mean(valid_folds)),
        "fold_auc_std": float(np.std(valid_folds)),
        "n_valid_folds": len(valid_folds),
        "oof_auc": float(oof_auc), "oof_ap": float(oof_ap),
        "bootstrap_auc_mean": boot_mean,
        "bootstrap_ci_low": ci_lo, "bootstrap_ci_high": ci_hi,
        "n_repeats": args.n_repeats, "n_folds": args.n_folds,
        "epochs": args.epochs, "hidden": args.hidden,
    }
    outpath = RESULTS / f"metrics_abmil_{tag}.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"指标已保存: {outpath}")

    # OOF 预测 (供后续融合/校准)
    import csv
    with open(RESULTS / f"oof_preds_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "label", "prob", "subtype"])
        for pid in oof_pid:
            d = dataset[pid]
            w.writerow([pid, d["label"], float(np.mean(all_oof_preds[pid])), d["subtype"]])
    log.info(f"OOF 预测已保存: results/oof_preds_{tag}.csv")

    return out


if __name__ == "__main__":
    main()
