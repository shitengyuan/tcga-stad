#!/usr/bin/env python3
"""
run_threshold_eval.py
═════════════════════
B方案: 在 OOF 预测上找最优阈值, 报临床相关指标 (敏感度/特异度/PPV/NPV)。

AUC 看排序能力, 但临床决策看阈值。对 class imbalance, 阈值 0.5 不是最优。
用 Youden's J (sensitivity + specificity - 1) 选阈值, 并报多个操作点。

用法:
  python run_threshold_eval.py
"""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("thresh")

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

# 二分类模型的 OOF 文件 (M1旧格式prob, M2/M3新格式prob_c1)
BINARY_MODELS = [
    ("M1_immune_sensitive", "oof_preds_M1_immune_sensitive.csv", "prob_c1"),
    ("M2_msi", "oof_preds_M2_msi.csv", "prob_c1"),
    ("M3_ebv", "oof_preds_M3_ebv.csv", "prob_c1"),
    ("M5_clinical", "oof_preds_M5_clinical.csv", "prob_c1"),
]


def find_thresholds(y, p):
    """找多个操作点的阈值。"""
    fpr, tpr, thresholds = roc_curve(y, p)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    youden_thresh = thresholds[best_idx]

    # 高敏感度阈值 (召回>=0.9, 临床初筛不漏诊)
    high_sens_idx = np.where(tpr >= 0.9)[0]
    high_sens_thresh = thresholds[high_sens_idx[0]] if len(high_sens_idx) else thresholds[0]

    # 高特异度阈值 (特异度>=0.9, 确诊不误诊)
    high_spec_idx = np.where((1-fpr) >= 0.9)[0]
    high_spec_thresh = thresholds[high_spec_idx[-1]] if len(high_spec_idx) else thresholds[-1]

    return {
        "youden": float(youden_thresh),
        "high_sensitivity": float(high_sens_thresh),  # 召回>=0.9
        "high_specificity": float(high_spec_thresh),  # 特异>=0.9
    }


def metrics_at(y, p, thresh):
    pred = (p >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp+fn) else 0  # 敏感类召回
    spec = tn / (tn + fp) if (tn+fp) else 0  # 不敏感类召回
    ppv = tp / (tp + fp) if (tp+fp) else 0   # 阳性预测值
    npv = tn / (tn + fn) if (tn+fn) else 0   # 阴性预测值
    return {
        "threshold": round(float(thresh), 3),
        "sensitivity": round(float(sens), 3),  # 敏感类召回 (不漏诊)
        "specificity": round(float(spec), 3),  # 不敏感类召回 (不误诊)
        "ppv": round(float(ppv), 3),
        "npv": round(float(npv), 3),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def main():
    all_results = {}
    for name, fname, prob_col in BINARY_MODELS:
        path = RESULTS / fname
        if not path.exists():
            log.warning(f"{name}: {fname} 不存在, 跳过")
            continue
        df = pd.read_csv(path)
        y = df["label"].values
        p = df[prob_col].values
        if len(np.unique(y)) < 2:
            continue

        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        threshs = find_thresholds(y, p)

        log.info(f"\n{'='*60}\n{name}: AUC={auc:.3f} AP={ap:.3f} (n={len(y)}, 正例={y.sum()})")
        log.info(f"  概率分布: 正例中位{np.median(p[y==1]):.3f}, 负例中位{np.median(p[y==0]):.3f}")

        ops = {}
        for label, t in [("youden", threshs["youden"]),
                         ("high_sensitivity(>=0.9)", threshs["high_sensitivity"]),
                         ("high_specificity(>=0.9)", threshs["high_specificity"]),
                         ("default_0.5", 0.5)]:
            m = metrics_at(y, p, t)
            ops[label] = m
            log.info(f"  [{label}] thresh={m['threshold']}: "
                     f"sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
                     f"ppv={m['ppv']:.3f} npv={m['npv']:.3f} "
                     f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

        all_results[name] = {
            "n": len(y), "n_pos": int(y.sum()), "auc": float(auc), "ap": float(ap),
            "prob_median_pos": float(np.median(p[y==1])),
            "prob_median_neg": float(np.median(p[y==0])),
            "thresholds": threshs, "operating_points": ops,
        }

    out_path = RESULTS / "threshold_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"\n保存: {out_path}")

    # 汇总表
    log.info(f"\n{'='*60}\n汇总 (Youden 阈值):")
    log.info(f"{'模型':25s} {'AUC':6s} {'阈值':6s} {'敏感度':7s} {'特异度':7s} {'PPV':6s} {'NPV':6s}")
    for name, r in all_results.items():
        op = r["operating_points"]["youden"]
        log.info(f"{name:25s} {r['auc']:.3f}  {op['threshold']:.3f}  "
                 f"{op['sensitivity']:.3f}   {op['specificity']:.3f}   "
                 f"{op['ppv']:.3f}  {op['npv']:.3f}")


if __name__ == "__main__":
    main()
