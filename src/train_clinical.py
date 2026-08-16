"""
train_clinical.py
═════════════════
M5 临床基线模型: 用临床特征预测免疫敏感 (不依赖图像)。

作为消融基线: 证明图像特征(UNI2-h)相对临床的增量。
site-stratified CV, sklearn LogisticRegression/XGBoost。

用法:
  python -m src.train_clinical
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("clinical")

BASE = Path(__file__).resolve().parent.parent
CLIN_CSV = BASE / "clinical.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """从 clinical.csv 构造临床特征矩阵 (不含分子/生存)。"""
    feat = pd.DataFrame(index=df.index)

    # 数值: age
    feat["age"] = pd.to_numeric(df.get("age"), errors="coerce")

    # Lauren 分型 (从 histological_diagnosis 派生)
    def lauren(h):
        h = str(h).lower()
        if "diffuse" in h or "signet" in h: return "diffuse"
        if "tubular" in h or "papillary" in h or "intestinal" in h: return "intestinal"
        if "mucinous" in h: return "intestinal"
        return "other"
    feat["lauren"] = df.get("histological_diagnosis","").apply(lauren)

    # 类别: sex, stage, primary_site
    feat["sex"] = df.get("sex","").fillna("unknown")
    feat["stage"] = df.get("ajcc_pathologic_tumor_stage","").fillna("unknown").apply(
        lambda s: str(s)[:3] if str(s).startswith("Stage") or str(s).startswith("stage") else str(s))
    feat["site"] = df.get("primary_site_patient","").fillna("unknown")

    # 淋巴结
    feat["ln_examined"] = pd.to_numeric(df.get("lymph_node_examined_count"), errors="coerce")

    # one-hot
    cat = ["lauren","sex","stage","site"]
    feat = pd.get_dummies(feat, columns=cat, dummy_na=True)
    return feat


def main():
    df = pd.read_csv(CLIN_CSV).set_index("patient_id")
    df = df[df["label_immune_sensitive"].isin(["IMMUNE_SENSITIVE","NON_SENSITIVE"])].copy()
    y = (df["label_immune_sensitive"]=="IMMUNE_SENSITIVE").astype(int).values
    sites = np.array([p.split("-")[1] for p in df.index])

    X = build_clinical_features(df)
    log.info(f"临床特征: {X.shape}, 正例率 {y.mean():.3f}, 站点 {len(set(sites))}")

    missing = X.isna().mean().reset_index()
    missing.columns = ["feature", "missing_rate"]
    missing["n_missing"] = X.isna().sum().values
    missing["n_total"] = len(X)
    missing.to_csv(RESULTS / "M5_clinical_feature_missingness.csv", index=False)

    # 小站点只进train
    from collections import Counter
    sc = Counter(sites)
    small = {s for s,c in sc.items() if c < 8}
    cv_mask = np.array([s not in small for s in sites])
    log.info(f"小站点 {len(small)}/{int((~cv_mask).sum())} 样本只进train")

    cv_idx = np.where(cv_mask)[0]
    small_train = np.where(~cv_mask)[0]
    y_cv = y[cv_idx]; sites_cv = sites[cv_idx]

    all_oof = np.full(len(df), np.nan)
    fold_aucs = []
    fold_registry = []
    per_fold_dir = MODEL_DIR / "per_fold" / "M5_clinical"
    per_fold_dir.mkdir(parents=True, exist_ok=True)
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fi, (tr_cv, va_cv) in enumerate(kf.split(np.zeros(len(cv_idx)), y_cv, sites_cv)):
        tr_full = np.concatenate([cv_idx[tr_cv], small_train])
        va_full = cv_idx[va_cv]
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)),
        ])
        pipe.fit(X.iloc[tr_full], y[tr_full])
        proba = pipe.predict_proba(X.iloc[va_full])[:, 1]
        all_oof[va_full] = proba
        auc = roc_auc_score(y[va_full], proba)
        fold_aucs.append(auc)
        checkpoint = per_fold_dir / f"M5_clinical_fold{fi+1}.joblib"
        joblib.dump({
            "pipeline": pipe,
            "feature_columns": list(X.columns),
            "fold": fi + 1,
            "seed": 42,
            "auc": float(auc),
            "train_patients": df.index[tr_full].tolist(),
            "val_patients": df.index[va_full].tolist(),
        }, checkpoint)
        for split, idxs in (("train", tr_full), ("val", va_full)):
            for idx in idxs:
                fold_registry.append({
                    "task": "clinical",
                    "repeat": 1,
                    "fold": fi + 1,
                    "split": split,
                    "patient_id": df.index[idx],
                    "site": sites[idx],
                    "label": int(y[idx]),
                    "seed": 42,
                    "fold_auc": float(auc),
                    "checkpoint": str(checkpoint.relative_to(BASE)),
                })
        log.info(f"  fold {fi+1}: AUC={auc:.3f} (train {len(tr_full)} / val {len(va_full)})")

    mask = ~np.isnan(all_oof)
    oof_auc = roc_auc_score(y[mask], all_oof[mask])
    oof_ap = average_precision_score(y[mask], all_oof[mask])
    # bootstrap
    rng = np.random.default_rng(42)
    aucs = []
    for _ in range(500):
        idx = rng.integers(0, mask.sum(), mask.sum())
        yt = y[mask][idx]
        if len(np.unique(yt))<2: continue
        aucs.append(roc_auc_score(yt, all_oof[mask][idx]))
    aucs = np.array(aucs)

    log.info("="*60)
    log.info(f"M5 临床基线: per-fold {np.mean(fold_aucs):.3f}±{np.std(fold_aucs):.3f}")
    log.info(f"OOF AUC={oof_auc:.3f} (AP={oof_ap:.3f})")
    log.info(f"Bootstrap AUC={aucs.mean():.3f} [{np.percentile(aucs,2.5):.3f}, {np.percentile(aucs,97.5):.3f}]")
    log.info("="*60)

    out = {"task":"clinical","name":"M5_clinical","n_classes":2,
           "n_samples": int(mask.sum()), "fold_aucs":[float(a) for a in fold_aucs],
           "fold_mean": float(np.mean(fold_aucs)),"fold_std": float(np.std(fold_aucs)),
           "oof_auc": float(oof_auc),"oof_ap": float(oof_ap),
           "bootstrap_auc": float(aucs.mean()),
           "bootstrap_ci": [float(np.percentile(aucs,2.5)), float(np.percentile(aucs,97.5))],
           "config": {"n_folds": 5, "seed": 42, "model": "LogisticRegression",
                      "class_weight": "balanced", "C": 0.5},
           "feature_missingness_csv": "results/M5_clinical_feature_missingness.csv",
           "per_fold_checkpoint_dir": "models/per_fold/M5_clinical"}
    with open(RESULTS/"metrics_M5_clinical.json","w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(RESULTS/"fold_registry_M5_clinical.csv","w",newline="") as f:
        import csv
        fieldnames = ["task", "repeat", "fold", "split", "patient_id", "site",
                      "label", "seed", "fold_auc", "checkpoint"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(fold_registry)

    # OOF 预测 (供 agent)
    with open(RESULTS/"oof_preds_M5_clinical.csv","w",newline="") as f:
        import csv
        w=csv.writer(f); w.writerow(["patient_id","label","prob_c0","prob_c1"])
        for i,pid in enumerate(df.index):
            if mask[i]:
                w.writerow([pid, int(y[i]), round(1-all_oof[i],4), round(all_oof[i],4)])
    log.info("保存: metrics_M5_clinical.json, oof_preds_M5_clinical.csv, fold_registry_M5_clinical.csv")


if __name__ == "__main__":
    main()
