#!/usr/bin/env python3
"""
run_plots.py
════════════
生成关键指标图:
  1. ROC 曲线 (M1/M2/M3/M5 多模型对比)
  2. 模型面板 AUC 柱状图 + CI
  3. 临床操作点图 (敏感度 vs 特异度, 不同阈值)
  4. 消融图 (图像 vs 临床)
  5. 概率分布图 (正例 vs 负例)
"""
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)

# 颜色
C = {"M1":"#e74c3c","M2":"#3498db","M3":"#2ecc71","M5":"#95a5a6","M4":"#9b59b6"}

MODELS = [
    ("M1_immune_sensitive", "oof_preds_M1_immune_sensitive.csv", "M1 免疫敏感", "M1"),
    ("M2_msi", "oof_preds_M2_msi.csv", "M2 MSI", "M2"),
    ("M3_ebv", "oof_preds_M3_ebv.csv", "M3 EBV", "M3"),
    ("M5_clinical", "oof_preds_M5_clinical.csv", "M5 临床", "M5"),
]


def load(name, fname):
    df = pd.read_csv(RESULTS / fname)
    return df["label"].values, df["prob_c1"].values


# ── 图1: ROC 曲线 ─────────────────────────────────────────
def plot_roc():
    plt.figure(figsize=(7, 6))
    for name, fname, label, key in MODELS:
        y, p = load(name, fname)
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        plt.plot(fpr, tpr, color=C[key], lw=2, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("1 - Specificity (假阳率)")
    plt.ylabel("Sensitivity (真阳率)")
    plt.title("ROC 曲线: 免疫敏感亚型预测 (TCGA-STAD, site-stratified OOF)")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig1_roc.png", dpi=150)
    plt.close()
    print("✓ fig1_roc.png")


# ── 图2: AUC 柱状图 + CI ─────────────────────────────────
def plot_auc_bar():
    names, aucs, lo, hi = [], [], [], []
    for name, fname, label, key in MODELS:
        d = json.load(open(RESULTS / f"metrics_{name}.json"))
        names.append(label)
        aucs.append(d["oof_auc"])
        ci = d.get("bootstrap_ci", [d["oof_auc"], d["oof_auc"]])
        lo.append(aucs[-1] - ci[0])
        hi.append(ci[1] - aucs[-1])
    colors = [C[m[3]] for m in MODELS]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, aucs, yerr=[lo, hi], capsize=6, color=colors, alpha=0.85)
    for b, a in zip(bars, aucs):
        plt.text(b.get_x() + b.get_width()/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=11)
    plt.axhline(0.81, color="gray", ls="--", alpha=0.6, label="Kather 2019 (0.81)")
    plt.axhline(0.835, color="gray", ls=":", alpha=0.6, label="Zhang 2025 (0.835)")
    plt.ylabel("OOF AUC")
    plt.title("模型面板 AUC 对比 (95% bootstrap CI)")
    plt.ylim(0.5, 1.0)
    plt.legend(fontsize=9)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig2_auc_bar.png", dpi=150)
    plt.close()
    print("✓ fig2_auc_bar.png")


# ── 图3: 临床操作点 (敏感度 vs 特异度) ───────────────────
def plot_operating_points():
    te = json.load(open(RESULTS / "threshold_eval.json"))
    plt.figure(figsize=(8, 6))
    for name, _, label, key in MODELS:
        if name not in te:
            continue
        ops = te[name]["operating_points"]
        # 4个操作点: youden, high_sens, high_spec, default
        pts = [("high_sensitivity(>=0.9)", "初筛"), ("youden", "Youden"),
               ("default_0.5", "默认0.5"), ("high_specificity(>=0.9)", "确诊")]
        xs = [ops[p[0]]["specificity"] for p in pts]
        ys = [ops[p[0]]["sensitivity"] for p in pts]
        plt.scatter(xs, ys, color=C[key], s=120, zorder=3, label=label)
        for (p, lab), x, y in zip(pts, xs, ys):
            plt.annotate(lab, (x, y), fontsize=7, xytext=(5, 5), textcoords="offset points")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("Specificity (特异度)")
    plt.ylabel("Sensitivity (敏感度)")
    plt.title("临床操作点 (不同阈值下的敏感度/特异度)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig3_operating_points.png", dpi=150)
    plt.close()
    print("✓ fig3_operating_points.png")


# ── 图4: 消融图 (图像 vs 临床) ───────────────────────────
def plot_ablation():
    fig, ax = plt.subplots(figsize=(7, 5))
    cats = ["M5 临床\n基线", "M1 图像\n(UNI2-h+ABMIL)"]
    vals = [0.620, 0.897]
    bars = ax.bar(cats, vals, color=[C["M5"], C["M1"]], alpha=0.85, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.annotate("", xy=(1, 0.897), xytext=(0, 0.620),
                arrowprops=dict(arrowstyle="->", color="green", lw=2))
    ax.text(0.5, 0.78, "+0.277\n(图像特征增量)", ha="center", color="green", fontsize=11, fontweight="bold")
    ax.set_ylabel("OOF AUC")
    ax.set_title("消融: 图像特征 vs 临床特征")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig4_ablation.png", dpi=150)
    plt.close()
    print("✓ fig4_ablation.png")


# ── 图5: 概率分布 (M1 正例 vs 负例) ──────────────────────
def plot_prob_dist():
    y, p = load("M1_immune_sensitive", "oof_preds_M1_immune_sensitive.csv")
    plt.figure(figsize=(8, 5))
    plt.hist(p[y == 0], bins=30, alpha=0.6, color=C["M5"], label=f"非敏感 (n={(y==0).sum()})")
    plt.hist(p[y == 1], bins=30, alpha=0.6, color=C["M1"], label=f"免疫敏感 (n={(y==1).sum()})")
    plt.axvline(0.5, color="red", ls="--", label="默认阈值 0.5")
    plt.xlabel("M1 预测概率 (免疫敏感)")
    plt.ylabel("样本数")
    plt.title("M1 OOF 预测概率分布")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "fig5_prob_dist.png", dpi=150)
    plt.close()
    print("✓ fig5_prob_dist.png")


# ── 图6: 模型面板热图 (6模型 AUC 一览) ──────────────────
def plot_panel_heatmap():
    rows = []
    for name, _, label, _ in MODELS:
        d = json.load(open(RESULTS / f"metrics_{name}.json"))
        rows.append({"模型": label, "AUC": d["oof_auc"]})
    # M4, M6
    d4 = json.load(open(RESULTS / "metrics_M4_subtype4.json"))
    rows.append({"模型": "M4 四亚型", "AUC": d4["oof_auc"]})
    d6 = json.load(open(RESULTS / "metrics_M6_survival.json"))
    rows.append({"模型": "M6 生存(C-index)", "AUC": d6["oof_cindex"]})
    df = pd.DataFrame(rows).set_index("模型")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(df.values, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks([0]); ax.set_xticklabels(["AUC/C-index"])
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index)
    for i, v in enumerate(df.values):
        ax.text(0, i, f"{v[0]:.3f}", ha="center", va="center", fontsize=12, fontweight="bold")
    plt.colorbar(im, label="值")
    plt.title("模型面板性能一览")
    plt.tight_layout()
    plt.savefig(FIG / "fig6_panel_heatmap.png", dpi=150)
    plt.close()
    print("✓ fig6_panel_heatmap.png")


if __name__ == "__main__":
    plot_roc()
    plot_auc_bar()
    plot_operating_points()
    plot_ablation()
    plot_prob_dist()
    plot_panel_heatmap()
    print(f"\n全部图已保存到 {FIG}")
