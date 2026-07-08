#!/usr/bin/env python3
"""
run_plots2.py — 补充图: 混淆矩阵 / per-site AUC / agent报告卡片
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIG = RESULTS / "figures"


# ── 图7: 混淆矩阵 (M1, Youden阈值) ───────────────────────
def plot_confusion():
    te = json.load(open(RESULTS / "threshold_eval.json"))
    op = te["M1_immune_sensitive"]["operating_points"]["youden"]
    m1 = pd.read_csv(RESULTS / "oof_preds_M1_immune_sensitive.csv")
    y = m1["label"].values
    pred = (m1["prob_c1"].values >= op["threshold"]).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])  # 行=真实[敏感,非], 列=预测

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["预测敏感", "预测不敏感"])
    ax.set_yticklabels(["真实敏感", "真实不敏感"])
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(j, i, f"{labels[i][j]}\n{v}", ha="center", va="center",
                    fontsize=14, color="white" if v > cm.max()/2 else "black")
    plt.colorbar(im)
    plt.title(f"M1 混淆矩阵 (Youden阈值={op['threshold']})\n"
              f"敏感度={op['sensitivity']:.3f}  特异度={op['specificity']:.3f}")
    plt.tight_layout()
    plt.savefig(FIG / "fig7_confusion.png", dpi=150)
    plt.close()
    print("✓ fig7_confusion.png")


# ── 图8: per-site AUC (M1) ───────────────────────────────
def plot_per_site():
    m1 = pd.read_csv(RESULTS / "oof_preds_M1_immune_sensitive.csv")
    m1["site"] = m1["patient_id"].str.split("-").str[1]
    sites = m1.groupby("site").agg(n=("label", "size"), pos=("label", "sum"))
    sites = sites[sites["pos"] > 0].sort_values("n", ascending=True)

    aucs = []
    for site in sites.index:
        sub = m1[m1["site"] == site]
        if len(np.unique(sub["label"])) < 2:
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(sub["label"], sub["prob_c1"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c" if a < 0.7 else "#2ecc71" for a in aucs]
    bars = ax.barh(range(len(sites)), aucs, color=colors, alpha=0.85)
    ax.set_yticks(range(len(sites)))
    ax.set_yticklabels([f"{s} (n={sites.loc[s,'n']}, 敏感={int(sites.loc[s,'pos'])})" for s in sites.index])
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5, label="随机(0.5)")
    ax.axvline(0.884, color="blue", ls=":", alpha=0.6, label="整体OOF AUC(0.884)")
    for i, a in enumerate(aucs):
        if not np.isnan(a):
            ax.text(a + 0.01, i, f"{a:.3f}", va="center", fontsize=9)
    ax.set_xlabel("AUC")
    ax.set_title("M1 各站点 AUC (site-stratified, 回应站点偏倚)")
    ax.set_xlim(0, 1.1)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fig8_per_site.png", dpi=150)
    plt.close()
    print("✓ fig8_per_site.png")


# ── 图9: agent 报告卡片 (3个代表case) ───────────────────
def plot_agent_cards():
    j = json.load(open(RESULTS / "agent_panel_judgments.json"))
    # 挑3个: 敏感高置信, 不敏感高置信, 边界
    cases = []
    for x in j:
        cases.append(x)
    if len(cases) < 3:
        print(f"仅 {len(cases)} 例, 跳过卡片图")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ["Case 1: 敏感(EBV)", "Case 2: 不敏感", "Case 3: 边界"]
    for ax, x, title in zip(axes, cases[:3], titles):
        ax.axis("off")
        s = x["agent_selection"]
        panel = x.get("model_panel", {})
        panel_str = "\n".join([f"  {k}: {v:.3f}" for k, v in panel.items()])
        text = (
            f"{title}\n{'='*40}\n"
            f"患者: {x['patient_id']}\n"
            f"M1预测: {x['final_pred']} (prob={x['m1_prob']:.3f})\n"
            f"置信度: {s.get('confidence')}\n\n"
            f"模型面板:\n{panel_str}\n\n"
            f"信任: {s.get('trusted_models')}\n"
            f"冲突: {s.get('conflicts')}\n"
            f"IHC建议: {s.get('ihc_suggest')}\n\n"
            f"报告:\n{x.get('report','')[:300]}"
        )
        ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
                verticalalignment="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#f9f9f9", alpha=0.9))
    plt.suptitle("Agent 面板: 3 个代表性病例报告", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(FIG / "fig9_agent_cards.png", dpi=150)
    plt.close()
    print("✓ fig9_agent_cards.png")


if __name__ == "__main__":
    plot_confusion()
    plot_per_site()
    plot_agent_cards()
    print(f"\n补充图已保存到 {FIG}")
