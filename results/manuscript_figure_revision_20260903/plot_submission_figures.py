"""Create vector figures for the revised TCGA-STAD manuscript.

All statistical panels are derived from the frozen patient-level tables.
No pathology morphology is inferred or labelled by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

TCGA = "#0072B2"
CPTAC = "#D55E00"
CLINICAL = "#666666"
FUSION = "#8E5EA2"
GREEN = "#009E73"
PANEL = "#1F2937"

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def panel(ax, label: str) -> None:
    ax.text(-0.15, 1.08, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", color=PANEL)


def box(ax, xy, w, h, text, *, face="#F8FAFC", edge="#475569", fontsize=8):
    patch = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                           linewidth=1.1, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color="#111827", wrap=True)


def arrow(ax, start, end, color="#475569"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", lw=1.2, color=color, shrinkA=4, shrinkB=4))


def fig1(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.55))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.96, "Study design and intended clinical use", ha="center", va="top",
            fontsize=13, fontweight="bold", color=PANEL)
    x = [0.035, 0.275, 0.515, 0.755]
    labels = ["Diagnostic\nH&E whole-slide image", "UNI2-h features\n+ ABMIL aggregation",
              "MSI/EBV testing-\npriority score", "Prioritize MMR/MSI and EBER\nconfirmatory testing\nMolecular results inform\ndownstream evaluation"]
    colors = ["#EAF2FB", "#EAF2FB", "#FFF4E5", "#EAF7F0"]
    widths = [0.19, 0.19, 0.19, 0.21]
    for i, (xx, label, width, color) in enumerate(zip(x, labels, widths, colors)):
        box(ax, (xx, 0.51), width, 0.22, label, face=color,
            edge=TCGA if i < 3 else GREEN, fontsize=6.5 if i == 3 else 8)
        if i:
            arrow(ax, (x[i - 1] + widths[i - 1], 0.62), (xx, 0.62),
                  TCGA if i < 3 else GREEN)
    ax.text(0.5, 0.40, "Testing-priority score is not an absolute clinical probability and does not predict immunotherapy benefit.",
            ha="center", va="center", fontsize=8, color="#7C2D12", fontweight="bold")
    box(ax, (0.12, 0.12), 0.31, 0.17,
        "Development evidence\nTCGA-STAD: 246 eligible patients\n231 centre-isolated OOF; 62 M1-positive",
        face="#EEF6FC", edge=TCGA, fontsize=7.2)
    box(ax, (0.57, 0.12), 0.31, 0.17,
        "External validation evidence\nCPTAC: 156 QC- and label-matched patients\n43 M1-positive; median 4 slides/patient",
        face="#FFF4E8", edge=CPTAC, fontsize=7.2)
    save(fig, out / "fig1_study_design.pdf")


def fig2(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.97, "Patient-level cohort flow", ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(0.27, 0.90, "TCGA-STAD development cohort", ha="center", fontsize=10, fontweight="bold", color=TCGA)
    ax.text(0.73, 0.90, "CPTAC external validation cohort", ha="center", fontsize=10, fontweight="bold", color=CPTAC)
    left = [(0.08, 0.72, "Eligible after feature QC\nand reference-label matching\nn = 246; M1-positive = 68"),
            (0.08, 0.49, "Sites with <8 patients\ntrain only\nn = 15; excluded from reported OOF"),
            (0.08, 0.25, "Centre-isolated internal evaluation\n5 folds × 2 repeats\nn = 231; M1-positive = 62")]
    right = [(0.58, 0.72, "H&E slides assessed for QC\nand reference-label matching"),
             (0.58, 0.49, "Patient-level aggregation\nmean of slide probabilities\nmedian 4 slides (range 1–8)"),
             (0.58, 0.25, "Independent external evaluation\nn = 156; M1-positive = 43\nfixed threshold = 0.5")]
    for seq, color in ((left, TCGA), (right, CPTAC)):
        for i, (xx, yy, text) in enumerate(seq):
            box(ax, (xx, yy), 0.34, 0.14, text, face="#F8FAFC", edge=color, fontsize=8)
            if i < len(seq) - 1:
                arrow(ax, (xx + 0.17, yy), (xx + 0.17, seq[i + 1][1] + 0.14), color)
    ax.text(0.5, 0.06,
            "Original pre-QC screening counts were not retained in the frozen release; all reported metrics use the stated patient-level analysis sets.",
            ha="center", va="center", fontsize=7.2, color="#4B5563")
    save(fig, out / "fig2_cohort_flow.pdf")


def m1_xy(frame: pd.DataFrame, cohort: str):
    if cohort == "TCGA":
        return frame["M1_label"].astype(int).to_numpy(), frame["M1_prob_c1"].astype(float).to_numpy()
    return frame["immune_sensitive"].astype(int).to_numpy(), frame["immune_sensitive_prob"].astype(float).to_numpy()


def ci_text(point, ci):
    return f"{point:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"


def fig3(tcga: pd.DataFrame, cptac: pd.DataFrame, metrics_t: dict, metrics_c: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.65))
    datasets = [("TCGA OOF", tcga, TCGA, metrics_t), ("CPTAC external", cptac, CPTAC, metrics_c)]
    for name, df, color, metric in datasets:
        y, p = m1_xy(df, "TCGA" if "TCGA" in name else "CPTAC")
        point = metric["tasks"]["M1_immune_sensitive"]["point"]
        ci = metric["tasks"]["M1_immune_sensitive"]["bootstrap_95ci"]
        fpr, tpr, _ = roc_curve(y, p)
        pre, rec, _ = precision_recall_curve(y, p)
        auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
        pred = p >= 0.5
        tp = ((pred == 1) & (y == 1)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
        axes[0].plot(fpr, tpr, color=color, lw=2.1, label=f"{name}: AUROC {ci_text(auc, ci['auroc'])}")
        axes[0].scatter([fp/(fp+tn)], [tp/(tp+fn)], color=color, s=34, zorder=4, edgecolor="white", linewidth=0.6)
        axes[1].plot(rec, pre, color=color, lw=2.1, label=f"{name}: AP {ci_text(ap, ci['average_precision'])}")
        axes[1].scatter([tp/(tp+fn)], [tp/(tp+fp)], color=color, s=34, zorder=4, edgecolor="white", linewidth=0.6)
    axes[0].plot([0, 1], [0, 1], ls="--", color="#9CA3AF", lw=1)
    for ax, title, xlabel, ylabel in [(axes[0], "ROC", "False-positive rate", "True-positive rate"),
                                       (axes[1], "Precision–recall", "Recall", "Precision")]:
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(-0.02, 1.02), ylim=(-0.02, 1.05))
        ax.grid(alpha=0.2); ax.legend(loc="lower right", frameon=True)
    axes[0].text(0.03, 0.98, "Filled circles: fixed threshold 0.5\nTCGA: n=231, 62 positive\nCPTAC: n=156, 43 positive",
                 transform=axes[0].transAxes, va="top", fontsize=7.2,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB"))
    panel(axes[0], "A"); panel(axes[1], "B")
    fig.suptitle("M1 performance in internal OOF and external validation", y=1.02, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig3_m1_roc_pr.pdf")


def calibration_panel(fig, outer, y, p, title, color, point, ci, letter):
    gs = outer.subgridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0]); hist = fig.add_subplot(gs[1], sharex=ax)
    bins = pd.qcut(p, q=5, duplicates="drop")
    frame = pd.DataFrame({"p": p, "y": y, "bin": bins})
    rows = []
    rng = np.random.default_rng(20260903)
    for _, grp in frame.groupby("bin", observed=True):
        obs = grp.y.mean(); boots = [grp.y.iloc[rng.integers(0, len(grp), len(grp))].mean() for _ in range(1000)]
        rows.append((grp.p.mean(), obs, np.quantile(boots, .025), np.quantile(boots, .975), len(grp)))
    rows = np.array(rows)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#9CA3AF")
    ax.errorbar(rows[:, 0], rows[:, 1], yerr=[rows[:, 1]-rows[:, 2], rows[:, 3]-rows[:, 1]],
                fmt="o", color=color, capsize=2.5, lw=1.1, ms=5)
    for x, yy, _, _, n in rows:
        ax.annotate(f"n={int(n)}", (x, yy), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=6.5)
    ax.set(title=title, ylabel="Observed fraction positive", xlim=(-.02, 1.02), ylim=(-.02, 1.05))
    ax.grid(alpha=.18)
    ax.tick_params(labelbottom=False)
    text = (f"Brier {point['brier']:.3f} ({ci['brier'][0]:.3f}–{ci['brier'][1]:.3f})\n"
            f"Intercept {point['calibration_intercept']:.3f}\nSlope {point['calibration_slope']:.3f}")
    ax.text(.03, .97, text, transform=ax.transAxes, va="top", fontsize=6.7,
            bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#D1D5DB"))
    hist.hist(p, bins=np.linspace(0, 1, 16), color=color, alpha=.75, edgecolor="white")
    hist.set(xlabel="Predicted M1 testing-priority score", ylabel="Patients", xlim=(-.02, 1.02))
    hist.grid(axis="y", alpha=.15)
    panel(ax, letter)


def capacity(y, p, capacity):
    k = int(np.ceil(capacity * len(y)))
    top = y[np.argsort(-p)[:k]]
    return top.sum() / y.sum(), int(top.sum()), k


def capacity_ci(y, p, capacities, seed=20260903):
    rng = np.random.default_rng(seed); values = np.empty((600, len(capacities)))
    n = len(y)
    for b in range(len(values)):
        ind = rng.integers(0, n, n)
        for j, c in enumerate(capacities):
            values[b, j] = capacity(y[ind], p[ind], c)[0]
    return np.quantile(values, [.025, .975], axis=0)


def capacity_panel(ax, y, p, title, color, clinical=None):
    caps = np.array([.05, .10, .15, .20, .25, .30, .40, .50, .75, 1.0])
    cov = np.array([capacity(y, p, c)[0] for c in caps])
    ci = capacity_ci(y, p, caps)
    ax.plot(caps * 100, cov * 100, color=color, lw=2.2, marker="o", ms=3.8, label="Image M1")
    ax.fill_between(caps * 100, ci[0] * 100, ci[1] * 100, color=color, alpha=.15, linewidth=0)
    if clinical is not None:
        ccov = np.array([capacity(y, clinical, c)[0] for c in caps])
        ax.plot(caps * 100, ccov * 100, color=CLINICAL, lw=1.7, ls="--", marker="s", ms=3.2, label="Clinical-only")
    ax.plot([0, 100], [0, 100], color="#9CA3AF", lw=1.2, ls=":", label="Random priority")
    for c in (.20, .30, .40):
        coverage, detected, k = capacity(y, p, c)
        ax.scatter(c * 100, coverage * 100, color=color, s=26, zorder=4)
        ax.annotate(f"{detected}/{int(y.sum())}", (c*100, coverage*100), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=6.5)
    ax.set(title=title, xlabel="Immediate-testing capacity (%)", ylabel="Positive coverage (%)",
           xlim=(0, 100), ylim=(0, 105))
    ax.grid(alpha=.18); ax.legend(loc="lower right", frameon=True)


def fig4(tcga: pd.DataFrame, cptac: pd.DataFrame, clinical: pd.DataFrame, metrics_t: dict, metrics_c: dict, out: Path) -> None:
    fig = plt.figure(figsize=(8.25, 7.0))
    outer = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0], hspace=.42, wspace=.28)
    y_t, p_t = m1_xy(tcga, "TCGA"); y_c, p_c = m1_xy(cptac, "CPTAC")
    task_t = metrics_t["tasks"]["M1_immune_sensitive"]; task_c = metrics_c["tasks"]["M1_immune_sensitive"]
    calibration_panel(fig, outer[0, 0], y_t, p_t, "TCGA OOF calibration", TCGA, task_t["point"], task_t["bootstrap_95ci"], "A")
    calibration_panel(fig, outer[0, 1], y_c, p_c, "CPTAC external calibration", CPTAC, task_c["point"], task_c["bootstrap_95ci"], "B")
    ax_t = fig.add_subplot(outer[1, 0]); ax_c = fig.add_subplot(outer[1, 1])
    clinical = clinical.set_index("patient_id").reindex(tcga.patient_id)
    capacity_panel(ax_t, y_t, p_t, "TCGA: M1 vs clinical-only", TCGA, clinical["clinical_only_prob"].to_numpy(float))
    capacity_panel(ax_c, y_c, p_c, "CPTAC: M1 external triage", CPTAC)
    panel(ax_t, "C"); panel(ax_c, "D")
    fig.suptitle("M1 calibration and testing-capacity analysis", y=.995, fontsize=11, fontweight="bold")
    save(fig, out / "fig4_m1_calibration_capacity.pdf")


def fig5(metrics: dict, out: Path) -> None:
    models = ["Image-only M1", "Clinical-only", "Late-fusion\n(exploratory)"]
    keys = ["Image_only_M1_matched_231", "Clinical_only_matched_231", "Image_plus_clinical_late_fusion_matched_231"]
    colors = [TCGA, CLINICAL, FUSION]
    fig, axes = plt.subplots(1, 3, figsize=(8.25, 3.2), gridspec_kw={"width_ratios": [1.05, 1.05, .9]})
    for ax, metric_name, title in zip(axes[:2], ["auroc", "average_precision"], ["AUROC", "Average precision"]):
        for i, (model, key, color) in enumerate(zip(models, keys, colors)):
            point = metrics["models"][key]["point"][metric_name]
            ci = metrics["models"][key]["bootstrap_95ci"][metric_name]
            ax.errorbar(point, 2-i, xerr=[[point-ci[0]], [ci[1]-point]], fmt="o", color=color, ms=6, capsize=3, lw=1.5)
            ax.text(point, 2-i+.17, f"{point:.3f} ({ci[0]:.3f}–{ci[1]:.3f})", ha="center", fontsize=6.5)
        ax.set(title=title, yticks=[0, 1, 2], yticklabels=list(reversed(models)) if metric_name == "auroc" else [], xlim=(0, 1), xlabel="Patient-level OOF estimate")
        ax.grid(axis="x", alpha=.2)
    deltas = [("Clinical-only − M1", -.2860, (-.3647, -.2057), CLINICAL),
              ("Late-fusion − M1", -.0697, (-.1137, -.0255), FUSION)]
    for i, (label, point, ci, color) in enumerate(deltas):
        axes[2].errorbar(point, 1-i, xerr=[[point-ci[0]], [ci[1]-point]], fmt="o", color=color, ms=6, capsize=3, lw=1.5)
        axes[2].text(point, 1-i+.17, f"{point:.3f} ({ci[0]:.3f}–{ci[1]:.3f})", ha="center", fontsize=6.4)
    axes[2].axvline(0, color="#9CA3AF", lw=1, ls="--")
    axes[2].set(title="Paired AUROC difference", yticks=[0, 1], yticklabels=[deltas[1][0], deltas[0][0]], xlim=(-.45, .1), xlabel="Difference relative to M1")
    axes[2].grid(axis="x", alpha=.2)
    for ax, letter in zip(axes, "ABC"):
        panel(ax, letter)
    fig.suptitle("Matched 231-patient comparison of image and clinical models", y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig5_clinical_increment.pdf")


def fig6_template(out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.1))
    labels = ["True positive", "False positive", "False negative", "True negative"]
    for ax, label, letter in zip(axes.flat, labels, "ABCD"):
        ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#F8FAFC")
        for spine in ax.spines.values(): spine.set_color("#94A3B8"); spine.set_linestyle("--")
        ax.text(.5, .60, f"{label} case", ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(.5, .40, "Whole-slide thumbnail + attention overlay\nHigh-magnification H&E + scale bar\nBlinded pathology-review fields", ha="center", va="center", fontsize=8, color="#475569")
        panel(ax, letter)
    fig.suptitle("Pathology-review figure framework — not for reporting before blinded review", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, out / "fig6_pathology_review_framework.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rel = args.release
    tcga = pd.read_csv(rel / "locked_predictions_tcga.csv")
    cptac = pd.read_csv(rel / "locked_predictions_cptac.csv")
    clinical = pd.read_csv(rel / "matched_clinical_oof_predictions_231.csv")
    metrics_t = json.loads((rel / "tcga_231_metrics_with_ci.json").read_text())
    metrics_c = json.loads((rel / "cptac_156_metrics_with_ci.json").read_text())
    metrics_clin = json.loads((rel / "matched_clinical_metrics_231.json").read_text())
    fig1(args.out); fig2(args.out); fig3(tcga, cptac, metrics_t, metrics_c, args.out)
    fig4(tcga, cptac, clinical, metrics_t, metrics_c, args.out); fig5(metrics_clin, args.out); fig6_template(args.out)


if __name__ == "__main__":
    main()
