#!/usr/bin/env python3
"""
run_agent.py
════════════
读 ABMIL 的 OOF 预测 → 筛难例 → agent 二意见 + 报告 → 评估修正效果。

输入:
  - results/oof_preds_site.csv  (ABMIL OOF 预测)
  - tcga_stad_uni2h/TCGA-STAD/vis/*.png  (overlay 图)
  - clinical.csv (临床信息)

输出:
  - results/agent_judgments.json  (每个难例的 agent 判断+报告)
  - results/agent_eval.json       (修正效果: 难例上 ABMIL vs ABMIL+agent)

用法:
  python run_agent.py                              # 全部难例
  python run_agent.py --max_cases 10               # 只跑前10个难例(省钱/测)
  python run_agent.py --abmil_csv results/oof_preds_site.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

from src.agent import ExplainableAgent

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("agent")

BASE = Path(__file__).resolve().parent
VIS_DIR = BASE / "tcga_stad_uni2h" / "TCGA-STAD" / "vis"
CLIN_CSV = BASE / "clinical.csv"
RESULTS = BASE / "results"


def find_overlay(slide_id: str) -> Path | None:
    """根据 slide_id 找 overlay PNG (文件名: {slide_id}__overlay.png)。"""
    p = VIS_DIR / f"{slide_id}__overlay.png"
    return p if p.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--abmil_csv", default=str(RESULTS / "oof_preds_site.csv"))
    p.add_argument("--app_id", default=None, help="Friday AppID (或设 FRIDAY_APP_ID)")
    p.add_argument("--max_cases", type=int, default=None, help="最多跑N个难例(测试用)")
    p.add_argument("--hard_low", type=float, default=0.35)
    p.add_argument("--hard_high", type=float, default=0.65)
    args = p.parse_args()

    # 读 ABMIL OOF 预测
    oof = pd.read_csv(args.abmil_csv)
    log.info(f"读 ABMIL OOF: {len(oof)} 例, 正例率 {oof['label'].mean():.3f}")

    # 临床信息
    clin = pd.read_csv(CLIN_CSV).set_index("patient_id")

    ag = ExplainableAgent(args.app_id, hard_low=args.hard_low, hard_high=args.hard_high)

    # 筛难例
    hard = oof[(oof["prob"] >= args.hard_low) & (oof["prob"] <= args.hard_high)].copy()
    log.info(f"难例 (prob∈[{args.hard_low},{args.hard_high}]): {len(hard)} / {len(oof)} "
             f"({len(hard)/len(oof):.1%})")
    if args.max_cases:
        hard = hard.head(args.max_cases)
        log.info(f"限制跑前 {len(hard)} 个难例")

    judgments = []
    for i, row in enumerate(hard.itertuples()):
        pid = row.patient_id
        # 找 overlay (需 slide_id; oof_preds 里有, 否则从 clinical 取第一个)
        slide_id = getattr(row, "slide_id", None) or str(clin.loc[pid, "slide_id"]).split(";")[0]
        overlay = find_overlay(slide_id)
        if overlay is None:
            log.warning(f"[{i+1}/{len(hard)}] {pid}: 无 overlay, 跳过")
            continue

        clin_info = {k: str(clin.loc[pid, k]) for k in
                     ["age", "sex", "histological_diagnosis", "subtype"]
                     if k in clin.columns and pd.notna(clin.loc[pid, k])}

        log.info(f"[{i+1}/{len(hard)}] {pid} abmil_prob={row.prob:.3f} label={row.label}")
        j = ag.judge(abmil_prob=row.prob, overlay_png=str(overlay), clinical=clin_info)
        j["patient_id"] = pid
        j["true_label"] = row.label
        j["slide_id"] = slide_id
        judgments.append(j)

    # 保存逐例判断
    out_j = RESULTS / "agent_judgments.json"
    with open(out_j, "w") as f:
        json.dump(judgments, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"逐例判断已保存: {out_j}")

    # ── 评估: 难例上 ABMIL vs ABMIL+agent ──────────────────
    eval_res = _evaluate(judgments)
    out_e = RESULTS / "agent_eval.json"
    with open(out_e, "w") as f:
        json.dump(eval_res, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"评估结果已保存: {out_e}")

    log.info("=" * 60)
    log.info(f"难例数: {eval_res['n_hard']}")
    log.info(f"ABMIL 难例准确率: {eval_res['abmil_acc']:.3f}")
    log.info(f"ABMIL+Agent 难例准确率: {eval_res['fused_acc']:.3f}")
    log.info(f"agent 同意 ABMIL: {eval_res['agree_rate']:.1%}")
    log.info(f"修正: +{eval_res['corrected_to_right']} 对 →错, {eval_res['corrected_to_wrong']} 错→对"
             if eval_res.get('corrected_to_right') is not None else "")
    log.info("=" * 60)


def _evaluate(judgments: list) -> dict:
    """评估 agent 对难例的修正效果。"""
    if not judgments:
        return {"n_hard": 0}
    n = len(judgments)
    abmil_correct = 0
    fused_correct = 0
    agree = 0
    to_right = 0  # ABMIL错, agent改对
    to_wrong = 0  # ABMIL对, agent改错
    for j in judgments:
        true = "sensitive" if j["true_label"] == 1 else "nonsensitive"
        abmil_pred = j["abmil_pred"]
        fused_pred = j.get("final_pred", abmil_pred)
        abmil_ok = abmil_pred == true
        fused_ok = fused_pred == true
        abmil_correct += abmil_ok
        fused_correct += fused_ok
        if abmil_pred == fused_pred:
            agree += 1
        elif not abmil_ok and fused_ok:
            to_right += 1
        elif abmil_ok and not fused_ok:
            to_wrong += 1
    return {
        "n_hard": n,
        "abmil_acc": round(abmil_correct / n, 3),
        "fused_acc": round(fused_correct / n, 3),
        "agree_rate": round(agree / n, 3),
        "corrected_to_right": to_right,
        "corrected_to_wrong": to_wrong,
        "net_gain": to_right - to_wrong,
    }


if __name__ == "__main__":
    main()
