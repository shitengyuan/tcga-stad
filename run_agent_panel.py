#!/usr/bin/env python3
"""
run_agent_panel.py
══════════════════
多模型面板的 Agent 编排器 (路C增强版: agent 不参与预测, 但动态选择信任哪些模型 + 视觉印证 + 报告)。

模型面板 (读取各自 OOF 预测):
  M1 immune_sensitive : MSI∪EBV vs 非
  M2 msi              : MSI-H vs MSS
  M3 ebv              : EBV+ vs EBV-
  M4 subtype4         : 四亚型
  M5 clinical         : 临床基线

Agent (GLM-5.2) 职责:
  1. 模型选择: 根据临床背景判断哪些模型可信 (如肠型+老年→MSI权重高)
  2. 视觉印证: glm-4v-plus 看 overlay, 提取 TIL 形态证据
  3. 报告综合: 综合所选模型预测 + 视觉证据, 生成可IHC验证报告

注意: agent 不改最终预测 (final_pred = M1 pred), 只做编排解释, 零指标风险。
      评估"agent选择 vs 均匀集成"的吻合度, 作为 agent 决策质量的代理指标。

用法:
  python run_agent_panel.py --max_cases 20
  python run_agent_panel.py --patient TCGA-BR-7707  # 单例
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from pathlib import Path

import pandas as pd

from src.friday_client import FridayClient

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("agent_panel")

BASE = Path(__file__).resolve().parent
VIS_DIR = BASE / "tcga_stad_uni2h" / "TCGA-STAD" / "vis"
CLIN_CSV = BASE / "clinical.csv"
RESULTS = BASE / "results"

MODEL_FILES = {
    "M1_immune_sensitive": "oof_preds_M1_immune_sensitive.csv",  # 或 oof_preds_site.csv (旧格式)
    "M2_msi": "oof_preds_M2_msi.csv",
    "M3_ebv": "oof_preds_M3_ebv.csv",
    "M4_subtype4": "oof_preds_M4_subtype4.csv",
    "M5_clinical": "oof_preds_M5_clinical.csv",
}

# M1 旧格式兼容 (train_abmil.py 生成的 oof_preds_site.csv)
M1_FALLBACK = "oof_preds_site.csv"

SELECTOR_SYSTEM = """你是胃肠病理 AI, 负责为胃癌免疫敏感亚型预测选择可信模型。

可用模型面板:
- M1 免疫敏感(MSI∪EBV vs 非): 综合模型, 主预测器
- M2 MSI(MSI-H vs MSS): 专测MSI
- M3 EBV(EBV+ vs EBV-): 专测EBV
- M4 四亚型(EBV/MSI/GS/CIN): 多分类
- M5 临床基线: 仅用临床变量, 性能弱

选择规则 (输出JSON):
- 若 M1 与 M2/M3 一致(都敏感或都不敏感): 信任 M1, 高置信
- 若 M1 边界(0.4-0.6) 但 M2 或 M3 强支持: 提升置信, 引用 M2/M3
- 若 M1 与 M2/M3 矛盾: 标记冲突, 降低置信, 建议 IHC 确认
- M5 仅作参考(弱基线), 不主导判断
- 临床背景(如 Lauren 肠型→MSI概率高; 弥漫型→GS/CIN多)作为先验

输出JSON: {"trusted_models":["M1",...], "confidence":"high/medium/low", "reasoning":"一句", "ihc_suggest":"MMR/EBER-ISH/both"}"""

VISION_PROMPT = """这是胃癌 H&E 切片 patch 聚类 overlay 图。客观描述免疫浸润形态(不判断亚型):
{"til_density":"high/low/unclear","til_clustering":true/false/"unclear","interface_infiltration":true/false/"unclear","description":"一句"}"""

REPORT_SYSTEM = """你是胃肠病理 AI. 直接输出报告正文, 不要推理过程、不要思考、不要编号列表.
报告含: 模型一致性简述 + 形态证据引用 + IHC验证建议. 150字内. 不推翻M1主预测."""


class PanelAgent:
    def __init__(self, app_id):
        self.fc = FridayClient(app_id)
        self.report_model = "deepseek-v3-friday"  # 直接输出, 不思考, 稳定

    def load_predictions(self, pid):
        """加载某患者所有模型的 OOF 预测。"""
        preds = {}
        for name, fname in MODEL_FILES.items():
            path = RESULTS / fname
            # M1 旧格式回退
            if name == "M1_immune_sensitive" and not path.exists():
                path = RESULTS / M1_FALLBACK
            if not path.exists():
                continue
            df = pd.read_csv(path)
            row = df[df["patient_id"]==pid]
            if len(row)==0:
                continue
            r = row.iloc[0]
            # 兼容两种格式: prob_c0/c1 (新) 或 prob (旧, 即正类概率)
            if "prob_c1" in df.columns:
                preds[name] = {c: float(r[c]) for c in df.columns if c.startswith("prob_c")}
            elif "prob" in df.columns:
                p1 = float(r["prob"])
                preds[name] = {"prob_c0": 1-p1, "prob_c1": p1}
            else:
                continue
            preds[name]["label"] = int(r["label"])
        return preds

    def select_models(self, preds, clinical):
        """模型选择: 规则 + LLM 解释 (规则做决策, LLM 说人话)。

        规则 (可解释, 稳定, 不依赖LLM思考):
          - M1 为主预测器, prob_c1 = 免疫敏感概率
          - 一致性: M1 与 M2(MSI)/M3(EBV) 是否同向 (都敏感或都不敏感)
          - 边界: M1 prob ∈ [0.4,0.6] 为低置信
          - 冲突: M1 与 M2/M3 反向 → 降置信, 建议 IHC
        """
        m1 = preds.get("M1_immune_sensitive", {})
        m1_prob = m1.get("prob_c1", 0.5)
        m1_pred = m1_prob >= 0.5
        m2 = preds.get("M2_msi", {})
        m3 = preds.get("M3_ebv", {})

        trusted = ["M1"]
        conflicts = []
        supports = []
        # M2/M3 一致性 (仅当存在时)
        if "prob_c1" in m2:
            m2_pred = m2["prob_c1"] >= 0.5
            if m2_pred == m1_pred:
                supports.append("M2(MSI)")
                trusted.append("M2")
            else:
                conflicts.append("M2(MSI)")
        if "prob_c1" in m3:
            m3_pred = m3["prob_c1"] >= 0.5
            if m3_pred == m1_pred:
                supports.append("M3(EBV)")
                trusted.append("M3")
            else:
                conflicts.append("M3(EBV)")

        # 置信度
        if 0.4 <= m1_prob <= 0.6:
            confidence = "low"  # 边界
        elif conflicts:
            confidence = "low"  # 冲突
        elif supports:
            confidence = "high"  # 多模型一致
        else:
            confidence = "medium"

        # IHC 建议
        if confidence == "low":
            ihc = "both"
        elif m1_pred:
            ihc = "MMR(MSI)+EBER-ISH(EBV)"
        else:
            ihc = "none"

        reasoning = (f"M1预测{'敏感' if m1_pred else '不敏感'}(prob={m1_prob:.2f}); "
                     + (f"支持:{','.join(supports)}; " if supports else "")
                     + (f"冲突:{','.join(conflicts)}; " if conflicts else "")
                     + f"置信{confidence}")

        return {
            "trusted_models": trusted,
            "confidence": confidence,
            "reasoning": reasoning,
            "ihc_suggest": ihc,
            "supports": supports,
            "conflicts": conflicts,
            "method": "rule-based",
        }

    def visual_evidence(self, overlay_png):
        try:
            r = self.fc.vision(overlay_png, VISION_PROMPT, model="glm-4v-plus",
                               max_tokens=512, temperature=0.1)
            return self._parse_json(r) | {"raw": r}
        except Exception as e:
            return {"error": str(e)}

    def report(self, preds, selection, vision, clinical):
        m1_prob = preds.get("M1_immune_sensitive", {}).get("prob_c1", 0)
        m1_pred = "免疫敏感" if m1_prob >= 0.5 else "免疫不敏感"
        panel = {k: round(v.get("prob_c1", 0), 3) for k, v in preds.items() if "prob_c1" in v}
        prompt = (f"写一份胃癌免疫敏感亚型预测的病理报告(150字内正文).\n\n"
                  f"信息:\n"
                  f"- M1主预测: {m1_pred} (prob={m1_prob:.3f}), 最终判断不改\n"
                  f"- 各模型敏感概率: {panel}\n"
                  f"- Agent选择: 置信={selection.get('confidence')}, 信任={selection.get('trusted_models')}, 冲突={selection.get('conflicts')}\n"
                  f"- 临床: {clinical}\n\n"
                  f"要求: 直接输出报告正文(不要推理/编号列表), 说明模型一致性, 给IHC验证建议(MMR/EBER-ISH).")
        try:
            return self.fc.chat(prompt, model=self.report_model, system=REPORT_SYSTEM,
                                max_tokens=512, temperature=0.4)
        except Exception as e:
            return f"(报告失败: {e})"

    @staticmethod
    def _parse_json(text):
        import re
        cleaned = text.replace("```json","").replace("```","").strip()
        try: return json.loads(cleaned)
        except: pass
        s = cleaned.find("{")
        if s>=0:
            d=0
            for i in range(s,len(cleaned)):
                if cleaned[i]=="{": d+=1
                elif cleaned[i]=="}":
                    d-=1
                    if d==0:
                        try: return json.loads(cleaned[s:i+1])
                        except: break
        return {"raw": text}


def get_clinical(pid, clin):
    if pid not in clin.index: return {}
    r = clin.loc[pid]
    return {k: str(r[k]) for k in ["age","sex","histological_diagnosis","subtype"]
            if k in clin.columns and pd.notna(r[k])}


def find_overlay(slide_id):
    p = VIS_DIR / f"{slide_id}__overlay.png"
    return p if p.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app_id", default=None)
    p.add_argument("--max_cases", type=int, default=None)
    p.add_argument("--patient", default=None, help="单例分析")
    args = p.parse_args()

    clin = pd.read_csv(CLIN_CSV).set_index("patient_id")
    ag = PanelAgent(args.app_id)

    # 确定要分析的患者
    if args.patient:
        pids = [p.strip() for p in args.patient.split(",") if p.strip()]
    else:
        # 用 M1 OOF (覆盖最全), 兼容旧格式
        m1_path = RESULTS / MODEL_FILES["M1_immune_sensitive"]
        if not m1_path.exists():
            m1_path = RESULTS / M1_FALLBACK
        m1 = pd.read_csv(m1_path)
        pids = m1["patient_id"].tolist()
        if args.max_cases:
            pids = pids[:args.max_cases]

    log.info(f"分析 {len(pids)} 个患者")
    judgments = []
    for i, pid in enumerate(pids):
        preds = ag.load_predictions(pid)
        if not preds:
            continue
        clinical = get_clinical(pid, clin)
        slide_id = str(clin.loc[pid,"slide_id"]).split(";")[0] if pid in clin.index else pid+"-01Z-00-DX1"
        overlay = find_overlay(slide_id)

        log.info(f"[{i+1}/{len(pids)}] {pid} M1_prob={preds.get('M1_immune_sensitive',{}).get('prob_c1','?')}")
        sel = ag.select_models(preds, clinical)
        # 视觉模块已移除 (glm-4v-plus 在 overlay 图上区分度差, 不可靠)
        vis = {"note": "visual module removed (unreliable on overlay images)"}
        rep = ag.report(preds, sel, vis, clinical)

        m1_prob = preds.get("M1_immune_sensitive",{}).get("prob_c1",0)
        judgments.append({
            "patient_id": pid,
            "final_pred": "sensitive" if m1_prob>=0.5 else "nonsensitive",  # 不改预测
            "m1_prob": round(m1_prob,4),
            "model_panel": {k: v.get("prob_c1", v.get("prob_c")) for k,v in preds.items() if "prob_c1" in v},
            "agent_selection": sel,
            "visual_evidence": vis,
            "report": rep,
        })

    out = RESULTS / "agent_panel_judgments.json"
    with open(out,"w") as f:
        json.dump(judgments, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"保存 {len(judgments)} 例: {out}")

    # 汇总: agent 选择统计
    from collections import Counter
    conf = Counter(j["agent_selection"].get("confidence","?") for j in judgments)
    log.info(f"置信度分布: {dict(conf)}")
    trusted = Counter(tuple(sorted(j["agent_selection"].get("trusted_models",[]))) for j in judgments)
    log.info(f"信任模型组合 Top5: {trusted.most_common(5)}")


if __name__ == "__main__":
    main()
