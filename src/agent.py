"""
agent.py
════════
胃癌免疫敏感亚型预测的可解释 Agent (路C: 只解释, 不改预测)。

2026-09-01 口径:
  正式 Agent 不读取真实 MSI/EBV/TCGA subtype/label/POLE, 不修改 M1 结果,
  不在缺少真实渲染 patch 和病理医生验证时声称 TIL/肿瘤/间质/坏死等形态学结论.
  主要创新点保留为: 模型面板编排 + 不确定性分层 + 分子检测前初筛建议.

职责:
  1. 固定 final_pred = ABMIL/M1 pred
  2. 汇总模型概率和可选临床非标签变量
  3. 输出 MMR/EBER-ISH 分子检测前初筛建议

输入:
  - abmil_prob: ABMIL 免疫敏感概率
  - overlay_png: patch 聚类 overlay 图(默认不启用视觉形态解读)
  - clinical (可选): age/sex/Lauren/stage 等非真实分型字段

输出:
  - final_pred: 恒等于 ABMIL pred (agent 不改)
  - visual_evidence: 默认标记为未启用
  - report: 初筛报告 + IHC/ISH 验证建议

用法:
  from src.agent import ExplainableAgent
  ag = ExplainableAgent(app_id="...")
  r = ag.explain(abmil_prob=0.49, overlay_png="vis/TCGA-XX.png")
"""
from __future__ import annotations
import json
import logging
import re
from typing import Dict, Optional

from src.friday_client import FridayClient

logger = logging.getLogger(__name__)

VISION_PROMPT = """你是胃肠病理AI助手。当前输入若只是坐标overlay或cluster示意图，
不得把颜色簇命名为TIL、肿瘤、间质、坏死或Crohn样反应，除非另有真实渲染patch和病理医生验证。
只允许描述图像/overlay是否可读、颜色簇分布是否集中或分散，并说明不能据此作形态学诊断。
只输出JSON:
{"visual_status":"usable/limited/unusable","description":"一句话说明","unsupported_morphology_claims":false}"""

REPORT_SYSTEM = """你是胃肠病理AI助手。只能基于模型概率做分子检测前初筛说明。
final_pred必须等于ABMIL/M1阈值结果；不得输出真实MSI/EBV/TCGA亚型；不得声称未经验证的形态学实体。
报告150字内，说明模型倾向、不确定性和MMR/EBER-ISH验证建议。"""


class ExplainableAgent:
    """可解释病理 agent (路C: 只解释, 不改预测)。"""

    def __init__(self, app_id: Optional[str] = None,
                 vision_model: str = "glm-4v-plus",
                 reason_model: str = "glm-5.2",
                 enable_visual: bool = False):
        self.fc = FridayClient(app_id)
        self.vision_model = vision_model
        self.reason_model = reason_model
        self.enable_visual = enable_visual

    def explain(self, abmil_prob: float, overlay_png: Optional[str] = None,
                clinical: Optional[Dict] = None) -> Dict:
        """对单个病例生成可解释报告 (不改预测)。

        对所有病例都可调用 (不限难例), 因为只解释不干预, 成本可控即可。
        """
        abmil_pred = "sensitive" if abmil_prob >= 0.5 else "nonsensitive"

        clinical = self._sanitize_clinical(clinical)
        vision = self._extract_visual_evidence(overlay_png)

        # 报告生成
        report = self._generate_report(abmil_prob, abmil_pred, vision, clinical)

        return {
            "abmil_prob": round(abmil_prob, 4),
            "abmil_pred": abmil_pred,
            "final_pred": abmil_pred,   # 恒等于 ABMIL, agent 不改
            "visual_evidence": vision,
            "report": report,
            "guardrails": {
                "final_pred_fixed_to_abmil": True,
                "no_label_leakage": True,
                "no_unverified_morphology": True,
            },
        }

    # ── 内部 ─────────────────────────────────────────────────

    def _extract_visual_evidence(self, overlay_png: Optional[str]) -> Dict:
        """Optional legacy visual call. Disabled by default for formal no-leakage runs."""
        if not self.enable_visual or not overlay_png:
            return {
                "visual_status": "disabled",
                "description": "正式Agent未启用视觉形态解读；attention/cluster需病理验证后才能命名。",
                "unsupported_morphology_claims": False,
            }
        try:
            raw = self.fc.vision(overlay_png, VISION_PROMPT,
                                 model=self.vision_model,
                                 max_tokens=512, temperature=0.1)
            parsed = self._parse_json(raw)
            parsed["raw"] = raw
            return parsed
        except Exception as e:
            logger.warning(f"视觉证据提取失败: {e}")
            return {"error": str(e), "description": "(视觉提取失败)", "unsupported_morphology_claims": False}

    @staticmethod
    def _sanitize_clinical(clinical: Optional[Dict]) -> Optional[Dict]:
        if not clinical:
            return None
        forbidden = {"label", "MSI", "EBV", "subtype", "M4_subtype", "M4_label", "POLE", "TCGA_subtype"}
        return {k: v for k, v in clinical.items() if k not in forbidden}

    @staticmethod
    def _parse_json(text: str) -> Dict:
        """从模型输出里提取 JSON (鲁棒: 代码块/reasoning/嵌套)。"""
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        start = cleaned.find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start:i + 1])
                        except json.JSONDecodeError:
                            break
        return {"raw": text, "description": text[:120]}

    def _generate_report(self, abmil_prob, abmil_pred, vision, clinical) -> str:
        """glm-5.2 综合出报告 (解释模型预测, 不推翻)。"""
        clin_str = json.dumps(clinical, ensure_ascii=False) if clinical else "无"
        prompt = (
            f"ABMIL模型预测: 免疫敏感概率 {abmil_prob:.3f} "
            f"(倾向{'免疫敏感' if abmil_pred=='sensitive' else '免疫不敏感'})\n"
            f"视觉证据状态: {json.dumps(vision, ensure_ascii=False)}\n"
            f"临床信息: {clin_str}\n"
            f"写报告: 解释模型概率和不确定性, 不要引用未经验证的形态学实体, "
            f"并建议 IHC(MMR蛋白MLH1/PMS2/MSH2/MSH6)或 EBER-ISH 验证项。"
        )
        try:
            return self.fc.chat(prompt, model=self.reason_model,
                                system=REPORT_SYSTEM,
                                max_tokens=2048, temperature=0.3)
        except Exception as e:
            return f"(报告生成失败: {e})"
