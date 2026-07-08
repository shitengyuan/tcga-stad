"""
agent.py
════════
胃癌免疫敏感亚型预测的可解释 Agent (路C: 只解释, 不改预测)。

定位 (研究 v2):
  SOTA 由 UNI2-h + ABMIL 扛 (AUC 0.884).
  agent 不干预预测 (final_pred 恒等于 ABMIL pred, 零指标风险),
  只提供可被 IHC/分子验证的形态学解释, 增强临床信任.

职责:
  1. 视觉证据提取: glm-4v-plus 看 overlay, 客观描述 TIL 形态(不判断亚型)
  2. 报告生成: glm-5.2 综合 ABMIL概率 + 视觉证据 + 临床, 出病理报告 + IHC验证建议

输入:
  - abmil_prob: ABMIL 免疫敏感概率
  - overlay_png: patch 聚类 overlay 图
  - clinical (可选): age/sex/Lauren/subtype

输出:
  - final_pred: 恒等于 ABMIL pred (agent 不改)
  - visual_evidence: TIL 形态客观描述
  - report: 病理风格报告 + IHC 验证锚点

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

# 视觉模型只做客观形态描述, 不判断亚型 (避免倾向性偏置)
VISION_PROMPT = """你是胃肠病理专家. 这是胃癌 H&E 切片的 patch 形态聚类 overlay 图
(不同颜色 = 不同形态原型).

任务: 客观描述图中与免疫浸润相关的形态学特征, 不要判断亚型.
逐项评估 (true/false/unclear) 并给一句话依据:
- til_density: 淋巴细胞(特定颜色簇)的整体密度 是否高
- til_clustering: 淋巴细胞是否成簇聚集(而非弥散)
- interface_infiltration: 淋巴细胞是否浸润肿瘤-基质界面
- crohn_like: 是否有 Crohn 样淋巴细胞反应

只输出JSON, 不要其他文字:
{"til_density":"high/low/unclear","til_clustering":true/false/"unclear","interface_infiltration":true/false/"unclear","crohn_like":true/false/"unclear","description":"一句话客观描述"}"""

REPORT_SYSTEM = """你是胃肠病理 AI 助手. 基于ABMIL模型预测和视觉形态证据, 写病理风格的免疫敏感亚型预测报告.
要求: (1)150字内 (2)引用具体形态证据 (3)说明ABMIL预测倾向 (4)给出可被IHC/分子验证的建议.
注意: 你是在解释模型为何这样预测, 不要推翻模型预测."""


class ExplainableAgent:
    """可解释病理 agent (路C: 只解释, 不改预测)。"""

    def __init__(self, app_id: Optional[str] = None,
                 vision_model: str = "glm-4v-plus",
                 reason_model: str = "glm-5.2"):
        self.fc = FridayClient(app_id)
        self.vision_model = vision_model
        self.reason_model = reason_model

    def explain(self, abmil_prob: float, overlay_png: str,
                clinical: Optional[Dict] = None) -> Dict:
        """对单个病例生成可解释报告 (不改预测)。

        对所有病例都可调用 (不限难例), 因为只解释不干预, 成本可控即可。
        """
        abmil_pred = "sensitive" if abmil_prob >= 0.5 else "nonsensitive"

        # 视觉证据提取 (客观描述, 不判断亚型)
        vision = self._extract_visual_evidence(overlay_png)

        # 报告生成
        report = self._generate_report(abmil_prob, abmil_pred, vision, clinical)

        return {
            "abmil_prob": round(abmil_prob, 4),
            "abmil_pred": abmil_pred,
            "final_pred": abmil_pred,   # 恒等于 ABMIL, agent 不改
            "visual_evidence": vision,
            "report": report,
        }

    # ── 内部 ─────────────────────────────────────────────────

    def _extract_visual_evidence(self, overlay_png: str) -> Dict:
        """glm-4v-plus 客观描述形态 (不判断亚型)。"""
        try:
            raw = self.fc.vision(overlay_png, VISION_PROMPT,
                                 model=self.vision_model,
                                 max_tokens=512, temperature=0.1)
            parsed = self._parse_json(raw)
            parsed["raw"] = raw
            return parsed
        except Exception as e:
            logger.warning(f"视觉证据提取失败: {e}")
            return {"error": str(e), "description": "(视觉提取失败)"}

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
            f"视觉形态证据: {json.dumps(vision, ensure_ascii=False)}\n"
            f"临床信息: {clin_str}\n"
            f"写报告: 解释模型为何这样预测, 引用形态证据, "
            f"并建议 IHC(MMR蛋白MLH1/PMS2/MSH2/MSH6)或 EBER-ISH 验证项."
        )
        try:
            return self.fc.chat(prompt, model=self.reason_model,
                                system=REPORT_SYSTEM,
                                max_tokens=2048, temperature=0.3)
        except Exception as e:
            return f"(报告生成失败: {e})"
