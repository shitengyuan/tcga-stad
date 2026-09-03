#!/usr/bin/env python3
"""Run a strict no-label-leakage Agent over model-probability inputs.

The Agent keeps the project novelty as a clinically constrained model-panel
orchestrator: it summarizes agreement/conflict across M1-M5, fixes the final
prediction to M1, assigns an uncertainty tier, and recommends confirmatory
MMR/EBER testing. It does not read true MSI/EBV/subtype/label/POLE fields
because those columns are absent from the prepared input table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.deepseek_client import DeepSeekClient, parse_json_object


ROOT = Path(__file__).resolve().parent
CPU = ROOT / "results" / "cpu_supplement"
DEFAULT_ENV = Path("/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/appkey.env")

SYSTEM_PROMPT = """你是胃肠病理AI助手，但不是最终诊断系统。
你的创新点是“模型面板编排 + 不确定性分层 + 分子检测前初筛建议”：
1. 只能基于M1-M5模型概率总结一致性/冲突；
2. final_pred必须等于M1固定阈值结果，不得修改M1；
3. 当前输入没有真实WSI图像或病理医生确认的patch证据，因此不得声称看到了TIL、肿瘤、间质、坏死、Crohn样反应等形态学实体；
4. 不得读取或推断真实MSI、EBV、TCGA四亚型、label、POLE；
5. 临床定位只能写“分子检测前初筛”，不能替代MSI/EBV确认或治疗决策；
6. 输出严格JSON，不要Markdown，不要额外解释。"""


def decide(row: pd.Series) -> dict[str, Any]:
    m1 = float(row["M1_immune_sensitive_prob_c1"])
    m2 = float(row["M2_msi_prob_c1"])
    m3 = float(row["M3_ebv_prob_c1"])
    m4 = [float(row[f"M4_subtype4_prob_c{i}"]) for i in range(4)]
    m5 = float(row["M5_clinical_prob_c1"])

    final_pred = "sensitive" if m1 >= 0.5 else "nonsensitive"
    m1_pos = m1 >= 0.5
    trusted = ["M1"]
    conflicts: list[str] = []
    supports: list[str] = []

    if (m2 >= 0.5) == m1_pos:
        trusted.append("M2")
        supports.append("M2")
    else:
        conflicts.append("M2")
    if (m3 >= 0.5) == m1_pos:
        trusted.append("M3")
        supports.append("M3")
    else:
        conflicts.append("M3")

    m4_pred = int(np.argmax(m4))
    m4_sensitive = m4_pred in (0, 1)
    if m4_sensitive == m1_pos:
        trusted.append("M4")
        supports.append("M4")
    else:
        conflicts.append("M4")

    if abs(m5 - 0.5) >= 0.25 and ((m5 >= 0.5) == m1_pos):
        trusted.append("M5")

    if 0.4 <= m1 <= 0.6 or conflicts:
        confidence = "low"
    elif len(supports) >= 2 and (m1 <= 0.1 or m1 >= 0.9):
        confidence = "high"
    else:
        confidence = "medium"

    if confidence == "low":
        ihc = "both"
    elif final_pred == "sensitive":
        ihc = "both"
    else:
        ihc = "none"

    report = (
        f"M1固定判断为{final_pred}，概率{m1:.3f}。"
        f"M2={m2:.3f}，M3={m3:.3f}，M4最高类索引={m4_pred}，M5={m5:.3f}。"
        f"可信模型：{','.join(trusted)}；冲突：{','.join(conflicts) if conflicts else '无'}。"
        f"定位为MSI/EBV分子检测前初筛，建议验证：{ihc}。"
    )
    return {
        "patient_id": row["patient_id"],
        "final_pred": final_pred,
        "confidence": confidence,
        "trusted_models": trusted,
        "conflicts": conflicts,
        "ihc_suggest": ihc,
        "screening_positioning": "仅用于MSI/EBV分子检测前初筛，不能替代分子检测、病理诊断或治疗决策。",
        "safety_flags": ["no_label_leakage", "m1_fixed", "no_unverified_morphology"],
        "report": report,
        "backend": "deterministic_rules_no_llm",
        "used_fields": [
            "patient_id",
            "site",
            "M1_immune_sensitive_prob_c*",
            "M2_msi_prob_c*",
            "M3_ebv_prob_c*",
            "M4_subtype4_prob_c*",
            "M5_clinical_prob_c*",
        ],
        "forbidden_fields_not_loaded": ["label", "MSI", "EBV", "subtype", "M4_subtype", "M4_label", "POLE"],
    }


def llm_prompt(row: pd.Series, rules: dict[str, Any]) -> str:
    allowed = {
        "patient_id": row["patient_id"],
        "site": row.get("site", ""),
        "M1_immune_sensitive_prob_c1": float(row["M1_immune_sensitive_prob_c1"]),
        "M2_msi_prob_c1": float(row["M2_msi_prob_c1"]),
        "M3_ebv_prob_c1": float(row["M3_ebv_prob_c1"]),
        "M4_subtype4_probabilities": {
            "prob_c0_EBV": float(row["M4_subtype4_prob_c0"]),
            "prob_c1_MSI": float(row["M4_subtype4_prob_c1"]),
            "prob_c2_GS": float(row["M4_subtype4_prob_c2"]),
            "prob_c3_CIN": float(row["M4_subtype4_prob_c3"]),
        },
        "M5_clinical_prob_c1": float(row["M5_clinical_prob_c1"]),
        "final_pred_fixed_from_M1": row["final_pred_fixed_from_M1"],
        "rule_result": {k: rules[k] for k in ["final_pred", "confidence", "trusted_models", "conflicts", "ihc_suggest"]},
    }
    schema = {
        "patient_id": "string",
        "final_pred": "sensitive|nonsensitive, exactly equal to final_pred_fixed_from_M1",
        "confidence": "high|medium|low",
        "trusted_models": "array of model names supporting the fixed M1 decision, selected from M1-M5",
        "conflicts": ["model names"],
        "ihc_suggest": "MMR|EBER-ISH|both|none",
        "screening_positioning": "one sentence, molecular-test pre-screen only",
        "report": "Chinese, <=150 Chinese characters, no unsupported morphology claims",
        "safety_flags": ["no_label_leakage", "m1_fixed", "no_unverified_morphology"],
    }
    return (
        "请基于以下允许输入生成Agent结构化输出。\n"
        "禁止使用任何真实标签或真实分子分型；禁止修改M1 final_pred。\n"
        f"允许输入JSON:\n{json.dumps(allowed, ensure_ascii=False)}\n"
        f"输出schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )


def deepseek_decide(row: pd.Series, rules: dict[str, Any], client: DeepSeekClient, model: str, temperature: float) -> dict[str, Any]:
    try:
        raw = client.chat(
            llm_prompt(row, rules),
            system=SYSTEM_PROMPT,
            model=model,
            temperature=temperature,
            max_tokens=1200,
            json_output=True,
        )
        parsed = parse_json_object(raw)
    except Exception as exc:
        parsed = dict(rules)
        parsed["backend"] = "deepseek_no_leakage_fallback_rules"
        parsed["deepseek_model"] = model
        parsed["temperature"] = temperature
        parsed["llm_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        parsed["guardrails"] = {
            "final_pred_fixed_to_M1": True,
            "forbidden_fields_not_loaded": rules["forbidden_fields_not_loaded"],
            "no_unverified_morphology": True,
            "fallback_to_rules_on_llm_error": True,
        }
        return parsed
    fixed = rules["final_pred"]
    if parsed.get("final_pred") != fixed:
        parsed["final_pred_before_guardrail"] = parsed.get("final_pred")
        parsed["final_pred"] = fixed
    for key in ["patient_id", "confidence", "trusted_models", "conflicts", "ihc_suggest", "screening_positioning", "safety_flags", "report"]:
        parsed.setdefault(key, rules.get(key))
    parsed["backend"] = "deepseek_no_leakage"
    parsed["deepseek_model"] = model
    parsed["temperature"] = temperature
    parsed["raw_response"] = raw
    parsed["guardrails"] = {
        "final_pred_fixed_to_M1": True,
        "forbidden_fields_not_loaded": rules["forbidden_fields_not_loaded"],
        "no_unverified_morphology": True,
    }
    return parsed


def check_rule_stability(df: pd.DataFrame, repeats: int) -> bool:
    repeat_runs = [[decide(row) for _, row in df.iterrows()] for _ in range(repeats)]
    stable = True
    for i in range(1, len(repeat_runs)):
        a = [
            (x["patient_id"], x["final_pred"], x["confidence"], tuple(x["trusted_models"]), tuple(x["conflicts"]), x["ihc_suggest"])
            for x in repeat_runs[0]
        ]
        b = [
            (x["patient_id"], x["final_pred"], x["confidence"], tuple(x["trusted_models"]), tuple(x["conflicts"]), x["ihc_suggest"])
            for x in repeat_runs[i]
        ]
        if a != b:
            stable = False
            break
    return stable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=CPU / "agent_no_leakage_inputs_289.csv")
    parser.add_argument("--out_dir", type=Path, default=CPU / "agent_no_leakage_outputs")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backend", choices=["rules", "deepseek"], default="rules")
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_cases", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.inputs)
    if args.max_cases is not None:
        df = df.head(args.max_cases).copy()

    rule_outputs = [decide(row) for _, row in df.iterrows()]
    if args.backend == "deepseek":
        client = DeepSeekClient(env_file=args.env_file)
        outputs = [
            deepseek_decide(row, rules, client, args.model, args.temperature)
            for (_, row), rules in zip(df.iterrows(), rule_outputs)
        ]
    else:
        outputs = rule_outputs

    safe_model = args.model.replace("/", "_").replace(":", "_")
    suffix = f"deepseek_{safe_model}" if args.backend == "deepseek" else "rules"
    jsonl = args.out_dir / f"agent_no_leakage_outputs_{suffix}.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for obj in outputs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    output_csv = args.out_dir / f"agent_no_leakage_outputs_{suffix}.csv"
    pd.DataFrame(outputs).to_csv(output_csv, index=False)

    summary = {
        "backend": "deepseek_no_leakage" if args.backend == "deepseek" else "deterministic_rules_no_llm",
        "model": args.model if args.backend == "deepseek" else None,
        "temperature": args.temperature if args.backend == "deepseek" else None,
        "input_csv": str(args.inputs.relative_to(ROOT)),
        "n_patients": int(len(df)),
        "rule_repeats": int(args.repeats),
        "rule_stable_across_repeats": check_rule_stability(df, args.repeats),
        "n_llm_fallback_to_rules": int(sum(o.get("backend") == "deepseek_no_leakage_fallback_rules" for o in outputs)),
        "output_jsonl": str(jsonl.relative_to(ROOT)),
        "output_csv": str(output_csv.relative_to(ROOT)),
        "important_note": (
            "DeepSeek report generation used no-label-leakage inputs; final_pred is guardrailed to M1."
            if args.backend == "deepseek"
            else "This is a formal no-leakage structured Agent output, but it is rule-based and not a DeepSeek/GLM LLM run."
        ),
    }
    (args.out_dir / f"agent_no_leakage_stability_{suffix}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
