#!/usr/bin/env python3
"""Run a strict no-label-leakage Agent over prepared model-probability inputs.

Default backend is deterministic rules. It does not call any LLM and cannot
read true MSI/EBV/subtype/label/POLE fields because those columns are absent
from the prepared input table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CPU = ROOT / "results" / "cpu_supplement"


def decide(row: pd.Series) -> dict[str, Any]:
    m1 = float(row["M1_immune_sensitive_prob_c1"])
    m2 = float(row["M2_msi_prob_c1"])
    m3 = float(row["M3_ebv_prob_c1"])
    m4 = [float(row[f"M4_subtype4_prob_c{i}"]) for i in range(4)]
    m5 = float(row["M5_clinical_prob_c1"])

    final_pred = "sensitive" if m1 >= 0.5 else "nonsensitive"
    m1_pos = m1 >= 0.5
    trusted = ["M1"]
    conflicts = []
    supports = []

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
        f"建议IHC：{ihc}。"
    )
    return {
        "patient_id": row["patient_id"],
        "final_pred": final_pred,
        "confidence": confidence,
        "trusted_models": trusted,
        "conflicts": conflicts,
        "ihc_suggest": ihc,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=CPU / "agent_no_leakage_inputs_289.csv")
    parser.add_argument("--out_dir", type=Path, default=CPU / "agent_no_leakage_outputs")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.inputs)
    outputs = [decide(row) for _, row in df.iterrows()]

    jsonl = args.out_dir / "agent_no_leakage_outputs_rules.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for obj in outputs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    pd.DataFrame(outputs).to_csv(args.out_dir / "agent_no_leakage_outputs_rules.csv", index=False)

    repeat_runs = []
    for rep in range(args.repeats):
        repeat_runs.append([decide(row) for _, row in df.iterrows()])
    stable = True
    for i in range(1, len(repeat_runs)):
        a = [(x["patient_id"], x["final_pred"], x["confidence"], tuple(x["trusted_models"]), tuple(x["conflicts"]), x["ihc_suggest"]) for x in repeat_runs[0]]
        b = [(x["patient_id"], x["final_pred"], x["confidence"], tuple(x["trusted_models"]), tuple(x["conflicts"]), x["ihc_suggest"]) for x in repeat_runs[i]]
        if a != b:
            stable = False
            break

    summary = {
        "backend": "deterministic_rules_no_llm",
        "input_csv": str(args.inputs.relative_to(ROOT)),
        "n_patients": int(len(df)),
        "repeats": int(args.repeats),
        "stable_across_repeats": stable,
        "output_jsonl": str(jsonl.relative_to(ROOT)),
        "output_csv": str((args.out_dir / "agent_no_leakage_outputs_rules.csv").relative_to(ROOT)),
        "important_note": "This is a formal no-leakage structured Agent output, but it is rule-based and not a DeepSeek/GLM LLM run.",
    }
    (args.out_dir / "agent_no_leakage_stability_rules.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
