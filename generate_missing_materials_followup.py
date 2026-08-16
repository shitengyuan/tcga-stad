#!/usr/bin/env python3
"""
Generate follow-up materials for items that were previously marked incomplete.

This script only uses existing CSV/JSON outputs. It does not load WSI/h5 files,
does not train models, and does not call any LLM service.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CPU = RESULTS / "cpu_supplement"
AUDIT = RESULTS / "audit_first_stage"
REPORTS = ROOT / "reports"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def binary_metrics(y_true: List[int], scores: List[float], threshold: float) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    for y, s in zip(y_true, scores):
        pred = 1 if s >= threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 1 and pred == 0:
            fn += 1
        elif y == 0 and pred == 1:
            fp += 1
        else:
            tn += 1
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    bacc = (sens + spec) / 2
    return {
        "threshold": threshold,
        "n": len(y_true),
        "n_pos": sum(y_true),
        "n_neg": len(y_true) - sum(y_true),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
        "accuracy": acc,
        "balanced_accuracy": bacc,
    }


def choose_screening_threshold(y_true: List[int], scores: List[float], target_sensitivity: float) -> Dict[str, Any]:
    candidates = sorted(set(scores + [min(scores) - 1e-12, max(scores) + 1e-12]))
    metrics = [binary_metrics(y_true, scores, t) for t in candidates]
    feasible = [m for m in metrics if m["sensitivity"] >= target_sensitivity]
    if not feasible:
        chosen = max(metrics, key=lambda m: (m["sensitivity"], m["specificity"], m["threshold"]))
        chosen["achieved_target"] = False
    else:
        # For screening, keep the highest specificity while satisfying target sensitivity.
        # Ties use the highest threshold to avoid unnecessary positives.
        chosen = max(feasible, key=lambda m: (m["specificity"], m["threshold"]))
        chosen["achieved_target"] = True
    chosen["target_sensitivity"] = target_sensitivity
    return chosen


def generate_high_sensitivity_thresholds() -> Dict[str, Any]:
    tasks = [
        ("M1_immune_sensitive", RESULTS / "oof_preds_M1_immune_sensitive.csv", "immune_sensitive"),
        ("M2_msi", RESULTS / "oof_preds_M2_msi.csv", "msi"),
        ("M3_ebv", RESULTS / "oof_preds_M3_ebv.csv", "ebv"),
        ("M5_clinical", RESULTS / "oof_preds_M5_clinical.csv", "clinical_baseline"),
    ]
    targets = [0.90, 0.95, 0.99]
    rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "existing patient-level OOF CSVs",
        "policy": "highest-specificity threshold satisfying target sensitivity; ties use highest threshold",
        "targets": targets,
        "tasks": {},
    }
    for task_name, path, endpoint in tasks:
        data = read_csv(path)
        y_true = [int(float(r["label"])) for r in data]
        scores = [to_float(r["prob_c1"]) for r in data]
        task_results = []
        for target in targets:
            m = choose_screening_threshold(y_true, scores, target)
            out = {
                "task": task_name,
                "endpoint": endpoint,
                **m,
            }
            rows.append(out)
            task_results.append(out)
        payload["tasks"][task_name] = task_results
    fieldnames = [
        "task",
        "endpoint",
        "target_sensitivity",
        "achieved_target",
        "threshold",
        "n",
        "n_pos",
        "n_neg",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "accuracy",
        "balanced_accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    csv_path = CPU / "high_sensitivity_thresholds_oof.csv"
    json_path = CPU / "high_sensitivity_thresholds_oof.json"
    write_csv(csv_path, rows, fieldnames)
    write_json(json_path, payload)
    return {"csv": str(csv_path.relative_to(ROOT)), "json": str(json_path.relative_to(ROOT)), "rows": len(rows)}


def load_agent_panel() -> Dict[str, Any]:
    path = RESULTS / "agent_panel_judgments.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {r.get("patient_id"): r for r in data}


def generate_showcase_existing_evidence() -> Dict[str, Any]:
    case_ids = ["TCGA-VQ-AA69", "TCGA-VQ-AA6K", "TCGA-BR-7703"]
    cohort_rows = {r["patient_id"]: r for r in read_csv(AUDIT / "final_289_patient_cohort.csv")}
    agent_rows = load_agent_panel()
    rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []
    for pid in case_ids:
        c = cohort_rows.get(pid, {})
        a = agent_rows.get(pid, {})
        out: Dict[str, Any] = {
            "patient_id": pid,
            "in_289_oof_cohort": bool(c),
            "has_agent_panel_output": bool(a),
            "subtype": c.get("subtype", ""),
            "M1_label": c.get("M1_label", ""),
            "MSI": c.get("MSI", ""),
            "EBV": c.get("EBV", ""),
            "M4_subtype": c.get("M4_subtype", ""),
            "fold": c.get("fold", ""),
            "M1_immune_sensitive_prob_c1": c.get("M1_immune_sensitive_prob_c1", ""),
            "M2_msi_prob_c1": c.get("M2_msi_prob_c1", ""),
            "M3_ebv_prob_c1": c.get("M3_ebv_prob_c1", ""),
            "M4_subtype4_prob_c0_EBV": c.get("M4_subtype4_prob_c0", ""),
            "M4_subtype4_prob_c1_MSI": c.get("M4_subtype4_prob_c1", ""),
            "M4_subtype4_prob_c2_GS": c.get("M4_subtype4_prob_c2", ""),
            "M4_subtype4_prob_c3_CIN": c.get("M4_subtype4_prob_c3", ""),
            "M5_clinical_prob_c1": c.get("M5_clinical_prob_c1", ""),
            "agent_final_pred": a.get("final_pred", ""),
            "agent_confidence": (a.get("agent_selection") or {}).get("confidence", ""),
            "agent_trusted_models": ";".join((a.get("agent_selection") or {}).get("trusted_models", [])),
            "agent_conflicts": ";".join((a.get("agent_selection") or {}).get("conflicts", [])),
            "agent_visual_status": (a.get("visual_evidence") or {}).get("note", "missing"),
            "wsi_thumbnail": "missing",
            "attention_heatmap": "missing",
            "cluster_overlay": "missing",
            "representative_patches": "missing",
            "evidence_status": "partial_existing_non_visual_evidence_only",
        }
        rows.append(out)
        json_rows.append({"summary": out, "agent_panel_raw": a})
    csv_path = CPU / "showcase_cases_existing_evidence.csv"
    json_path = CPU / "showcase_cases_existing_evidence.json"
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "case_ids": case_ids,
            "m4_class_order": ["EBV", "MSI", "GS", "CIN"],
            "important_note": "This is not a complete visual evidence package. Thumbnail, attention heatmap, cluster overlay, and representative patches are absent in the current checkout.",
            "cases": json_rows,
        },
    )
    return {"csv": str(csv_path.relative_to(ROOT)), "json": str(json_path.relative_to(ROOT)), "rows": len(rows)}


def missing_material_tasks(generated: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": "T01",
            "material": "内部TCGA原始clinical.csv、原始字段、POLE病例清单、患者-全部切片映射",
            "priority": "P0",
            "status": "blocked_external_file",
            "current_action": "已保留重建版289例队列表；无法从OOF反推原始字段",
            "output_path": "results/audit_first_stage/final_289_patient_cohort.csv",
            "blocker": "clinical.csv 和 TCGA h5/features 当前不在checkout",
            "unblock_condition": "找回 clinical.csv 与 tcga_stad_uni2h/TCGA-STAD/features 后运行 build_tcga_label_table.py",
        },
        {
            "id": "T02",
            "material": "训练时原始CV fold落盘文件",
            "priority": "P0",
            "status": "blocked_original_artifact_missing",
            "current_action": "已按seed=42和site-group策略重建fold",
            "output_path": "results/audit_first_stage/cv_folds_reconstructed.csv",
            "blocker": "训练时没有保存原始fold文件",
            "unblock_condition": "找回旧fold文件，或用当前代码重新训练并保存fold registry",
        },
        {
            "id": "T03",
            "material": "M1-M6每fold正式权重、best epoch、训练日志、超参搜索记录",
            "priority": "P0",
            "status": "code_ready_retrain_required",
            "current_action": "已登记当前M1-M4 last-fold checkpoint及hash；已改造M1-M4/M5/M6训练脚本，使未来重训保存per-fold权重、fold registry、best epoch和M5 pipeline",
            "output_path": "results/cpu_supplement/model_registry_current.json; src/train_multitask.py; src/train_clinical.py; src/train_survival.py",
            "blocker": "旧实验无法反向生成每fold权重/训练日志；M5 pipeline/M6权重旧产物缺失",
            "unblock_condition": "恢复clinical.csv与TCGA特征后按改造脚本重训",
        },
        {
            "id": "T04",
            "material": "M6患者级OOF风险表和权重",
            "priority": "P1",
            "status": "code_ready_retrain_required",
            "current_action": "仅可交付旧metrics_M6_survival.json；已改造src/train_survival.py，使未来重跑保存oof_preds_M6_survival.csv和fold_registry_M6_survival.csv",
            "output_path": "results/metrics_M6_survival.json; src/train_survival.py",
            "blocker": "旧M6训练未保存OOF risk CSV和权重，当前缺clinical.csv与TCGA特征，不能直接重跑",
            "unblock_condition": "恢复clinical.csv与TCGA特征后重跑src/train_survival.py",
        },
        {
            "id": "T05",
            "material": "内部OOF高敏感度筛查阈值表",
            "priority": "P1",
            "status": "completed_now",
            "current_action": "已用现有M1/M2/M3/M5 OOF生成0.90/0.95/0.99目标敏感度阈值",
            "output_path": f"{generated['thresholds']['csv']}; {generated['thresholds']['json']}",
            "blocker": "",
            "unblock_condition": "已完成",
        },
        {
            "id": "T06",
            "material": "三个展示病例AA69、AA6K、BR-7703现有证据包",
            "priority": "P1",
            "status": "partial_completed_now",
            "current_action": "已整理OOF概率、M4四分类概率、Agent panel原始输出和视觉证据缺失状态",
            "output_path": f"{generated['showcase']['csv']}; {generated['showcase']['json']}",
            "blocker": "WSI缩略图、attention heatmap、cluster overlay、代表patch缺失",
            "unblock_condition": "找回/生成对应视觉证据文件后补齐case package",
        },
        {
            "id": "T07",
            "material": "15-20个代表病例完整证据包",
            "priority": "P1",
            "status": "blocked_visual_artifacts_missing",
            "current_action": "已可用错例表筛选候选病例，但不能生成视觉包",
            "output_path": "results/cpu_supplement/errors_M*_oof.csv",
            "blocker": "TCGA h5/坐标、缩略图、attention、cluster overlay、代表patch缺失",
            "unblock_condition": "恢复TCGA h5/坐标并运行可视化/attention导出流程，人工选取15-20例",
        },
        {
            "id": "T08",
            "material": "无监督cluster中心、代表patch、命名依据和病理医师验证",
            "priority": "P1",
            "status": "blocked_manual_pathology_required",
            "current_action": "已在报告中禁止将旧Agent形态描述作为正式cluster命名依据",
            "output_path": "results/audit_first_stage/agent_leakage_audit_and_no_leakage_plan.json",
            "blocker": "缺cluster中心/代表patch/病理医师验证记录",
            "unblock_condition": "重新导出cluster evidence sheet，并由病理医师逐簇确认",
        },
        {
            "id": "T09",
            "material": "正式无泄漏Agent全量输出和重复运行稳定性",
            "priority": "P1",
            "status": "blocked_llm_run_required",
            "current_action": "已生成无泄漏输入、prompt和schema",
            "output_path": "results/cpu_supplement/agent_no_leakage_inputs_289.csv; results/cpu_supplement/agent_no_leakage_prompt.txt; results/cpu_supplement/agent_no_leakage_output_schema.json",
            "blocker": "尚未调用LLM按无泄漏协议全量重跑；无重复运行记录",
            "unblock_condition": "确认FRIDAY_APP_ID/模型/温度/次数后运行正式Agent批处理并保存全量JSON",
        },
        {
            "id": "T10",
            "material": "Lauren、部位、分期、切片质量等亚组分析",
            "priority": "P2",
            "status": "blocked_external_file",
            "current_action": "已完成site和M4 subtype亚组",
            "output_path": "results/cpu_supplement/subgroup_metrics_by_site_and_subtype.csv",
            "blocker": "clinical.csv和切片质量QC表缺失",
            "unblock_condition": "恢复clinical.csv和WSI QC表后重新生成亚组指标",
        },
        {
            "id": "T11",
            "material": "图像+临床融合模型真实结果",
            "priority": "P2",
            "status": "blocked_model_not_implemented",
            "current_action": "已完成M1 vs M5配对bootstrap；M5确认为临床基线",
            "output_path": "results/cpu_supplement/paired_model_comparison_M1_vs_M5.json",
            "blocker": "未发现融合模型训练脚本、OOF、权重或metrics",
            "unblock_condition": "设计并实现图像+临床融合模型，按同一fold生成OOF和配对统计",
        },
        {
            "id": "T12",
            "material": "失败实验和多随机种子结果",
            "priority": "P2",
            "status": "blocked_experiment_tracking_missing",
            "current_action": "已整理当前commit、run config和现有manifest",
            "output_path": "results/audit_first_stage/reproducibility_manifest.json; results/gpu_deliverables/gpu_deliverables_run_config.json",
            "blocker": "没有统一experiment tracker；未保存失败实验/多seed训练输出",
            "unblock_condition": "建立实验登记表，重新跑多seed实验并保存每次输出",
        },
    ]


def generate_task_registry(generated: Dict[str, Any]) -> Dict[str, Any]:
    rows = missing_material_tasks(generated)
    csv_path = CPU / "missing_materials_task_list.csv"
    json_path = CPU / "missing_materials_task_list.json"
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(ROOT),
            "tasks": rows,
        },
    )
    return {"csv": str(csv_path.relative_to(ROOT)), "json": str(json_path.relative_to(ROOT)), "rows": len(rows)}


def generate_report(generated: Dict[str, Any]) -> Dict[str, Any]:
    tasks = missing_material_tasks(generated)
    report = REPORTS / "不可交付材料任务清单与执行进展.md"
    completed = [t for t in tasks if "completed" in t["status"]]
    blocked = [t for t in tasks if t not in completed]
    lines = [
        "# 不可交付材料任务清单与执行进展",
        "",
        f"日期：{datetime.now().date().isoformat()}",
        f"项目目录：`{ROOT}`",
        "",
        "## 总览",
        "",
        f"- 任务总数：{len(tasks)}",
        f"- 本轮已完成/部分完成：{len(completed)}",
        f"- 仍阻塞：{len(blocked)}",
        "",
        "本轮完成的补齐材料：",
        "",
        f"- 内部 OOF 高敏感度阈值表：`{generated['thresholds']['csv']}`",
        f"- 三个展示病例现有证据包：`{generated['showcase']['csv']}`",
        f"- 不可交付材料任务台账：`{generated['tasks']['csv']}`",
        "",
        "## 任务清单",
        "",
        "| ID | 优先级 | 材料 | 状态 | 本轮动作 | 输出 | 阻塞原因 | 解除条件 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in tasks:
        lines.append(
            "| {id} | {priority} | {material} | {status} | {current_action} | `{output_path}` | {blocker} | {unblock_condition} |".format(
                **{k: str(v).replace("|", "/") for k, v in t.items()}
            )
        )
    lines.extend(
        [
            "",
            "## 当前执行顺序建议",
            "",
            "1. 先确认是否能找回 `clinical.csv` 和 TCGA h5/features；这是补齐 POLE、slide_id、原始标签字段、真实 fold 复核、M5 缺失率和多个亚组分析的前置条件。",
            "2. 在数据恢复后，优先改造并重跑 M1-M4/M6，保存每 fold 权重、best epoch、训练日志、OOF、fold registry 和模型登记表。",
            "3. 同步生成 WSI 证据包：thumbnail、attention heatmap、cluster overlay、代表 patch 和坐标。",
            "4. 等视觉证据包完成后，再运行无泄漏 Agent 全量输出和重复运行稳定性实验。",
            "5. 最后补充 cluster 命名病理验证、融合模型、多 seed/失败实验登记，形成论文级固定版本。",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"md": str(report.relative_to(ROOT))}


def main() -> None:
    CPU.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Any] = {}
    generated["thresholds"] = generate_high_sensitivity_thresholds()
    generated["showcase"] = generate_showcase_existing_evidence()
    generated["tasks"] = generate_task_registry(generated)
    generated["report"] = generate_report(generated)
    write_json(CPU / "missing_materials_followup_manifest.json", generated)
    print(json.dumps(generated, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
