#!/usr/bin/env python3
"""Refresh delivery registries and completion audit from current local artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
AUDIT = RESULTS / "audit_first_stage"
CPU = RESULTS / "cpu_supplement"
REPORTS = ROOT / "reports"
M4_CLASSES = ["EBV", "MSI", "GS", "CIN"]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_log_dir() -> Path | None:
    dirs = sorted((RESULTS / "logs").glob("all_gpu_tasks_*"), key=lambda p: p.stat().st_mtime)
    return dirs[-1] if dirs else None


def collect_python_versions() -> dict[str, Any]:
    mods = ["torch", "numpy", "pandas", "sklearn", "h5py", "openslide", "timm", "PIL"]
    out: dict[str, Any] = {"python": platform.python_version(), "modules": {}}
    for mod in mods:
        try:
            m = __import__(mod)
            out["modules"][mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            out["modules"][mod] = {"error": repr(exc)}
    try:
        import torch

        out["torch_cuda"] = {
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else [],
        }
    except Exception:
        pass
    return out


def model_registry() -> dict[str, Any]:
    CPU.mkdir(parents=True, exist_ok=True)
    commit = run_text(["git", "rev-parse", "HEAD"])
    status = run_text(["git", "status", "--short"])
    log_dir = latest_log_dir()
    registry: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_status_short": status,
        "is_clean_git_version": not bool(status),
        "latest_gpu_log_dir": str(log_dir.relative_to(ROOT)) if log_dir else None,
        "models": [],
    }
    model_specs = {
        "M1_immune_sensitive": {"file": "models/M1_immune_sensitive.pt", "task": "immune_sensitive", "n_classes": 2},
        "M2_msi": {"file": "models/M2_msi.pt", "task": "msi", "n_classes": 2},
        "M3_ebv": {"file": "models/M3_ebv.pt", "task": "ebv", "n_classes": 2},
        "M4_subtype4": {"file": "models/M4_subtype4.pt", "task": "subtype4", "n_classes": 4, "class_order": M4_CLASSES},
        "M5_clinical": {"task": "clinical", "n_classes": 2},
        "M6_survival": {"task": "survival", "n_classes": 1},
    }
    for name, spec in model_specs.items():
        metrics = RESULTS / f"metrics_{name}.json"
        oof = RESULTS / f"oof_preds_{name}.csv"
        fold_registry = RESULTS / f"fold_registry_{name}.csv"
        entry: dict[str, Any] = {
            "name": name,
            **spec,
            "metrics_json": str(metrics.relative_to(ROOT)) if metrics.exists() else None,
            "oof_csv": str(oof.relative_to(ROOT)) if oof.exists() else None,
            "fold_registry_csv": str(fold_registry.relative_to(ROOT)) if fold_registry.exists() else None,
        }
        if "file" in spec:
            p = ROOT / spec["file"]
            entry["sha256"] = sha256(p)
            entry["exists"] = p.exists()
        if metrics.exists():
            m = read_json(metrics)
            entry["metrics_summary"] = {
                k: m.get(k)
                for k in ["n_samples", "fold_scores", "fold_cindex", "fold_mean", "fold_std", "oof_auc", "oof_ap", "oof_cindex", "bootstrap_ci", "config"]
                if k in m
            }
        if fold_registry.exists():
            fr = pd.read_csv(fold_registry)
            entry["fold_registry_rows"] = int(len(fr))
            entry["folds"] = sorted(pd.unique(fr["fold"]).tolist()) if "fold" in fr.columns else []
            entry["repeats"] = sorted(pd.unique(fr["repeat"]).tolist()) if "repeat" in fr.columns else []
        if name.startswith("M") and name not in {"M5_clinical", "M6_survival"}:
            files = sorted((ROOT / "models" / "per_fold" / name).glob("*.pt"))
            entry["per_fold_checkpoints"] = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in files]
        elif name == "M5_clinical":
            files = sorted((ROOT / "models" / "per_fold" / name).glob("*.joblib"))
            entry["per_fold_pipelines"] = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in files]
        elif name == "M6_survival":
            files = sorted((ROOT / "models" / "per_fold" / name).glob("*"))
            entry["per_fold_artifacts"] = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p)} for p in files if p.is_file()]
        registry["models"].append(entry)
    registry["remaining_model_gaps"] = [
        "M6 per-fold survival artifacts are only complete after rerunning patched src/train_survival.py.",
        "M1-M4 per-epoch CSV logs are only complete after rerunning patched src/train_multitask.py.",
        "No hyperparameter-search record is present.",
        "No multi-seed formal experiment registry is present.",
    ]
    write_json(CPU / "model_registry_current.json", registry)
    return registry


def uni2h_config() -> dict[str, Any]:
    weights = Path("/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/uni2-h-weights/pytorch_model.bin")
    manifest = AUDIT / "tcga_uni2h_feature_manifest_summary.json"
    cptac_manifest = RESULTS / "external_cptac_features_20x256" / "feature_manifest.csv"
    cfg = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "encoder": "UNI2-h",
        "weights_path": str(weights),
        "weights_sha256": sha256(weights),
        "tcga_feature_manifest_summary": str(manifest.relative_to(ROOT)) if manifest.exists() else None,
        "tcga_feature_manifest": "results/audit_first_stage/tcga_uni2h_feature_manifest.csv",
        "cptac_feature_manifest": str(cptac_manifest.relative_to(ROOT)) if cptac_manifest.exists() else None,
        "feature_dim": 1536,
        "tcga_hf_archive": {
            "path": "external_downloads/hf_uni2h/TCGA/TCGA-STAD.tar.gz",
            "sha256": sha256(ROOT / "external_downloads/hf_uni2h/TCGA/TCGA-STAD.tar.gz"),
        },
        "cptac_extraction_config": {
            "script": "extract_cptac_uni2h_20x256.py",
            "script_sha256": sha256(ROOT / "extract_cptac_uni2h_20x256.py"),
            "target_mpp": 0.5,
            "magnification": "20x-equivalent",
            "patch_size_20x": 256,
            "encoder_input_size": 224,
            "stride_factor": 1.0,
            "tissue_threshold": 0.35,
            "mask_max_size": 2048,
            "max_patches_default": 0,
            "format_default": "pt",
        },
        "environment": collect_python_versions(),
    }
    if manifest.exists():
        cfg["tcga_feature_summary"] = read_json(manifest)
    if cptac_manifest.exists():
        df = pd.read_csv(cptac_manifest)
        cfg["cptac_feature_summary"] = {
            "n_feature_rows": int(len(df)),
            "n_patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else None,
            "status_counts": df["status"].value_counts(dropna=False).to_dict() if "status" in df.columns else {},
        }
    write_json(AUDIT / "uni2h_feature_extraction_config_current.json", cfg)
    return cfg


def brier_scores() -> dict[str, Any]:
    CPU.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    specs = [
        ("internal_oof", "M1_immune_sensitive", RESULTS / "oof_preds_M1_immune_sensitive.csv", "label", "prob_c1"),
        ("internal_oof", "M2_msi", RESULTS / "oof_preds_M2_msi.csv", "label", "prob_c1"),
        ("internal_oof", "M3_ebv", RESULTS / "oof_preds_M3_ebv.csv", "label", "prob_c1"),
        ("internal_oof", "M5_clinical", RESULTS / "oof_preds_M5_clinical.csv", "label", "prob_c1"),
    ]
    for cohort, model, path, label_col, score_col in specs:
        if not path.exists():
            continue
        df = pd.read_csv(path).dropna(subset=[label_col, score_col])
        rows.append(
            {
                "cohort": cohort,
                "model": model,
                "n": int(len(df)),
                "n_pos": int(df[label_col].astype(int).sum()),
                "brier_score": float(brier_score_loss(df[label_col].astype(int), df[score_col].astype(float))),
                "source": str(path.relative_to(ROOT)),
            }
        )
    cptac = RESULTS / "external_cptac_feature_infer_20x256_4gpu" / "figures" / "cptac_patient_predictions_with_qc_labels.csv"
    cptac_specs = [
        ("M1_immune_sensitive", "immune_sensitive", "immune_sensitive_prob"),
        ("M2_msi", "msi", "msi_prob"),
        ("M3_ebv", "ebv", "ebv_prob"),
    ]
    if cptac.exists():
        df = pd.read_csv(cptac)
        for model, label_col, score_col in cptac_specs:
            if label_col in df.columns and score_col in df.columns:
                sub = df.dropna(subset=[label_col, score_col])
                rows.append(
                    {
                        "cohort": "cptac_qc_labeled",
                        "model": model,
                        "n": int(len(sub)),
                        "n_pos": int(sub[label_col].astype(int).sum()),
                        "brier_score": float(brier_score_loss(sub[label_col].astype(int), sub[score_col].astype(float))),
                        "source": str(cptac.relative_to(ROOT)),
                    }
                )
    out_csv = CPU / "brier_scores.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    summary = {"generated_at": datetime.now().isoformat(timespec="seconds"), "rows": rows, "csv": str(out_csv.relative_to(ROOT))}
    write_json(CPU / "brier_scores.json", summary)
    return summary


def package_latest_lightweight() -> dict[str, Any]:
    package_dir = RESULTS / "delivery_package"
    package_dir.mkdir(parents=True, exist_ok=True)
    filelist = package_dir / "delivery_filelist_20260817.txt"
    tar_path = package_dir / f"tcga_stad_delivery_lightweight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    include = [
        "README.md",
        "reports",
        "src",
        "run_all_gpu_tasks.sh",
        "run_remaining_gpu_tasks.sh",
        "run_cptac_feature_inference_4gpu.sh",
        "extract_cptac_uni2h_20x256.py",
        "eval_cptac_features.py",
        "plot_cptac_feature_results.py",
        "audit_first_stage.py",
        "generate_cpu_supplement.py",
        "generate_missing_materials_followup.py",
        "generate_cluster_numeric_evidence.py",
        "finalize_delivery_materials.py",
        "generate_visual_evidence_package.py",
        "run_agent_no_leakage.py",
        "results/audit_first_stage",
        "results/cpu_supplement",
        "results/figures",
        "results/visual_evidence_package",
        "results/external_cptac_feature_infer_20x256_4gpu",
        "results/logs",
        "results/metrics_M1_immune_sensitive.json",
        "results/metrics_M2_msi.json",
        "results/metrics_M3_ebv.json",
        "results/metrics_M4_subtype4.json",
        "results/metrics_M5_clinical.json",
        "results/metrics_M6_survival.json",
        "results/oof_preds_M1_immune_sensitive.csv",
        "results/oof_preds_M2_msi.csv",
        "results/oof_preds_M3_ebv.csv",
        "results/oof_preds_M4_subtype4.csv",
        "results/oof_preds_M5_clinical.csv",
        "results/oof_preds_M6_survival.csv",
        "results/fold_registry_M1_immune_sensitive.csv",
        "results/fold_registry_M2_msi.csv",
        "results/fold_registry_M3_ebv.csv",
        "results/fold_registry_M4_subtype4.csv",
        "results/fold_registry_M5_clinical.csv",
        "results/fold_registry_M6_survival.csv",
    ]
    paths: list[str] = []
    for rel in include:
        p = ROOT / rel
        if p.exists():
            if p.is_dir():
                for child in p.rglob("*"):
                    if child.is_file():
                        if any(part in {"tcga_stad_uni2h", "external_cptac_features_20x256", "external_downloads", "_feature_shards"} for part in child.parts):
                            continue
                        paths.append(str(child.relative_to(ROOT)))
            else:
                paths.append(rel)
    paths = sorted(set(paths))
    filelist.write_text("\n".join(paths) + "\n", encoding="utf-8")
    subprocess.check_call(["tar", "-czf", str(tar_path), "-C", str(ROOT), "-T", str(filelist)])
    meta = {
        "tar": str(tar_path.relative_to(ROOT)),
        "filelist": str(filelist.relative_to(ROOT)),
        "n_files": len(paths),
        "sha256": sha256(tar_path),
        "note": "Lightweight package excludes raw WSI, h5/pt features and large external downloads.",
    }
    write_json(package_dir / "latest_lightweight_delivery_package.json", meta)
    return meta


def completion_matrix(registry: dict[str, Any], uni_cfg: dict[str, Any], brier: dict[str, Any], package: dict[str, Any]) -> list[dict[str, Any]]:
    visual_summary = RESULTS / "visual_evidence_package" / "visual_evidence_summary.json"
    visual_status = "partial"
    visual_next = "已生成coordinate-level thumbnail/attention/cluster证据；病理命名和rendered patch验证仍需人工确认"
    if not visual_summary.exists():
        visual_status = "script_ready_gpu_optional"
        visual_next = "运行generate_visual_evidence_package.py生成thumbnail/attention/cluster坐标图；病理命名仍需人工确认"

    agent_outputs = CPU / "agent_no_leakage_outputs" / "agent_no_leakage_outputs_rules.csv"
    agent_stability = CPU / "agent_no_leakage_outputs" / "agent_no_leakage_stability_rules.json"
    agent_status = "partial_rule_based_complete" if agent_outputs.exists() and agent_stability.exists() else "script_ready"
    agent_next = (
        "规则版无标签泄漏结构化输出已生成；如需正式DeepSeek/GLM Agent结论，仍需配置FRIDAY_APP_ID后重跑LLM版"
        if agent_status == "partial_rule_based_complete"
        else "运行run_agent_no_leakage.py生成规则版无泄漏Agent输出；正式LLM版需FRIDAY_APP_ID"
    )

    rows = [
        {"item": "公开246例patient-slide-label-feature清单", "status": "complete", "path": "results/audit_first_stage/tcga_public_feature_matched_246_cohort_after_gpu_run.csv", "next_action": ""},
        {"item": "历史289例OOF队列原始slide_id/POLE", "status": "partial", "path": "results/audit_first_stage/final_289_patient_cohort.csv", "next_action": "需要历史训练用clinical或用公开246例替代说明"},
        {"item": "M1-M4 OOF/metrics/fold registry/per-fold权重", "status": "complete", "path": "results/oof_preds_M*_*.csv; results/fold_registry_M*.csv; models/per_fold/M1-M4", "next_action": ""},
        {"item": "M1-M4逐epoch日志", "status": "script_ready_gpu_rerun_required", "path": "src/train_multitask.py; run_remaining_gpu_tasks.sh", "next_action": "GPU重跑RUN_M1_M4=1生成results/training_epoch_logs/M1-M4"},
        {"item": "M5临床变量/缺失率/per-fold pipeline", "status": "complete", "path": "results/M5_clinical_feature_missingness.csv; models/per_fold/M5_clinical", "next_action": ""},
        {"item": "M6 OOF/fold registry", "status": "complete", "path": "results/oof_preds_M6_survival.csv; results/fold_registry_M6_survival.csv", "next_action": ""},
        {"item": "M6 per-fold survival权重", "status": "script_ready_gpu_rerun_required", "path": "src/train_survival.py; run_remaining_gpu_tasks.sh", "next_action": "GPU重跑RUN_M6=1生成models/per_fold/M6_survival"},
        {"item": "UNI2-h配置/权重hash", "status": "complete", "path": "results/audit_first_stage/uni2h_feature_extraction_config_current.json", "next_action": ""},
        {"item": "Brier score", "status": "complete", "path": "results/cpu_supplement/brier_scores.csv", "next_action": ""},
        {"item": "CPTAC推理/评估/图表", "status": "complete", "path": "results/external_cptac_feature_infer_20x256_4gpu", "next_action": "注意可评估QC标签42例，完整推理167例/666 slides"},
        {"item": "展示病例视觉证据", "status": visual_status, "path": "results/visual_evidence_package", "next_action": visual_next},
        {"item": "Agent无泄漏全量结构化输出", "status": agent_status, "path": "results/cpu_supplement/agent_no_leakage_outputs", "next_action": agent_next},
        {"item": "图像+临床融合模型", "status": "missing_model_not_implemented", "path": "", "next_action": "需单独设计并训练融合模型"},
        {"item": "多seed/失败实验登记", "status": "missing_experiment_required", "path": "", "next_action": "需按实验矩阵重跑并登记"},
        {"item": "最新轻量交付包", "status": "complete", "path": package["tar"], "next_action": ""},
        {"item": "干净Git固定版本", "status": "missing_commit_required", "path": "", "next_action": "过滤数据后git add/commit/push"},
    ]
    out_csv = CPU / "delivery_completion_matrix_20260817.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item", "status", "path", "next_action"])
        w.writeheader()
        w.writerows(rows)
    write_json(CPU / "delivery_completion_matrix_20260817.json", rows)
    return rows


def write_report(rows: list[dict[str, Any]], package: dict[str, Any]) -> None:
    done = sum(1 for r in rows if r["status"] == "complete")
    partial = sum(1 for r in rows if r["status"].startswith("partial") or "script_ready" in r["status"])
    missing = len(rows) - done - partial
    lines = [
        "# 交付材料最终核对与补齐进展",
        "",
        f"日期：2026-08-17",
        f"项目目录：`{ROOT}`",
        f"当前 Git commit：`{run_text(['git', 'rev-parse', 'HEAD'])}`",
        "",
        "## 总结",
        "",
        f"- 已完成：{done}",
        f"- 部分完成/脚本已准备：{partial}",
        f"- 仍缺失：{missing}",
        f"- 最新轻量交付包：`{package['tar']}`",
        "",
        "目前可以交付队列、标签、分折、OOF、metrics、CPTAC外部验证、模型登记、UNI2-h配置、Brier score和轻量交付包。仍不能声称完全论文级闭环，主要缺正式无泄漏Agent全量输出、WSI视觉证据人工验证、M6 per-fold survival权重重跑、多seed/失败实验和图像+临床融合模型。",
        "",
        "## 明细",
        "",
        "| 材料 | 状态 | 路径 | 下一步 |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['item']} | {r['status']} | `{r['path']}` | {r['next_action']} |")
    (REPORTS / "交付材料最终核对与补齐进展_20260817.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    registry = model_registry()
    uni_cfg = uni2h_config()
    brier = brier_scores()
    package = package_latest_lightweight()
    rows = completion_matrix(registry, uni_cfg, brier, package)
    write_report(rows, package)
    print(json.dumps({
        "model_registry": "results/cpu_supplement/model_registry_current.json",
        "uni2h_config": "results/audit_first_stage/uni2h_feature_extraction_config_current.json",
        "brier": "results/cpu_supplement/brier_scores.csv",
        "completion_matrix": "results/cpu_supplement/delivery_completion_matrix_20260817.csv",
        "report": "reports/交付材料最终核对与补齐进展_20260817.md",
        "package": package,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
