#!/usr/bin/env bash
set -euo pipefail

# One-shot GPU runner for the restored TCGA-STAD project state.
#
# Default behavior:
#   1. preflight checks
#   2. train M1-M4 image models
#   3. train M5 clinical baseline
#   4. train M6 survival model
#   5. refresh feature/cohort audit outputs
#   6. optionally extract CPTAC UNI2-h features, run CPTAC feature inference/plots,
#      and run no-leakage Agent
#
# Typical use on a 4-GPU node:
#   cd /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/tcga-stad
#   GPUS=0,1,2,3 PARALLEL_M1_M4=1 bash run_all_gpu_tasks.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEFAULT_PY="/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python"
if [[ -z "${PY:-}" && -x "$DEFAULT_PY" ]]; then
  PY="$DEFAULT_PY"
else
  PY="${PY:-python}"
fi
GPUS="${GPUS:-}"
PARALLEL_M1_M4="${PARALLEL_M1_M4:-1}"
RUN_M1_M4="${RUN_M1_M4:-1}"
RUN_M5="${RUN_M5:-1}"
RUN_M6="${RUN_M6:-1}"
RUN_AUDIT_REFRESH="${RUN_AUDIT_REFRESH:-1}"
RUN_CLUSTER_NUMERIC="${RUN_CLUSTER_NUMERIC:-1}"
RUN_CPTAC_EXTRACT="${RUN_CPTAC_EXTRACT:-0}"
RUN_CPTAC_INFERENCE="${RUN_CPTAC_INFERENCE:-0}"
RUN_CPTAC_PLOTS="${RUN_CPTAC_PLOTS:-0}"
RUN_AGENT="${RUN_AGENT:-0}"

MAX_PATCHES="${MAX_PATCHES:-8000}"
EPOCHS="${EPOCHS:-30}"
N_REPEATS="${N_REPEATS:-2}"
N_FOLDS="${N_FOLDS:-5}"
N_BOOT="${N_BOOT:-500}"
SEED="${SEED:-42}"
MIN_SITE_FOR_VAL="${MIN_SITE_FOR_VAL:-8}"

M6_MAX_PATCHES="${M6_MAX_PATCHES:-4000}"
M6_EPOCHS="${M6_EPOCHS:-15}"

SVS_DIR="${SVS_DIR:-/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/dataset/cptac-stad-histopathology}"
UNI_WEIGHTS="${UNI_WEIGHTS:-/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/uni2-h-weights/pytorch_model.bin}"
FEATURE_DIR="${FEATURE_DIR:-$ROOT/results/external_cptac_features_20x256}"
OUT_DIR="${OUT_DIR:-$ROOT/results/external_cptac_feature_infer_20x256_allgpu_${RUN_ID:-pending}}"
LABELS_CSV="${LABELS_CSV:-$SVS_DIR/labels/cptac_stad_2026_tcga_subtype_labels_qc_pass.csv}"
CPTAC_MAX_PATCHES="${CPTAC_MAX_PATCHES:-0}"
CPTAC_BATCH_SIZE="${CPTAC_BATCH_SIZE:-64}"
CPTAC_FEATURE_FORMAT="${CPTAC_FEATURE_FORMAT:-pt}"
CPTAC_PATCH_SIZE_20X="${CPTAC_PATCH_SIZE_20X:-256}"
CPTAC_ENCODER_INPUT_SIZE="${CPTAC_ENCODER_INPUT_SIZE:-224}"
CPTAC_TARGET_MPP="${CPTAC_TARGET_MPP:-0.5}"
CPTAC_STRIDE_FACTOR="${CPTAC_STRIDE_FACTOR:-1.0}"
CPTAC_TISSUE_THRESHOLD="${CPTAC_TISSUE_THRESHOLD:-0.35}"
CPTAC_MASK_MAX_SIZE="${CPTAC_MASK_MAX_SIZE:-2048}"
CPTAC_OVERWRITE_FEATURES="${CPTAC_OVERWRITE_FEATURES:-0}"
CPTAC_INFER_MAX_PATCHES="${CPTAC_INFER_MAX_PATCHES:-}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
if [[ "$OUT_DIR" == *"_pending" ]]; then
  OUT_DIR="$ROOT/results/external_cptac_feature_infer_20x256_allgpu_${RUN_ID}"
fi
LOG_DIR="${LOG_DIR:-$ROOT/results/logs/all_gpu_tasks_${RUN_ID}}"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[$(date '+%F %T')] START all GPU tasks"
echo "ROOT=$ROOT"
echo "PY=$PY"
echo "LOG_DIR=$LOG_DIR"
echo "RUN_ID=$RUN_ID"

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

step() {
  echo
  echo "[$(date '+%F %T')] === $* ==="
}

run_logged() {
  local name="$1"
  shift
  step "$name"
  echo "COMMAND: $*"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

detect_gpus() {
  if [[ -n "$GPUS" ]]; then
    echo "$GPUS"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local ids
    ids="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
    if [[ -n "$ids" ]]; then
      echo "$ids"
      return
    fi
  fi
  echo ""
}

write_status() {
  local status="$1"
  "$PY" - <<PY
import json, pathlib, time
path = pathlib.Path("$LOG_DIR/run_status.json")
obj = {
    "status": "$status",
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "run_id": "$RUN_ID",
    "log_dir": "$LOG_DIR",
    "root": "$ROOT",
    "config": {
        "GPUS": "$GPU_IDS",
        "PARALLEL_M1_M4": "$PARALLEL_M1_M4",
        "RUN_M1_M4": "$RUN_M1_M4",
        "RUN_M5": "$RUN_M5",
        "RUN_M6": "$RUN_M6",
        "RUN_AUDIT_REFRESH": "$RUN_AUDIT_REFRESH",
        "RUN_CLUSTER_NUMERIC": "$RUN_CLUSTER_NUMERIC",
        "RUN_CPTAC_EXTRACT": "$RUN_CPTAC_EXTRACT",
        "RUN_CPTAC_INFERENCE": "$RUN_CPTAC_INFERENCE",
        "RUN_CPTAC_PLOTS": "$RUN_CPTAC_PLOTS",
        "RUN_AGENT": "$RUN_AGENT",
        "MAX_PATCHES": "$MAX_PATCHES",
        "EPOCHS": "$EPOCHS",
        "N_REPEATS": "$N_REPEATS",
        "N_FOLDS": "$N_FOLDS",
        "N_BOOT": "$N_BOOT",
        "SEED": "$SEED",
        "M6_MAX_PATCHES": "$M6_MAX_PATCHES",
        "M6_EPOCHS": "$M6_EPOCHS",
        "SVS_DIR": "$SVS_DIR",
        "UNI_WEIGHTS": "$UNI_WEIGHTS",
        "FEATURE_DIR": "$FEATURE_DIR",
        "OUT_DIR": "$OUT_DIR",
        "LABELS_CSV": "$LABELS_CSV",
        "CPTAC_MAX_PATCHES": "$CPTAC_MAX_PATCHES",
        "CPTAC_BATCH_SIZE": "$CPTAC_BATCH_SIZE",
        "CPTAC_FEATURE_FORMAT": "$CPTAC_FEATURE_FORMAT",
    },
}
path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

refresh_public_audit_materials() {
  run_logged audit_tcga_features "$PY" audit_tcga_feature_h5.py \
    --feature_dir tcga_stad_uni2h/TCGA-STAD/features \
    --out_csv results/audit_first_stage/tcga_uni2h_feature_manifest_after_gpu_run.csv \
    --out_json results/audit_first_stage/tcga_uni2h_feature_manifest_after_gpu_run_summary.json

  run_logged build_tcga_label_table_with_features "$PY" build_tcga_label_table.py \
    --clinical_csv external_downloads/tcga_stad/clinical_with_gdc_slides.csv \
    --feature_dir tcga_stad_uni2h/TCGA-STAD/features \
    --out_csv results/audit_first_stage/tcga_label_table_from_cbioportal_clinical_with_gdc_slides_and_features_after_gpu_run.csv \
    --flow_json results/audit_first_stage/tcga_case_flow_from_cbioportal_clinical_with_gdc_slides_and_features_after_gpu_run.json

  run_logged refresh_public_feature_cohort "$PY" - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

out_dir = Path("results/audit_first_stage")
lab = pd.read_csv("results/audit_first_stage/tcga_label_table_from_cbioportal_clinical_with_gdc_slides_and_features_after_gpu_run.csv")
feat = pd.read_csv("results/audit_first_stage/tcga_uni2h_feature_manifest_after_gpu_run.csv")
feature_ids = set(feat["slide_id"])
rows = []
for _, r in lab.iterrows():
    slide_ids = [s for s in str(r["slide_id"]).split(";") if s and s != "nan"]
    matched = [s for s in slide_ids if s in feature_ids]
    if not matched:
        continue
    rows.append({
        "patient_id": r["patient_id"],
        "slide_id": matched[0],
        "all_matched_slide_ids": ";".join(matched),
        "site": r["site"],
        "MSI": int(r["MSI"]),
        "EBV": int(r["EBV"]),
        "M1_label": int(r["M1_label"]),
        "M4_subtype": r["M4_subtype"],
        "M4_label": int(r["M4_label"]),
        "POLE": int(r["POLE"]),
        "subtype": r["subtype"],
    })
cohort = pd.DataFrame(rows).sort_values("patient_id")
cohort = cohort.merge(feat[["slide_id", "n_patches", "feature_dim", "has_coords", "file_size"]], on="slide_id", how="left")
cohort.to_csv(out_dir / "tcga_public_feature_matched_246_cohort_after_gpu_run.csv", index=False)

missing = lab[lab["feature_status"].eq("missing_features")].copy()
missing.to_csv(out_dir / "tcga_public_label_patients_missing_uni2h_features_after_gpu_run.csv", index=False)

y = cohort["M1_label"].astype(int).values
sites = cohort["site"].astype(str).values
from collections import Counter
small = {s for s, c in Counter(sites).items() if c < 8}
cv_mask = np.array([s not in small for s in sites])
cv_idx = np.where(cv_mask)[0]
small_train = np.where(~cv_mask)[0]
small_train_set = set(small_train.tolist())
fold_rows = []
kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold_i, (tr_cv, va_cv) in enumerate(kf.split(np.zeros(len(cv_idx)), y[cv_idx], sites[cv_idx]), start=1):
    tr_full = np.concatenate([cv_idx[tr_cv], small_train])
    va_full = cv_idx[va_cv]
    for split, idxs in (("train", tr_full), ("val", va_full)):
        for idx in idxs:
            row = cohort.iloc[idx].to_dict()
            row.update({"fold": fold_i, "split": split, "seed": 42, "group_site": sites[idx], "small_site_train_only": bool(idx in small_train_set)})
            fold_rows.append(row)
folds = pd.DataFrame(fold_rows)
folds.to_csv(out_dir / "tcga_public_feature_matched_cv_folds_after_gpu_run.csv", index=False)
fold_summary = folds[folds["split"].eq("val")].groupby("fold").agg(
    n_val=("patient_id", "nunique"),
    positive_M1=("M1_label", "sum"),
    n_sites=("site", "nunique"),
    patch_median=("n_patches", "median"),
).reset_index()
fold_summary.to_csv(out_dir / "tcga_public_feature_matched_cv_fold_summary_after_gpu_run.csv", index=False)

summary = {
    "n_label_patients": int(lab["patient_id"].nunique()),
    "n_feature_matched_patients": int(cohort["patient_id"].nunique()),
    "n_missing_feature_patients": int(missing["patient_id"].nunique()),
    "feature_matched_subtype_counts": cohort["subtype"].value_counts().to_dict(),
    "missing_feature_subtype_counts": missing["subtype"].value_counts().to_dict(),
    "outputs": {
        "cohort": "results/audit_first_stage/tcga_public_feature_matched_246_cohort_after_gpu_run.csv",
        "folds": "results/audit_first_stage/tcga_public_feature_matched_cv_folds_after_gpu_run.csv",
        "fold_summary": "results/audit_first_stage/tcga_public_feature_matched_cv_fold_summary_after_gpu_run.csv",
        "missing_features": "results/audit_first_stage/tcga_public_label_patients_missing_uni2h_features_after_gpu_run.csv",
    },
}
(out_dir / "tcga_public_feature_matched_audit_summary_after_gpu_run.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
}

combine_cptac_feature_manifests() {
  run_logged combine_cptac_feature_manifests "$PY" - <<PY
import json
from pathlib import Path
import pandas as pd

out_dir = Path("$FEATURE_DIR")
world = int("$GPU_COUNT")
frames = []
errors = []
for rank in range(world):
    manifest = out_dir / f"feature_manifest.rank{rank}.csv"
    if manifest.exists() and manifest.stat().st_size > 0:
        frames.append(pd.read_csv(manifest))
    err = out_dir / f"errors.rank{rank}.json"
    if err.exists():
        try:
            data = json.loads(err.read_text(encoding="utf-8"))
            if isinstance(data, list):
                errors.extend(data)
            else:
                errors.append({"rank": rank, "error": "errors file is not a list", "path": str(err)})
        except Exception as exc:
            errors.append({"rank": rank, "error": repr(exc), "path": str(err)})
if not frames:
    raise SystemExit(f"No per-rank feature manifests found under {out_dir}")
df = pd.concat(frames, ignore_index=True)
if {"patient_id", "slide_id"}.issubset(df.columns):
    df = df.sort_values(["patient_id", "slide_id"])
df.to_csv(out_dir / "feature_manifest.csv", index=False)
(out_dir / "feature_errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote feature manifest: {out_dir / 'feature_manifest.csv'} rows={len(df)}")
print(f"Wrote feature errors: {out_dir / 'feature_errors.json'} n={len(errors)}")
PY
}

preflight() {
  step preflight
  require_path "clinical.csv"
  require_path "external_downloads/tcga_stad/clinical_with_gdc_slides.csv"
  require_path "tcga_stad_uni2h/TCGA-STAD/features"
  require_path "audit_tcga_feature_h5.py"
  require_path "generate_cluster_numeric_evidence.py"
  require_path "src/train_multitask.py"
  require_path "src/train_clinical.py"
  require_path "src/train_survival.py"
  if [[ "$RUN_CPTAC_EXTRACT" == "1" ]]; then
    require_path "$SVS_DIR"
    require_path "$UNI_WEIGHTS"
    require_path "extract_cptac_uni2h_20x256.py"
    command -v torchrun >/dev/null 2>&1 || {
      echo "Missing torchrun; use the same Python environment that provides PyTorch." >&2
      exit 1
    }
  fi
  if [[ "$RUN_CPTAC_INFERENCE" == "1" ]]; then
    if [[ "$RUN_CPTAC_EXTRACT" != "1" ]]; then
      require_path "$FEATURE_DIR"
    fi
    require_path "run_cptac_feature_inference_4gpu.sh"
    require_path "models/M1_immune_sensitive.pt"
    require_path "models/M2_msi.pt"
    require_path "models/M3_ebv.pt"
    require_path "models/M4_subtype4.pt"
  fi
  if [[ "$RUN_CPTAC_PLOTS" == "1" ]]; then
    require_path "plot_cptac_feature_results.py"
  fi

  "$PY" - <<'PY'
import importlib, pathlib, sys
mods = ["torch", "h5py", "numpy", "pandas", "sklearn", "lifelines"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    print("Missing Python dependencies:")
    print("\n".join(missing))
    sys.exit(2)
feature_count = len(list(pathlib.Path("tcga_stad_uni2h/TCGA-STAD/features").glob("*.h5")))
print(f"feature_h5_count={feature_count}")
if feature_count == 0:
    sys.exit("No h5 features found")
PY

  "$PY" - <<'PY'
from src.feature_loader import FeatureLoader

fl = FeatureLoader("tcga_stad_uni2h/TCGA-STAD/features", "clinical.csv")
counts = {}
for task in ["immune_sensitive", "msi", "ebv", "subtype4"]:
    n = 0
    for pid, row in fl.clinical.iterrows():
        _, keep = fl._make_label(task, str(row["subtype"]), row.get("label_immune_sensitive", ""))
        if keep and fl._resolve_slide_ids(pid, row.get("slide_id", "")):
            n += 1
    counts[task] = n
print(f"feature_label_matched_patients={counts}")
empty = [task for task, n in counts.items() if n == 0]
if empty:
    raise SystemExit(f"No matched patients for tasks: {empty}")
PY

  if [[ "$RUN_CPTAC_EXTRACT" == "1" ]]; then
    "$PY" - <<'PY'
import importlib
import sys

mods = ["openslide", "timm", "PIL", "tqdm"]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append(f"{mod}: {type(exc).__name__}: {exc}")
if missing:
    print("Missing Python dependencies for CPTAC feature extraction:")
    print("\n".join(missing))
    sys.exit(2)
PY
  fi

  if [[ -z "$GPU_IDS" ]]; then
    echo "No GPU detected. Run this script on a GPU node or set GPUS=0,1,2,3." >&2
    exit 3
  fi
  echo "GPU_IDS=$GPU_IDS"
  nvidia-smi || true
}

GPU_IDS="$(detect_gpus)"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
GPU_COUNT="${#GPU_ARRAY[@]}"
FINISHED=0
on_exit() {
  local code=$?
  if [[ "$FINISHED" != "1" ]]; then
    if [[ "$code" == "0" ]]; then
      write_status interrupted
    else
      write_status failed
    fi
  fi
}
trap on_exit EXIT
write_status running

preflight

if [[ "$RUN_M1_M4" == "1" ]]; then
  if [[ "$PARALLEL_M1_M4" == "1" && "$GPU_COUNT" -ge 4 ]]; then
    step "train M1-M4 in parallel"
    tasks=(immune_sensitive msi ebv subtype4)
    pids=()
    for i in "${!tasks[@]}"; do
      task="${tasks[$i]}"
      gpu="${GPU_ARRAY[$i]}"
      log="$LOG_DIR/train_${task}.log"
      echo "Starting task=$task on GPU=$gpu log=$log"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$PY" -m src.train_multitask \
          --task "$task" \
          --device cuda \
          --max_patches "$MAX_PATCHES" \
          --epochs "$EPOCHS" \
          --n_repeats "$N_REPEATS" \
          --n_folds "$N_FOLDS" \
          --n_boot "$N_BOOT" \
          --seed "$SEED" \
          --min_site_for_val "$MIN_SITE_FOR_VAL"
      ) > "$log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if [[ "$failed" != "0" ]]; then
      echo "At least one M1-M4 task failed. Check $LOG_DIR/train_*.log" >&2
      exit 4
    fi
  else
    gpu="${GPU_ARRAY[0]}"
    export CUDA_VISIBLE_DEVICES="$gpu"
    run_logged train_M1_M4_all "$PY" -m src.train_multitask \
      --task all \
      --device cuda \
      --max_patches "$MAX_PATCHES" \
      --epochs "$EPOCHS" \
      --n_repeats "$N_REPEATS" \
      --n_folds "$N_FOLDS" \
      --n_boot "$N_BOOT" \
      --seed "$SEED" \
      --min_site_for_val "$MIN_SITE_FOR_VAL"
  fi
fi

if [[ "$RUN_M5" == "1" ]]; then
  run_logged train_M5_clinical "$PY" -m src.train_clinical
fi

if [[ "$RUN_M6" == "1" ]]; then
  gpu="${GPU_ARRAY[0]}"
  export CUDA_VISIBLE_DEVICES="$gpu"
  run_logged train_M6_survival "$PY" -m src.train_survival \
    --device cuda \
    --max_patches "$M6_MAX_PATCHES" \
    --epochs "$M6_EPOCHS" \
    --n_folds "$N_FOLDS" \
    --seed "$SEED" \
    --min_site_for_val "$MIN_SITE_FOR_VAL"
fi

if [[ "$RUN_AUDIT_REFRESH" == "1" ]]; then
  refresh_public_audit_materials
fi

if [[ "$RUN_CLUSTER_NUMERIC" == "1" ]]; then
  run_logged generate_cluster_numeric_evidence "$PY" generate_cluster_numeric_evidence.py
fi

if [[ "$RUN_CPTAC_EXTRACT" == "1" ]]; then
  step "CPTAC UNI2-h feature extraction"
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  extract_args=(
    torchrun --standalone --nproc_per_node="$GPU_COUNT"
    "$ROOT/extract_cptac_uni2h_20x256.py"
    --svs_dir "$SVS_DIR"
    --uni_weights "$UNI_WEIGHTS"
    --out_dir "$FEATURE_DIR"
    --format "$CPTAC_FEATURE_FORMAT"
    --max_patches "$CPTAC_MAX_PATCHES"
    --batch_size "$CPTAC_BATCH_SIZE"
    --patch_size_20x "$CPTAC_PATCH_SIZE_20X"
    --encoder_input_size "$CPTAC_ENCODER_INPUT_SIZE"
    --target_mpp "$CPTAC_TARGET_MPP"
    --stride_factor "$CPTAC_STRIDE_FACTOR"
    --tissue_threshold "$CPTAC_TISSUE_THRESHOLD"
    --mask_max_size "$CPTAC_MASK_MAX_SIZE"
    --seed "$SEED"
  )
  if [[ "$CPTAC_OVERWRITE_FEATURES" == "1" ]]; then
    extract_args+=(--overwrite)
  fi
  "${extract_args[@]}" 2>&1 | tee "$LOG_DIR/cptac_feature_extract.log"
  combine_cptac_feature_manifests
fi

if [[ "$RUN_CPTAC_INFERENCE" == "1" ]]; then
  step "CPTAC feature inference"
  FEATURE_DIR="$FEATURE_DIR" \
  OUT_DIR="$OUT_DIR" \
  NUM_GPUS="$GPU_COUNT" \
  GPU_IDS="$GPU_IDS" \
  MAX_PATCHES="$CPTAC_INFER_MAX_PATCHES" \
  LABELS_CSV="$LABELS_CSV" \
  bash run_cptac_feature_inference_4gpu.sh 2>&1 | tee "$LOG_DIR/cptac_feature_inference.log"
fi

if [[ "$RUN_CPTAC_PLOTS" == "1" ]]; then
  plot_args=("$PY" plot_cptac_feature_results.py --pred_dir "$OUT_DIR" --out_dir "$OUT_DIR/figures")
  if [[ -f "$LABELS_CSV" ]]; then
    plot_args+=(--labels_csv "$LABELS_CSV")
  fi
  run_logged plot_cptac_feature_results "${plot_args[@]}"
fi

if [[ "$RUN_AGENT" == "1" ]]; then
  if [[ -z "${FRIDAY_APP_ID:-}" ]]; then
    echo "RUN_AGENT=1 but FRIDAY_APP_ID is empty. Skipping Agent run." | tee "$LOG_DIR/agent_skipped.log"
  else
    run_logged run_agent_panel "$PY" run_agent_panel.py --app_id "$FRIDAY_APP_ID"
  fi
fi

write_status completed
FINISHED=1
echo
echo "[$(date '+%F %T')] DONE all GPU tasks"
echo "Logs: $LOG_DIR"
