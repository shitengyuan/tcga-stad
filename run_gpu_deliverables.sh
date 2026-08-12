#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run GPU-dependent deliverables for the TCGA-STAD/CPTAC external validation package.

Default workflow:
  1. Check environment and GPU availability.
  2. Extract CPTAC UNI2-h 20x/256 features with torchrun on NUM_GPUS GPUs.
  3. Run multi-GPU ABMIL inference on saved CPTAC features.
  4. Generate CPTAC plots/metrics.
  5. Refresh first-stage audit outputs.
  6. Refresh CPU supplement outputs that depend on the new inference.
  7. Package lightweight delivery materials into a tar.gz.

Common run:
  bash run_gpu_deliverables.sh

Useful overrides:
  NUM_GPUS=4 MAX_PATCHES=8192 bash run_gpu_deliverables.sh
  MAX_PATCHES=0 bash run_gpu_deliverables.sh              # all sampled tissue patches
  RUN_CPTAC_EXTRACT=0 bash run_gpu_deliverables.sh        # reuse existing features
  RUN_PACKAGE=0 bash run_gpu_deliverables.sh

Internal retraining is off by default because current training code overwrites
models/*.pt and does not save per-fold checkpoints. To run it anyway:
  RUN_INTERNAL_TRAIN=1 CONFIRM_OVERWRITE_MODELS=1 bash run_gpu_deliverables.sh

Environment variables:
  ROOT, PY, NUM_GPUS, SVS_DIR, UNI_WEIGHTS, MODEL_DIR, FEATURE_DIR, INFER_DIR,
  LABELS_CSV, MAX_PATCHES, BATCH_SIZE, RUN_CPTAC_EXTRACT, RUN_CPTAC_INFER,
  RUN_CPTAC_PLOTS, RUN_AUDIT, RUN_CPU_SUPPLEMENT, RUN_PACKAGE,
  RUN_INTERNAL_TRAIN, CONFIRM_OVERWRITE_MODELS, PACKAGE_FEATURES.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PY:-/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python}"
NUM_GPUS="${NUM_GPUS:-4}"

SVS_DIR="${SVS_DIR:-/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/dataset/cptac-stad-histopathology}"
UNI_WEIGHTS="${UNI_WEIGHTS:-/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/uni2-h-weights/pytorch_model.bin}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models}"
FEATURE_DIR="${FEATURE_DIR:-$ROOT/results/external_cptac_features_20x256}"
INFER_DIR="${INFER_DIR:-$ROOT/results/external_cptac_feature_infer_20x256_4gpu}"
LABELS_CSV="${LABELS_CSV:-$SVS_DIR/labels/cptac_stad_2026_tcga_subtype_labels_qc_pass.csv}"

STAGE_DIR="${STAGE_DIR:-$ROOT/results/gpu_deliverables}"
LOG_DIR="${LOG_DIR:-$STAGE_DIR/logs}"
PACKAGE_DIR="${PACKAGE_DIR:-$STAGE_DIR/package}"

MAX_PATCHES="${MAX_PATCHES:-8192}"
INFER_MAX_PATCHES="${INFER_MAX_PATCHES:-}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FEATURE_FORMAT="${FEATURE_FORMAT:-pt}"
PATCH_SIZE_20X="${PATCH_SIZE_20X:-256}"
ENCODER_INPUT_SIZE="${ENCODER_INPUT_SIZE:-224}"
TARGET_MPP="${TARGET_MPP:-0.5}"
STRIDE_FACTOR="${STRIDE_FACTOR:-1.0}"
TISSUE_THRESHOLD="${TISSUE_THRESHOLD:-0.35}"
MASK_MAX_SIZE="${MASK_MAX_SIZE:-2048}"
SEED="${SEED:-42}"
N_BOOT="${N_BOOT:-2000}"

RUN_CPTAC_EXTRACT="${RUN_CPTAC_EXTRACT:-1}"
RUN_CPTAC_INFER="${RUN_CPTAC_INFER:-1}"
RUN_CPTAC_PLOTS="${RUN_CPTAC_PLOTS:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_CPU_SUPPLEMENT="${RUN_CPU_SUPPLEMENT:-1}"
RUN_PACKAGE="${RUN_PACKAGE:-1}"
RUN_INTERNAL_TRAIN="${RUN_INTERNAL_TRAIN:-0}"
CONFIRM_OVERWRITE_MODELS="${CONFIRM_OVERWRITE_MODELS:-0}"
OVERWRITE_FEATURES="${OVERWRITE_FEATURES:-0}"
PACKAGE_FEATURES="${PACKAGE_FEATURES:-0}"

export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR" "$PACKAGE_DIR"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

fail() {
  printf '[%s] ERROR: %s\n' "$(date -Is)" "$*" >&2
  exit 2
}

run_logged() {
  local name="$1"
  shift
  log "START $name"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  log "DONE  $name"
}

require_file() {
  [[ -f "$1" ]] || fail "required file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || fail "required directory not found: $1"
}

write_run_config() {
  "$PY" - <<PY
import json
import os
import platform
import subprocess
from pathlib import Path

root = Path("$ROOT")
out = Path("$STAGE_DIR") / "gpu_deliverables_run_config.json"
def run(cmd):
    try:
        return subprocess.check_output(cmd, cwd=str(root), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
cfg = {
    "generated_at": __import__("datetime").datetime.now().isoformat(),
    "root": "$ROOT",
    "python": "$PY",
    "python_version": platform.python_version(),
    "git_commit": run(["git", "rev-parse", "HEAD"]),
    "git_status_short": run(["git", "status", "--short"]),
    "num_gpus": int("$NUM_GPUS"),
    "svs_dir": "$SVS_DIR",
    "uni_weights": "$UNI_WEIGHTS",
    "model_dir": "$MODEL_DIR",
    "feature_dir": "$FEATURE_DIR",
    "infer_dir": "$INFER_DIR",
    "labels_csv": "$LABELS_CSV",
    "max_patches": "$MAX_PATCHES",
    "infer_max_patches": "$INFER_MAX_PATCHES",
    "batch_size": int("$BATCH_SIZE"),
    "patch_size_20x": int("$PATCH_SIZE_20X"),
    "encoder_input_size": int("$ENCODER_INPUT_SIZE"),
    "target_mpp": float("$TARGET_MPP"),
    "stride_factor": float("$STRIDE_FACTOR"),
    "tissue_threshold": float("$TISSUE_THRESHOLD"),
    "mask_max_size": int("$MASK_MAX_SIZE"),
    "seed": int("$SEED"),
    "n_boot": int("$N_BOOT"),
    "stages": {
        "run_cptac_extract": "$RUN_CPTAC_EXTRACT",
        "run_cptac_infer": "$RUN_CPTAC_INFER",
        "run_cptac_plots": "$RUN_CPTAC_PLOTS",
        "run_audit": "$RUN_AUDIT",
        "run_cpu_supplement": "$RUN_CPU_SUPPLEMENT",
        "run_package": "$RUN_PACKAGE",
        "run_internal_train": "$RUN_INTERNAL_TRAIN",
    },
}
try:
    import torch
    cfg["torch"] = {
        "version": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
except Exception as e:
    cfg["torch_probe_error"] = repr(e)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(cfg, indent=2, ensure_ascii=False), flush=True)
PY
}

preflight() {
  require_file "$PY"
  require_dir "$ROOT"
  require_dir "$MODEL_DIR"
  require_file "$MODEL_DIR/M1_immune_sensitive.pt"
  require_file "$MODEL_DIR/M2_msi.pt"
  require_file "$MODEL_DIR/M3_ebv.pt"
  require_file "$MODEL_DIR/M4_subtype4.pt"
  require_dir "$SVS_DIR"
  require_file "$UNI_WEIGHTS"
  if [[ -n "$LABELS_CSV" && ! -f "$LABELS_CSV" ]]; then
    log "WARN labels CSV not found, plots/audit will still run but supervised CPTAC metrics may be limited: $LABELS_CSV"
  fi
  "$PY" - <<PY
import sys
import torch
need = int("$NUM_GPUS")
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} count={torch.cuda.device_count()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this environment")
if torch.cuda.device_count() < need:
    raise SystemExit(f"Need {need} GPUs but torch sees {torch.cuda.device_count()}")
PY
}

run_cptac_extract() {
  mkdir -p "$FEATURE_DIR"
  local args=(
    torchrun --standalone --nproc_per_node="$NUM_GPUS"
    "$ROOT/extract_cptac_uni2h_20x256.py"
    --svs_dir "$SVS_DIR"
    --uni_weights "$UNI_WEIGHTS"
    --out_dir "$FEATURE_DIR"
    --format "$FEATURE_FORMAT"
    --max_patches "$MAX_PATCHES"
    --batch_size "$BATCH_SIZE"
    --patch_size_20x "$PATCH_SIZE_20X"
    --encoder_input_size "$ENCODER_INPUT_SIZE"
    --target_mpp "$TARGET_MPP"
    --stride_factor "$STRIDE_FACTOR"
    --tissue_threshold "$TISSUE_THRESHOLD"
    --mask_max_size "$MASK_MAX_SIZE"
    --seed "$SEED"
  )
  if [[ "$OVERWRITE_FEATURES" == "1" ]]; then
    args+=(--overwrite)
  fi
  run_logged "01_extract_cptac_uni2h_20x256" "${args[@]}"
}

run_cptac_infer() {
  require_dir "$FEATURE_DIR"
  mkdir -p "$INFER_DIR"
  log "START 02_infer_cptac_features_4gpu"
  (
    export PY FEATURE_DIR MODEL_DIR OUT_DIR="$INFER_DIR" NUM_GPUS
    export PATTERN="*"
    export MAX_PATCHES="$INFER_MAX_PATCHES"
    if [[ -f "$LABELS_CSV" ]]; then
      export LABELS_CSV
    else
      export LABELS_CSV=""
    fi
    bash "$ROOT/run_cptac_feature_inference_4gpu.sh"
  ) 2>&1 | tee "$LOG_DIR/02_infer_cptac_features_4gpu.log"
  log "DONE  02_infer_cptac_features_4gpu"
}

run_cptac_plots() {
  require_dir "$INFER_DIR"
  local args=(
    "$PY" "$ROOT/plot_cptac_feature_results.py"
    --pred_dir "$INFER_DIR"
    --out_dir "$INFER_DIR/figures"
  )
  if [[ -f "$LABELS_CSV" ]]; then
    args+=(--labels_csv "$LABELS_CSV")
  fi
  run_logged "03_plot_cptac_feature_results" "${args[@]}"
}

sync_cptac_inference_for_audit() {
  local audit_infer_dir="$ROOT/results/audit_first_stage/cptac_193_feature_inference"
  local default_feature_dir="$ROOT/results/external_cptac_features_20x256"
  require_file "$INFER_DIR/external_feature_slide_predictions.csv"
  require_file "$INFER_DIR/external_feature_patient_predictions.csv"
  mkdir -p "$audit_infer_dir"
  cp -f "$INFER_DIR/external_feature_slide_predictions.csv" "$audit_infer_dir/external_feature_slide_predictions.csv"
  cp -f "$INFER_DIR/external_feature_patient_predictions.csv" "$audit_infer_dir/external_feature_patient_predictions.csv"
  if [[ -f "$INFER_DIR/external_feature_errors.json" ]]; then
    cp -f "$INFER_DIR/external_feature_errors.json" "$audit_infer_dir/external_feature_errors.json"
  fi
  if [[ "$FEATURE_DIR" != "$default_feature_dir" ]]; then
    log "WARN audit_first_stage.py uses $default_feature_dir for CPTAC feature counts; custom FEATURE_DIR will not change those counts unless you also copy/link it there."
  fi
}

run_audit() {
  sync_cptac_inference_for_audit
  local args=(
    "$PY" "$ROOT/audit_first_stage.py"
    --device cpu
    --n_boot "$N_BOOT"
    --seed "$SEED"
    --skip_cptac_inference
  )
  run_logged "04_audit_first_stage" "${args[@]}"
}

run_cpu_supplement() {
  run_logged "05_generate_cpu_supplement" "$PY" "$ROOT/generate_cpu_supplement.py"
}

run_internal_train() {
  if [[ "$CONFIRM_OVERWRITE_MODELS" != "1" ]]; then
    fail "RUN_INTERNAL_TRAIN=1 would overwrite results/oof_preds_*.csv, results/metrics_*.json and models/*.pt. Re-run with CONFIRM_OVERWRITE_MODELS=1 if intended."
  fi
  if [[ ! -f "$ROOT/clinical.csv" || ! -d "$ROOT/tcga_stad_uni2h/TCGA-STAD/features" ]]; then
    fail "internal training needs clinical.csv and tcga_stad_uni2h/TCGA-STAD/features"
  fi
  for task in immune_sensitive msi ebv subtype4; do
    run_logged "10_train_${task}" \
      "$PY" -m src.train_multitask \
      --task "$task" \
      --device cuda \
      --max_patches "${TRAIN_MAX_PATCHES:-999999}" \
      --epochs "${TRAIN_EPOCHS:-30}" \
      --n_folds "${TRAIN_N_FOLDS:-5}" \
      --n_repeats "${TRAIN_N_REPEATS:-3}" \
      --n_boot "${TRAIN_N_BOOT:-1000}" \
      --seed "$SEED"
  done
  run_logged "11_train_survival" \
    "$PY" -m src.train_survival \
    --device cuda \
    --max_patches "${SURV_MAX_PATCHES:-999999}" \
    --epochs "${SURV_EPOCHS:-20}" \
    --n_folds "${SURV_N_FOLDS:-5}" \
    --seed "$SEED"
}

make_package() {
  mkdir -p "$PACKAGE_DIR"
  local list="$PACKAGE_DIR/package_filelist.txt"
  local tar_path="$PACKAGE_DIR/gpu_deliverables_$(date +%Y%m%d_%H%M%S).tar.gz"
  local feature_rel=""
  local infer_rel=""
  : > "$list"

  case "$FEATURE_DIR" in
    "$ROOT"/*) feature_rel="${FEATURE_DIR#$ROOT/}" ;;
  esac
  case "$INFER_DIR" in
    "$ROOT"/*) infer_rel="${INFER_DIR#$ROOT/}" ;;
  esac

  add_path() {
    local p="$1"
    if [[ -e "$ROOT/$p" ]]; then
      printf '%s\n' "$p" >> "$list"
    fi
  }

  add_path "README.md"
  add_path "run_gpu_deliverables.sh"
  add_path "run_cptac_feature_inference_4gpu.sh"
  add_path "extract_cptac_uni2h_20x256.py"
  add_path "eval_cptac_features.py"
  add_path "plot_cptac_feature_results.py"
  add_path "audit_first_stage.py"
  add_path "generate_cpu_supplement.py"
  add_path "build_tcga_label_table.py"
  add_path "reports/GPU交付脚本说明.md"
  add_path "reports/第一阶段审计交付清单.md"
  add_path "reports/CPU补充材料交付清单.md"
  add_path "results/audit_first_stage"
  add_path "results/cpu_supplement"
  add_path "results/gpu_deliverables/gpu_deliverables_run_config.json"
  add_path "results/gpu_deliverables/logs"
  if [[ -n "$infer_rel" ]]; then
    add_path "$infer_rel/external_feature_slide_predictions.csv"
    add_path "$infer_rel/external_feature_patient_predictions.csv"
    add_path "$infer_rel/external_feature_errors.json"
    add_path "$infer_rel/CPTAC_STAD_external_validation_report.md"
    add_path "$infer_rel/figures"
  else
    log "WARN INFER_DIR is outside ROOT, skipping inference files in package: $INFER_DIR"
  fi
  if [[ -n "$feature_rel" ]]; then
    add_path "$feature_rel/feature_manifest.csv"
    for f in "$FEATURE_DIR"/feature_manifest.rank*.csv "$FEATURE_DIR"/errors.rank*.json; do
      [[ -e "$f" ]] && printf '%s\n' "${f#$ROOT/}" >> "$list"
    done
  else
    log "WARN FEATURE_DIR is outside ROOT, skipping feature manifests in package: $FEATURE_DIR"
  fi

  if [[ "$PACKAGE_FEATURES" == "1" && -n "$feature_rel" ]]; then
    find "$FEATURE_DIR" -maxdepth 1 -type f \( -name '*.pt' -o -name '*.h5' -o -name '*.hdf5' \) -printf '%P\n' \
      | while read -r f; do printf '%s\n' "$feature_rel/$f" >> "$list"; done
  elif [[ "$PACKAGE_FEATURES" == "1" ]]; then
    log "WARN PACKAGE_FEATURES=1 ignored because FEATURE_DIR is outside ROOT"
  fi

  sort -u "$list" -o "$list"
  tar -czf "$tar_path" -C "$ROOT" -T "$list"
  log "Wrote package: $tar_path"
  log "Package file list: $list"
}

main() {
  cd "$ROOT"
  log "GPU deliverables start"
  preflight 2>&1 | tee "$LOG_DIR/00_preflight.log"
  write_run_config 2>&1 | tee "$LOG_DIR/00_run_config.log"

  if [[ "$RUN_INTERNAL_TRAIN" == "1" ]]; then
    run_internal_train
  fi
  if [[ "$RUN_CPTAC_EXTRACT" == "1" ]]; then
    run_cptac_extract
  fi
  if [[ "$RUN_CPTAC_INFER" == "1" ]]; then
    run_cptac_infer
  fi
  if [[ "$RUN_CPTAC_PLOTS" == "1" ]]; then
    run_cptac_plots
  fi
  if [[ "$RUN_AUDIT" == "1" ]]; then
    run_audit
  fi
  if [[ "$RUN_CPU_SUPPLEMENT" == "1" ]]; then
    run_cpu_supplement
  fi
  if [[ "$RUN_PACKAGE" == "1" ]]; then
    make_package
  fi

  log "GPU deliverables done"
  log "Feature dir: $FEATURE_DIR"
  log "Inference dir: $INFER_DIR"
  log "Audit dir: $ROOT/results/audit_first_stage"
  log "CPU supplement dir: $ROOT/results/cpu_supplement"
  log "Logs: $LOG_DIR"
}

main "$@"
