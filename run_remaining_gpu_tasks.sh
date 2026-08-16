#!/usr/bin/env bash
set -euo pipefail

# Run only the remaining GPU/long-running deliverable tasks.
#
# Defaults:
#   - rerun M6 survival to save per-fold encoder/PCA/Cox artifacts
#   - generate visual evidence coordinate heatmaps/overlays
#   - refresh registries, Brier scores, completion matrix and lightweight package
#
# Optional:
#   RUN_M1_M4=1 reruns M1-M4 with patched per-epoch CSV logging.
#   RUN_CPTAC=1 reruns CPTAC feature inference/plots using existing features.
#   RUN_AGENT=1 requires FRIDAY_APP_ID and runs the old panel agent; keep this off
#              for formal no-leakage work unless the prompt/workflow is reviewed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python}"
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
GPU0="${GPU_ARRAY[0]}"

RUN_M1_M4="${RUN_M1_M4:-0}"
RUN_M6="${RUN_M6:-1}"
RUN_VISUAL="${RUN_VISUAL:-1}"
RUN_CPTAC="${RUN_CPTAC:-0}"
RUN_AGENT="${RUN_AGENT:-0}"
RUN_FINALIZE="${RUN_FINALIZE:-1}"

MAX_PATCHES="${MAX_PATCHES:-8000}"
EPOCHS="${EPOCHS:-30}"
N_REPEATS="${N_REPEATS:-2}"
N_FOLDS="${N_FOLDS:-5}"
N_BOOT="${N_BOOT:-1000}"
SEED="${SEED:-42}"
MIN_SITE_FOR_VAL="${MIN_SITE_FOR_VAL:-8}"

M6_MAX_PATCHES="${M6_MAX_PATCHES:-4000}"
M6_EPOCHS="${M6_EPOCHS:-15}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/results/logs/remaining_gpu_tasks_${RUN_ID}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/master.log") 2>&1

run_logged() {
  local name="$1"
  shift
  echo
  echo "[$(date '+%F %T')] === $name ==="
  echo "COMMAND: $*"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

echo "[$(date '+%F %T')] START remaining GPU tasks"
echo "ROOT=$ROOT"
echo "PY=$PY"
echo "GPUS=$GPUS"
echo "LOG_DIR=$LOG_DIR"

"$PY" - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} count={torch.cuda.device_count()}", flush=True)
PY

if [[ "$RUN_M1_M4" == "1" ]]; then
  tasks=(immune_sensitive msi ebv subtype4)
  pids=()
  for i in "${!tasks[@]}"; do
    task="${tasks[$i]}"
    gpu="${GPU_ARRAY[$i]:-$GPU0}"
    log="$LOG_DIR/train_${task}.log"
    echo "Starting M1-M4 task=$task on CUDA_VISIBLE_DEVICES=$gpu"
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
    echo "M1-M4 rerun failed. Check $LOG_DIR/train_*.log" >&2
    exit 4
  fi
fi

if [[ "$RUN_M6" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU0"
  run_logged train_M6_survival_save_weights "$PY" -m src.train_survival \
    --device cuda \
    --max_patches "$M6_MAX_PATCHES" \
    --epochs "$M6_EPOCHS" \
    --n_folds "$N_FOLDS" \
    --seed "$SEED" \
    --min_site_for_val "$MIN_SITE_FOR_VAL"
fi

if [[ "$RUN_VISUAL" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU0"
  run_logged generate_visual_evidence_package "$PY" generate_visual_evidence_package.py --device cuda:0
fi

if [[ "$RUN_CPTAC" == "1" ]]; then
  FEATURE_DIR="${FEATURE_DIR:-$ROOT/results/external_cptac_features_20x256}" \
  OUT_DIR="${OUT_DIR:-$ROOT/results/external_cptac_feature_infer_20x256_4gpu}" \
  GPU_IDS="$GPUS" \
  bash run_cptac_feature_inference_4gpu.sh 2>&1 | tee "$LOG_DIR/cptac_feature_inference.log"
  run_logged plot_cptac_feature_results "$PY" plot_cptac_feature_results.py \
    --pred_dir "${OUT_DIR:-$ROOT/results/external_cptac_feature_infer_20x256_4gpu}" \
    --out_dir "${OUT_DIR:-$ROOT/results/external_cptac_feature_infer_20x256_4gpu}/figures" \
    --labels_csv "/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/dataset/cptac-stad-histopathology/labels/cptac_stad_2026_tcga_subtype_labels_qc_pass.csv"
fi

if [[ "$RUN_AGENT" == "1" ]]; then
  if [[ -z "${FRIDAY_APP_ID:-}" ]]; then
    echo "RUN_AGENT=1 but FRIDAY_APP_ID is empty" >&2
    exit 5
  fi
  run_logged run_agent_panel "$PY" run_agent_panel.py --app_id "$FRIDAY_APP_ID"
fi

if [[ "$RUN_FINALIZE" == "1" ]]; then
  run_logged finalize_delivery_materials "$PY" finalize_delivery_materials.py
fi

echo
echo "[$(date '+%F %T')] DONE remaining GPU tasks"
echo "Logs: $LOG_DIR"
