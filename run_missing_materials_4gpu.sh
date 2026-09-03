#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for the remaining computational materials requested in
# /medical/文档/计算机改进部分.docx.
#
# Typical 4-GPU use:
#   cd /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/tcga-stad
#   GPUS=0,1,2,3 bash run_missing_materials_4gpu.sh
#
# Optional second encoder hook:
#   SECOND_ENCODER_TRAIN_CMD='bash your_second_encoder_abmil_same_folds.sh' \
#   GPUS=0,1,2,3 bash run_missing_materials_4gpu.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEFAULT_PY="/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python"
if [[ -z "${PY:-}" && -x "$DEFAULT_PY" ]]; then
  PY="$DEFAULT_PY"
else
  PY="${PY:-python}"
fi

GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
if [[ "${#GPU_ARR[@]}" -lt 1 ]]; then
  echo "No GPU id found in GPUS=$GPUS" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/results/logs/missing_materials_4gpu_${RUN_ID}}"
LOCKED_DIR="$ROOT/results/locked_release_20260902"
GPU_OUT="$LOCKED_DIR/gpu_missing"
mkdir -p "$LOG_DIR" "$GPU_OUT"
MASTER_LOG="$LOG_DIR/master.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

RUN_LOCKED_REFRESH="${RUN_LOCKED_REFRESH:-1}"
RUN_POOLING_BASELINES="${RUN_POOLING_BASELINES:-1}"
RUN_TRANSFORMER="${RUN_TRANSFORMER:-1}"
RUN_FUSION="${RUN_FUSION:-1}"
RUN_SECOND_ENCODER="${RUN_SECOND_ENCODER:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

N_BOOT="${N_BOOT:-2000}"
EPOCHS="${EPOCHS:-30}"
N_REPEATS="${N_REPEATS:-2}"
TRANSFORMER_MAX_PATCHES="${TRANSFORMER_MAX_PATCHES:-2048}"
FUSION_MAX_PATCHES="${FUSION_MAX_PATCHES:-4096}"
HIDDEN="${HIDDEN:-256}"
DROPOUT="${DROPOUT:-0.25}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
TRANSFORMER_LAYERS="${TRANSFORMER_LAYERS:-2}"
TRANSFORMER_HEADS="${TRANSFORMER_HEADS:-4}"
TRANSFORMER_TASKS="${TRANSFORMER_TASKS:-immune_sensitive msi ebv subtype4}"

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

write_status() {
  local status="$1"
  "$PY" - "$status" <<'PY'
import json, os, pathlib, sys, time
root = pathlib.Path(os.getcwd())
status = sys.argv[1]
path = pathlib.Path(os.environ["LOG_DIR"]) / "run_status.json"
payload = {
    "status": status,
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "root": str(root),
    "log_dir": os.environ["LOG_DIR"],
    "locked_dir": str(root / "results" / "locked_release_20260902"),
    "config": {
        "GPUS": os.environ.get("GPUS"),
        "N_BOOT": os.environ.get("N_BOOT"),
        "EPOCHS": os.environ.get("EPOCHS"),
        "N_REPEATS": os.environ.get("N_REPEATS"),
        "TRANSFORMER_MAX_PATCHES": os.environ.get("TRANSFORMER_MAX_PATCHES"),
        "FUSION_MAX_PATCHES": os.environ.get("FUSION_MAX_PATCHES"),
        "TRANSFORMER_TASKS": os.environ.get("TRANSFORMER_TASKS"),
        "FORCE_RERUN": os.environ.get("FORCE_RERUN"),
    },
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

export LOG_DIR GPUS N_BOOT EPOCHS N_REPEATS TRANSFORMER_MAX_PATCHES FUSION_MAX_PATCHES TRANSFORMER_TASKS FORCE_RERUN
write_status "running"

step "preflight"
echo "ROOT=$ROOT"
echo "PY=$PY"
echo "GPUS=$GPUS"
echo "LOG_DIR=$LOG_DIR"
echo "LOCKED_DIR=$LOCKED_DIR"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
test -f "$LOCKED_DIR/patient_manifest.csv" || {
  echo "Missing locked release files. Run complete_computer_improvement_deliverables.py first." >&2
  exit 1
}

if [[ "$RUN_LOCKED_REFRESH" == "1" ]]; then
  run_logged refresh_locked_release_2000boot env N_BOOT="$N_BOOT" "$PY" complete_computer_improvement_deliverables.py
fi

if [[ "$RUN_POOLING_BASELINES" == "1" ]]; then
  run_logged run_pooling_baselines "$PY" run_pooling_baselines.py
fi

if [[ "$RUN_TRANSFORMER" == "1" ]]; then
  step "train transformer aggregator baselines"
  pids=()
  idx=0
  for task in $TRANSFORMER_TASKS; do
    case "$task" in
      immune_sensitive) metric_name="transformer_M1_immune_sensitive/metrics_transformer_M1_immune_sensitive.json" ;;
      msi) metric_name="transformer_M2_msi/metrics_transformer_M2_msi.json" ;;
      ebv) metric_name="transformer_M3_ebv/metrics_transformer_M3_ebv.json" ;;
      subtype4) metric_name="transformer_M4_subtype4/metrics_transformer_M4_subtype4.json" ;;
      *) echo "Unknown transformer task: $task" >&2; exit 1 ;;
    esac
    if [[ "$FORCE_RERUN" != "1" && -f "$GPU_OUT/$metric_name" ]]; then
      echo "Skipping transformer_${task}; metrics already exists: $GPU_OUT/$metric_name"
      continue
    fi
    gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
    name="transformer_${task}"
    log="$LOG_DIR/${name}.log"
    echo "Starting $name on GPU=$gpu log=$log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PY" train_missing_gpu_models.py \
        --model_kind transformer \
        --task "$task" \
        --device cuda \
        --epochs "$EPOCHS" \
        --n_repeats "$N_REPEATS" \
        --max_patches "$TRANSFORMER_MAX_PATCHES" \
        --hidden "$HIDDEN" \
        --dropout "$DROPOUT" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --n_layers "$TRANSFORMER_LAYERS" \
        --n_heads "$TRANSFORMER_HEADS"
    ) >"$log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "At least one transformer task failed. Check $LOG_DIR/transformer_*.log" >&2
    write_status "failed_transformer"
    exit 1
  fi
fi

if [[ "$RUN_FUSION" == "1" ]]; then
  if [[ "$FORCE_RERUN" != "1" && -f "$GPU_OUT/fusion_M1_immune_sensitive/metrics_fusion_M1_immune_sensitive.json" ]]; then
    echo "Skipping fusion; metrics already exists: $GPU_OUT/fusion_M1_immune_sensitive/metrics_fusion_M1_immune_sensitive.json"
  else
    run_logged train_end_to_end_image_clinical_fusion env CUDA_VISIBLE_DEVICES="${GPU_ARR[0]}" "$PY" train_missing_gpu_models.py \
      --model_kind fusion \
      --task immune_sensitive \
      --device cuda \
      --epochs "$EPOCHS" \
      --n_repeats "$N_REPEATS" \
      --max_patches "$FUSION_MAX_PATCHES" \
      --hidden "$HIDDEN" \
      --dropout "$DROPOUT" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY"
  fi
fi

if [[ "$RUN_SECOND_ENCODER" == "1" ]]; then
  step "second encoder hook"
  if [[ -n "${SECOND_ENCODER_TRAIN_CMD:-}" ]]; then
    echo "COMMAND: $SECOND_ENCODER_TRAIN_CMD"
    bash -lc "$SECOND_ENCODER_TRAIN_CMD" 2>&1 | tee "$LOG_DIR/second_encoder.log"
  else
    echo "SECOND_ENCODER_TRAIN_CMD is not set; cannot train second-encoder ABMIL."
    echo "本地没有第二编码器特征/权重。请先准备第二编码器特征，并把训练命令放入 SECOND_ENCODER_TRAIN_CMD。"
  fi
fi

run_logged refresh_completion_status env N_BOOT=100 "$PY" - <<'PY'
import sys
sys.path.insert(0, ".")
import complete_computer_improvement_deliverables as c
tm = c.read_json(c.OUT / "tcga_231_metrics_with_ci.json")
cm = c.read_json(c.OUT / "cptac_156_metrics_with_ci.json")
clin = c.read_json(c.OUT / "matched_clinical_metrics_231.json")
c.write_completion_status()
c.write_report(tm, cm, clin)
PY

step "final summary"
"$PY" - <<'PY'
from pathlib import Path
import json
import pandas as pd
root = Path.cwd()
out = root / "results" / "locked_release_20260902"
status = pd.read_csv(out / "completion_status_after_20260902.csv")
print(status.to_string(index=False))
for p in sorted((out / "gpu_missing").glob("*/*metrics*.json")):
    print(f"\n{p.relative_to(root)}")
    data = json.loads(p.read_text(encoding="utf-8"))
    keys = ["auroc", "average_precision", "auroc_macro_ovr", "average_precision_macro", "accuracy", "f1", "macro_f1"]
    print({k: data[k] for k in keys if k in data})
print(f"\nlocked_release={out}")
print(f"log_dir={Path('$LOG_DIR')}")
PY

write_status "complete"
echo "[$(date '+%F %T')] DONE"
