#!/usr/bin/env bash
set -euo pipefail

# End-to-end CONCH + ABMIL second-encoder baseline.
#
# Typical use:
#   cd /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/tcga-stad
#   export HF_TOKEN=...  # required if MahmoodLab/CONCH access is gated
#   GPUS=0,1,2,3 bash run_conch_abmil_4gpu.sh
#
# If you already downloaded TCGA SVS and extracted CONCH features:
#   RUN_DOWNLOAD=0 RUN_EXTRACT=0 GPUS=0,1,2,3 bash run_conch_abmil_4gpu.sh

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
NUM_GPUS="${NUM_GPUS:-${#GPU_ARR[@]}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/results/logs/conch_abmil_4gpu_${RUN_ID}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/master.log") 2>&1

RUN_DOWNLOAD="${RUN_DOWNLOAD:-1}"
RUN_EXTRACT="${RUN_EXTRACT:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"

SVS_DIR="${SVS_DIR:-$ROOT/external_downloads/tcga_stad/locked_246_svs}"
CONCH_FEATURE_DIR="${CONCH_FEATURE_DIR:-$ROOT/results/second_encoder_features_20x256/CONCH/TCGA-STAD/features}"
CONCH_OUT_DIR="${CONCH_OUT_DIR:-$ROOT/results/locked_release_20260902/second_encoder/conch_abmil}"
MANIFEST="${MANIFEST:-$ROOT/results/locked_release_20260902/patient_manifest.csv}"
UNI_FEATURE_DIR="${UNI_FEATURE_DIR:-$ROOT/tcga_stad_uni2h/TCGA-STAD/features}"

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-12}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-64}"
EXTRACT_MAX_PATCHES="${EXTRACT_MAX_PATCHES:-0}"
TRAIN_MAX_PATCHES="${TRAIN_MAX_PATCHES:-8000}"
EPOCHS="${EPOCHS:-30}"
N_REPEATS="${N_REPEATS:-2}"
HIDDEN="${HIDDEN:-256}"
DROPOUT="${DROPOUT:-0.25}"
LR="${LR:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
CONCH_BACKEND="${CONCH_BACKEND:-auto}"
CONCH_CHECKPOINT_PATH="${CONCH_CHECKPOINT_PATH:-}"
CONCH_HF_REPO="${CONCH_HF_REPO:-hf_hub:MahmoodLab/CONCH}"

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
path = pathlib.Path(os.environ["LOG_DIR"]) / "run_status.json"
payload = {
    "status": sys.argv[1],
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "root": os.getcwd(),
    "log_dir": os.environ["LOG_DIR"],
    "svs_dir": os.environ["SVS_DIR"],
    "conch_feature_dir": os.environ["CONCH_FEATURE_DIR"],
    "conch_out_dir": os.environ["CONCH_OUT_DIR"],
    "config": {
        "GPUS": os.environ.get("GPUS"),
        "NUM_GPUS": os.environ.get("NUM_GPUS"),
        "RUN_DOWNLOAD": os.environ.get("RUN_DOWNLOAD"),
        "RUN_EXTRACT": os.environ.get("RUN_EXTRACT"),
        "RUN_TRAIN": os.environ.get("RUN_TRAIN"),
        "FORCE_RERUN": os.environ.get("FORCE_RERUN"),
        "EXTRACT_MAX_PATCHES": os.environ.get("EXTRACT_MAX_PATCHES"),
        "TRAIN_MAX_PATCHES": os.environ.get("TRAIN_MAX_PATCHES"),
        "EPOCHS": os.environ.get("EPOCHS"),
        "N_REPEATS": os.environ.get("N_REPEATS"),
        "CONCH_BACKEND": os.environ.get("CONCH_BACKEND"),
        "CONCH_HF_REPO": os.environ.get("CONCH_HF_REPO"),
    },
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

export LOG_DIR SVS_DIR CONCH_FEATURE_DIR CONCH_OUT_DIR GPUS NUM_GPUS RUN_DOWNLOAD RUN_EXTRACT RUN_TRAIN FORCE_RERUN
export EXTRACT_MAX_PATCHES TRAIN_MAX_PATCHES EPOCHS N_REPEATS CONCH_BACKEND CONCH_HF_REPO
write_status "running"

step "preflight"
echo "ROOT=$ROOT"
echo "PY=$PY"
echo "GPUS=$GPUS"
echo "LOG_DIR=$LOG_DIR"
echo "SVS_DIR=$SVS_DIR"
echo "CONCH_FEATURE_DIR=$CONCH_FEATURE_DIR"
echo "CONCH_OUT_DIR=$CONCH_OUT_DIR"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
"$PY" - <<'PY'
import importlib.util
mods = ["torch", "openslide", "h5py", "timm"]
for m in mods:
    spec = importlib.util.find_spec(m)
    print(f"{m}: {'OK' if spec else 'MISSING'}")
if importlib.util.find_spec("conch") is None:
    print("conch package: MISSING; extractor will try timm fallback.")
else:
    print("conch package: OK")
PY

if [[ "$RUN_DOWNLOAD" == "1" ]]; then
  run_logged download_tcga_locked_slides "$PY" download_tcga_locked_slides_from_gdc.py \
    --manifest "$MANIFEST" \
    --out_dir "$SVS_DIR" \
    --workers "$DOWNLOAD_WORKERS"
fi

if [[ "$RUN_EXTRACT" == "1" ]]; then
  step "extract CONCH features"
  extract_args=(
    extract_conch_features_from_locked_coords.py
    --manifest "$MANIFEST"
    --uni_feature_dir "$UNI_FEATURE_DIR"
    --svs_dir "$SVS_DIR"
    --out_dir "$CONCH_FEATURE_DIR"
    --backend "$CONCH_BACKEND"
    --hf_repo "$CONCH_HF_REPO"
    --batch_size "$EXTRACT_BATCH_SIZE"
    --max_patches "$EXTRACT_MAX_PATCHES"
  )
  if [[ -n "$CONCH_CHECKPOINT_PATH" ]]; then
    extract_args+=(--checkpoint_path "$CONCH_CHECKPOINT_PATH")
  fi
  if [[ "$FORCE_RERUN" == "1" ]]; then
    extract_args+=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -m torch.distributed.run --standalone --nproc_per_node="$NUM_GPUS" "${extract_args[@]}" \
    2>&1 | tee "$LOG_DIR/extract_conch_features.log"
  "$PY" - <<'PY'
from pathlib import Path
import os
import pandas as pd
root = Path.cwd()
base = Path(os.environ["CONCH_FEATURE_DIR"]).parent
frames = []
for p in sorted(base.glob("feature_manifest.rank*.csv")):
    if p.exists() and p.stat().st_size:
        frames.append(pd.read_csv(p))
if frames:
    df = pd.concat(frames, ignore_index=True).sort_values(["patient_id", "slide_id"])
    df.to_csv(base / "feature_manifest.csv", index=False)
    print(df["status"].value_counts(dropna=False).to_string())
    print(base / "feature_manifest.csv")
else:
    raise SystemExit("No CONCH rank manifests found.")
PY
fi

if [[ "$RUN_TRAIN" == "1" ]]; then
  step "train CONCH + ABMIL M1-M4"
  pids=()
  idx=0
  for task in immune_sensitive msi ebv subtype4; do
    case "$task" in
      immune_sensitive) model_name="M1_immune_sensitive" ;;
      msi) model_name="M2_msi" ;;
      ebv) model_name="M3_ebv" ;;
      subtype4) model_name="M4_subtype4" ;;
    esac
    metric="$CONCH_OUT_DIR/$model_name/metrics_conch_abmil_${model_name}.json"
    if [[ "$FORCE_RERUN" != "1" && -f "$metric" ]]; then
      echo "Skipping CONCH+ABMIL $task; metrics already exists: $metric"
      continue
    fi
    gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
    log="$LOG_DIR/train_conch_abmil_${task}.log"
    echo "Starting CONCH+ABMIL $task on GPU=$gpu log=$log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      "$PY" train_second_encoder_abmil.py \
        --task "$task" \
        --feature_dir "$CONCH_FEATURE_DIR" \
        --out_dir "$CONCH_OUT_DIR" \
        --device cuda \
        --max_patches "$TRAIN_MAX_PATCHES" \
        --epochs "$EPOCHS" \
        --n_repeats "$N_REPEATS" \
        --hidden "$HIDDEN" \
        --dropout "$DROPOUT" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY"
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
    write_status "failed_train"
    echo "At least one CONCH+ABMIL training task failed. Check $LOG_DIR/train_conch_abmil_*.log" >&2
    exit 1
  fi
fi

step "refresh delivery status"
"$PY" - <<'PY'
import complete_computer_improvement_deliverables as deliver

deliver.write_completion_status()
print(deliver.OUT / "completion_status_after_20260902.csv")
print(deliver.OUT / "completion_status_after_20260902.json")
PY

step "final summary"
"$PY" - <<'PY'
from pathlib import Path
import json
import os
import pandas as pd
base = Path(os.environ["CONCH_FEATURE_DIR"]).parent
manifest = base / "feature_manifest.csv"
if manifest.exists():
    df = pd.read_csv(manifest)
    print("CONCH feature manifest:")
    print(df["status"].value_counts(dropna=False).to_string())
out = Path(os.environ["CONCH_OUT_DIR"])
for p in sorted(out.glob("*/metrics_conch_abmil_*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    vals = {k: d.get(k) for k in ["n", "n_pos", "auroc", "average_precision", "auroc_macro_ovr", "average_precision_macro", "accuracy", "f1", "macro_f1"] if k in d}
    print(f"\n{p}")
    print(vals)
PY

write_status "complete"
echo "[$(date '+%F %T')] DONE"
