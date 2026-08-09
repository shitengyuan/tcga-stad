#!/usr/bin/env bash
set -euo pipefail

ROOT="/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/tcga-stad"

# Override these at runtime if needed:
#   PY=/path/to/python DEVICE=cuda:0 FEATURE_DIR=/path/to/features bash run_cptac_feature_inference.sh
PY="${PY:-/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python}"
FEATURE_DIR="${FEATURE_DIR:-$ROOT/results/external_cptac_features_20x256}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models}"
OUT_DIR="${OUT_DIR:-$ROOT/results/external_cptac_feature_infer_20x256}"
DEVICE="${DEVICE:-cuda:0}"
PATTERN="${PATTERN:-*}"
MAX_PATCHES="${MAX_PATCHES:-}"
MAX_SLIDES="${MAX_SLIDES:-}"
LABELS_CSV="${LABELS_CSV:-}"

if [ ! -x "$PY" ]; then
  echo "Python not executable: $PY" >&2
  exit 2
fi

if [ ! -d "$FEATURE_DIR" ]; then
  echo "Feature dir not found: $FEATURE_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

args=(
  "$ROOT/eval_cptac_features.py"
  --feature_dir "$FEATURE_DIR"
  --model_dir "$MODEL_DIR"
  --out_dir "$OUT_DIR"
  --device "$DEVICE"
  --pattern "$PATTERN"
)

if [ -n "$MAX_PATCHES" ]; then
  args+=(--max_patches "$MAX_PATCHES")
fi

if [ -n "$MAX_SLIDES" ]; then
  args+=(--max_slides "$MAX_SLIDES")
fi

if [ -n "$LABELS_CSV" ]; then
  args+=(--labels_csv "$LABELS_CSV")
fi

echo "Python: $PY"
echo "Feature dir: $FEATURE_DIR"
echo "Model dir: $MODEL_DIR"
echo "Output dir: $OUT_DIR"
echo "Device: $DEVICE"
"$PY" "${args[@]}"
