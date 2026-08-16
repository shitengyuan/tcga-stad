#!/usr/bin/env bash
set -euo pipefail

# Retrain internal TCGA-STAD models on the restored public-source clinical table
# and restored UNI2-h TCGA-STAD h5 features. Run this on a GPU node for M1-M4/M6.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-$ROOT/results/logs/internal_public_retrain_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

if [[ ! -f "$ROOT/clinical.csv" ]]; then
  echo "Missing $ROOT/clinical.csv" >&2
  exit 1
fi
if [[ ! -d "$ROOT/tcga_stad_uni2h/TCGA-STAD/features" ]]; then
  echo "Missing $ROOT/tcga_stad_uni2h/TCGA-STAD/features" >&2
  exit 1
fi

echo "ROOT=$ROOT"
echo "PY=$PY"
echo "DEVICE=$DEVICE"
echo "LOG_DIR=$LOG_DIR"

"$PY" -m src.train_multitask --task all --device "$DEVICE" 2>&1 | tee "$LOG_DIR/train_M1_M4_multitask.log"
"$PY" -m src.train_clinical 2>&1 | tee "$LOG_DIR/train_M5_clinical.log"
"$PY" -m src.train_survival --device "$DEVICE" 2>&1 | tee "$LOG_DIR/train_M6_survival.log"

"$PY" audit_tcga_feature_h5.py \
  --feature_dir tcga_stad_uni2h/TCGA-STAD/features \
  --out_csv results/audit_first_stage/tcga_uni2h_feature_manifest_after_retrain.csv \
  --out_json results/audit_first_stage/tcga_uni2h_feature_manifest_after_retrain_summary.json \
  2>&1 | tee "$LOG_DIR/audit_features_after_retrain.log"

echo "DONE internal public retrain"
