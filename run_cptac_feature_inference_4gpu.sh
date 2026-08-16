#!/usr/bin/env bash
set -euo pipefail

ROOT="/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/tcga-stad"

# Override at runtime if needed:
#   FEATURE_DIR=/path/to/features OUT_DIR=/path/to/out NUM_GPUS=4 bash run_cptac_feature_inference_4gpu.sh
#   GPU_IDS=2,3,4,5 FEATURE_DIR=/path/to/features bash run_cptac_feature_inference_4gpu.sh
PY="${PY:-/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python}"
FEATURE_DIR="${FEATURE_DIR:-$ROOT/results/external_cptac_features_20x256}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models}"
OUT_DIR="${OUT_DIR:-$ROOT/results/external_cptac_feature_infer_20x256_4gpu}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-}"
PATTERN="${PATTERN:-*}"
MAX_PATCHES="${MAX_PATCHES:-}"
LABELS_CSV="${LABELS_CSV:-}"

if [ -n "$GPU_IDS" ]; then
  IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
  NUM_GPUS="${#GPU_ARRAY[@]}"
else
  GPU_ARRAY=()
  for rank in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ARRAY+=("$rank")
  done
fi

if [ ! -x "$PY" ]; then
  echo "Python not executable: $PY" >&2
  exit 2
fi
if [ ! -d "$FEATURE_DIR" ]; then
  echo "Feature dir not found: $FEATURE_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
SHARD_ROOT="$OUT_DIR/_feature_shards"
rm -rf "$SHARD_ROOT"
mkdir -p "$SHARD_ROOT"
for rank in $(seq 0 $((NUM_GPUS - 1))); do
  mkdir -p "$SHARD_ROOT/rank_${rank}"
done

export FEATURE_DIR SHARD_ROOT NUM_GPUS
"$PY" - <<'PY'
import os
from pathlib import Path

feature_dir = Path(os.environ["FEATURE_DIR"])
shard_root = Path(os.environ["SHARD_ROOT"])
num_gpus = int(os.environ["NUM_GPUS"])
suffixes = {".h5", ".hdf5", ".pt", ".pth", ".npy", ".npz"}
files = sorted(p for p in feature_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
if not files:
    raise SystemExit(f"No feature files found in {feature_dir}")
for i, path in enumerate(files):
    rank = i % num_gpus
    link = shard_root / f"rank_{rank}" / path.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(path.resolve())
print(f"Sharded {len(files)} feature files into {num_gpus} ranks under {shard_root}", flush=True)
PY

pids=()
for rank in $(seq 0 $((NUM_GPUS - 1))); do
  rank_out="$OUT_DIR/rank_${rank}"
  mkdir -p "$rank_out"
  args=(
    "$ROOT/eval_cptac_features.py"
    --feature_dir "$SHARD_ROOT/rank_${rank}"
    --model_dir "$MODEL_DIR"
    --out_dir "$rank_out"
    --device cuda:0
    --pattern "$PATTERN"
  )
  if [ -n "$MAX_PATCHES" ]; then
    args+=(--max_patches "$MAX_PATCHES")
  fi
  if [ -n "$LABELS_CSV" ]; then
    args+=(--labels_csv "$LABELS_CSV")
  fi

  gpu="${GPU_ARRAY[$rank]}"
  echo "Starting rank $rank on CUDA_VISIBLE_DEVICES=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "${args[@]}" > "$rank_out/infer.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [ "$failed" -ne 0 ]; then
  echo "At least one rank failed. Check $OUT_DIR/rank_*/infer.log" >&2
  exit 1
fi

export OUT_DIR
"$PY" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

out_dir = Path(os.environ["OUT_DIR"])
rank_dirs = sorted(p for p in out_dir.glob("rank_*") if p.is_dir())

slide_parts = []
errors = []
for rank_dir in rank_dirs:
    slide_csv = rank_dir / "external_feature_slide_predictions.csv"
    err_json = rank_dir / "external_feature_errors.json"
    if slide_csv.exists():
        slide_parts.append(pd.read_csv(slide_csv))
    if err_json.exists():
        try:
            errors.extend(json.loads(err_json.read_text()))
        except Exception as e:
            errors.append({"rank_dir": str(rank_dir), "error": repr(e)})

if not slide_parts:
    raise SystemExit("No rank slide prediction CSVs found")

slide_df = pd.concat(slide_parts, ignore_index=True).sort_values(["patient_id", "slide_id"])
slide_df.to_csv(out_dir / "external_feature_slide_predictions.csv", index=False)

prob_cols = [c for c in slide_df.columns if "_prob_c" in c]
keep = ["patient_id", *prob_cols, "immune_sensitive_prob", "msi_prob", "ebv_prob"]
patient_df = slide_df[keep].groupby("patient_id", as_index=False).mean(numeric_only=True)
patient_df["immune_sensitive_pred"] = (patient_df["immune_sensitive_prob"] >= 0.5).astype(int)
patient_df["msi_pred"] = (patient_df["msi_prob"] >= 0.5).astype(int)
patient_df["ebv_pred"] = (patient_df["ebv_prob"] >= 0.5).astype(int)
m4_cols = [f"M4_subtype4_prob_c{i}" for i in range(4)]
names = ["EBV", "MSI", "GS", "CIN"]
patient_df["subtype4_pred_class"] = patient_df[m4_cols].to_numpy().argmax(axis=1).astype(int)
patient_df["subtype4_pred"] = [names[i] for i in patient_df["subtype4_pred_class"]]
patient_df = patient_df.merge(slide_df.groupby("patient_id").size().rename("n_slides").reset_index(), on="patient_id")
patient_df = patient_df.merge(slide_df.groupby("patient_id")["n_patches"].sum().rename("n_patches_total").reset_index(), on="patient_id")
patient_df.sort_values("patient_id").to_csv(out_dir / "external_feature_patient_predictions.csv", index=False)

(out_dir / "external_feature_errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
print(f"Wrote merged slide predictions: {out_dir / 'external_feature_slide_predictions.csv'} rows={len(slide_df)}", flush=True)
print(f"Wrote merged patient predictions: {out_dir / 'external_feature_patient_predictions.csv'} rows={len(patient_df)}", flush=True)
print(f"Wrote merged errors: {out_dir / 'external_feature_errors.json'} n={len(errors)}", flush=True)
PY
