#!/usr/bin/env bash
set -euo pipefail

# Download CONCH model files from Hugging Face and locked TCGA-STAD slides from GDC.
#
# Token handling:
#   export HF_TOKEN=...
# or put HF_TOKEN=... in /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/appkey.env
# The token is not printed and is not written to manifests.
#
# Typical use:
#   cd /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/tcga-stad
#   GPUS=0,1,2,3 bash download_conch_model_and_tcga.sh
#
# Download only CONCH model:
#   RUN_TCGA=0 bash download_conch_model_and_tcga.sh
#
# Download only TCGA slides:
#   RUN_CONCH=0 TCGA_WORKERS=16 bash download_conch_model_and_tcga.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEFAULT_PY="/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python"
if [[ -z "${PY:-}" && -x "$DEFAULT_PY" ]]; then
  PY="$DEFAULT_PY"
else
  PY="${PY:-python}"
fi

ENV_FILE="${ENV_FILE:-/gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/appkey.env}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$ROOT/results/logs/download_conch_tcga_${RUN_ID}}"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/master.log") 2>&1

RUN_CONCH="${RUN_CONCH:-1}"
RUN_TCGA="${RUN_TCGA:-1}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

CONCH_REPO_ID="${CONCH_REPO_ID:-MahmoodLab/CONCH}"
CONCH_REVISION="${CONCH_REVISION:-}"
HF_ENDPOINT="${HF_ENDPOINT:-}"
CONCH_OUT_DIR="${CONCH_OUT_DIR:-$ROOT/external_models/CONCH/MahmoodLab_CONCH}"

TCGA_MANIFEST="${TCGA_MANIFEST:-$ROOT/results/locked_release_20260902/patient_manifest.csv}"
TCGA_OUT_DIR="${TCGA_OUT_DIR:-$ROOT/external_downloads/tcga_stad/locked_246_svs}"
TCGA_WORKERS="${TCGA_WORKERS:-12}"
TCGA_RETRIES="${TCGA_RETRIES:-5}"
TCGA_TIMEOUT="${TCGA_TIMEOUT:-180}"
TCGA_LIMIT="${TCGA_LIMIT:-0}"

step() {
  echo
  echo "[$(date '+%F %T')] === $* ==="
}

write_status() {
  local status="$1"
  "$PY" - "$status" <<'PY'
import json, os, pathlib, sys, time
p = pathlib.Path(os.environ["LOG_DIR"]) / "run_status.json"
payload = {
    "status": sys.argv[1],
    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "root": os.getcwd(),
    "log_dir": os.environ["LOG_DIR"],
    "run_conch": os.environ.get("RUN_CONCH"),
    "run_tcga": os.environ.get("RUN_TCGA"),
    "conch_repo_id": os.environ.get("CONCH_REPO_ID"),
    "conch_out_dir": os.environ.get("CONCH_OUT_DIR"),
    "tcga_manifest": os.environ.get("TCGA_MANIFEST"),
    "tcga_out_dir": os.environ.get("TCGA_OUT_DIR"),
    "tcga_workers": os.environ.get("TCGA_WORKERS"),
    "token_recorded": False,
}
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

export LOG_DIR RUN_CONCH RUN_TCGA CONCH_REPO_ID CONCH_OUT_DIR TCGA_MANIFEST TCGA_OUT_DIR TCGA_WORKERS
write_status "running"

step "preflight"
echo "ROOT=$ROOT"
echo "PY=$PY"
echo "LOG_DIR=$LOG_DIR"
echo "RUN_CONCH=$RUN_CONCH"
echo "RUN_TCGA=$RUN_TCGA"
echo "CONCH_REPO_ID=$CONCH_REPO_ID"
if [[ -n "$HF_ENDPOINT" ]]; then
  echo "HF_ENDPOINT=$HF_ENDPOINT"
fi
echo "CONCH_OUT_DIR=$CONCH_OUT_DIR"
echo "TCGA_MANIFEST=$TCGA_MANIFEST"
echo "TCGA_OUT_DIR=$TCGA_OUT_DIR"
echo "TCGA_WORKERS=$TCGA_WORKERS"
"$PY" - <<'PY'
import importlib.util
for mod in ["huggingface_hub", "pandas"]:
    print(f"{mod}: {'OK' if importlib.util.find_spec(mod) else 'MISSING'}")
PY

if [[ "$RUN_CONCH" == "1" ]]; then
  step "download CONCH model"
  conch_args=(
    download_conch_model_from_hf.py
    --repo_id "$CONCH_REPO_ID"
    --out_dir "$CONCH_OUT_DIR"
    --env_file "$ENV_FILE"
  )
  if [[ -n "$HF_ENDPOINT" ]]; then
    conch_args+=(--endpoint "$HF_ENDPOINT")
  fi
  if [[ -n "$CONCH_REVISION" ]]; then
    conch_args+=(--revision "$CONCH_REVISION")
  fi
  if [[ "$FORCE_DOWNLOAD" == "1" ]]; then
    conch_args+=(--force_download)
  fi
  "$PY" "${conch_args[@]}" 2>&1 | tee "$LOG_DIR/download_conch_model.log"
fi

if [[ "$RUN_TCGA" == "1" ]]; then
  step "download locked TCGA-STAD slides"
  tcga_args=(
    download_tcga_locked_slides_from_gdc.py
    --manifest "$TCGA_MANIFEST"
    --out_dir "$TCGA_OUT_DIR"
    --workers "$TCGA_WORKERS"
    --retries "$TCGA_RETRIES"
    --timeout "$TCGA_TIMEOUT"
  )
  if [[ "$TCGA_LIMIT" != "0" ]]; then
    tcga_args+=(--limit "$TCGA_LIMIT")
  fi
  if [[ "$FORCE_DOWNLOAD" == "1" ]]; then
    tcga_args+=(--overwrite)
  fi
  "$PY" "${tcga_args[@]}" 2>&1 | tee "$LOG_DIR/download_tcga_slides.log"
fi

step "summary"
"$PY" - <<'PY'
from pathlib import Path
import json
import os
import pandas as pd

conch_dir = Path(os.environ["CONCH_OUT_DIR"])
conch_manifest = conch_dir / "download_manifest.json"
if conch_manifest.exists():
    d = json.loads(conch_manifest.read_text(encoding="utf-8"))
    print("CONCH:", {"local_dir": d.get("local_dir"), "n_files": d.get("n_files"), "total_bytes": d.get("total_bytes")})
else:
    print("CONCH manifest missing:", conch_manifest)

tcga_manifest = Path(os.environ["TCGA_OUT_DIR"]) / "gdc_download_manifest.csv"
if tcga_manifest.exists():
    df = pd.read_csv(tcga_manifest)
    print("TCGA download status:")
    print(df["status"].value_counts(dropna=False).to_string())
    if "bytes" in df:
        print("TCGA downloaded/skipped bytes:", int(df["bytes"].fillna(0).sum()))
else:
    print("TCGA manifest missing:", tcga_manifest)
PY

write_status "complete"
echo "[$(date '+%F %T')] DONE"
