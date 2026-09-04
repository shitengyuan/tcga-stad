#!/usr/bin/env python3
"""Download locked TCGA slide images from GDC using slide filenames.

The locked patient manifest stores slide IDs like:
TCGA-B7-5816-01Z-00-DX1.1B05F96A-D5E2-4366-A098-A861313F3461

The suffix after the last dot is embedded in the SVS filename, but is not the
current GDC file_id. This script resolves each filename through the GDC files
endpoint, downloads the corresponding SVS, and writes an audit manifest.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "results" / "locked_release_20260902" / "patient_manifest.csv"
DEFAULT_OUT = ROOT / "external_downloads" / "tcga_stad" / "locked_246_svs"
GDC_DATA_URL = "https://api.gdc.cancer.gov/data/{uuid}"
GDC_FILES_URL = "https://api.gdc.cancer.gov/files"


def slide_uuid(slide_id: str) -> str:
    if "." not in slide_id:
        raise ValueError(f"slide_id has no UUID suffix: {slide_id}")
    return slide_id.rsplit(".", 1)[1]


def resolve_file_id(slide_id: str, timeout: int) -> tuple[str | None, str]:
    file_name = f"{slide_id}.svs"
    filters = {"op": "=", "content": {"field": "file_name", "value": [file_name]}}
    params = urllib.parse.urlencode(
        {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,state,access,data_format,data_type",
            "format": "JSON",
            "size": "2",
        }
    )
    try:
        with urllib.request.urlopen(f"{GDC_FILES_URL}?{params}", timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        hits = payload.get("data", {}).get("hits", [])
        if not hits:
            return None, "no_file_name_match"
        hit = hits[0]
        return hit.get("file_id"), json.dumps(hit, ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        return None, f"resolve_failed:{type(exc).__name__}:{exc}"


def download_one(
    row: dict[str, Any],
    out_dir: Path,
    retries: int,
    timeout: int,
    overwrite: bool,
    skip_resolve: bool,
    resolve_only: bool,
) -> dict[str, Any]:
    slide_id = str(row["slide_id"])
    uuid_suffix = slide_uuid(slide_id)
    file_id = uuid_suffix if skip_resolve else None
    file_meta = "used_slide_id_suffix_as_file_id" if skip_resolve else ""
    out_path = out_dir / f"{slide_id}.svs"
    tmp_path = out_path.with_suffix(out_path.suffix + f".tmp.{os.getpid()}")
    rec = {
        "patient_id": row.get("patient_id", ""),
        "slide_id": slide_id,
        "gdc_file_uuid_suffix": uuid_suffix,
        "gdc_file_id": "",
        "gdc_file_meta": "",
        "out_path": str(out_path),
        "status": "",
        "bytes": 0,
        "error": "",
    }
    if out_path.exists() and out_path.stat().st_size > 0 and not overwrite:
        rec["status"] = "skipped_existing"
        rec["bytes"] = int(out_path.stat().st_size)
        return rec
    if not file_id:
        file_id, file_meta = resolve_file_id(slide_id, timeout)
        rec["gdc_file_meta"] = file_meta
        if not file_id:
            rec["status"] = "failed_resolve"
            rec["error"] = file_meta
            return rec
    rec["gdc_file_id"] = file_id
    if file_meta:
        rec["gdc_file_meta"] = file_meta
    if resolve_only:
        rec["status"] = "resolved"
        return rec
    url = GDC_DATA_URL.format(uuid=file_id)
    for attempt in range(1, retries + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            with urllib.request.urlopen(url, timeout=timeout) as resp, tmp_path.open("wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if tmp_path.stat().st_size == 0:
                raise RuntimeError("downloaded empty file")
            tmp_path.replace(out_path)
            rec["status"] = "downloaded"
            rec["bytes"] = int(out_path.stat().st_size)
            return rec
        except Exception as exc:
            rec["error"] = f"attempt {attempt}/{retries}: {type(exc).__name__}: {exc}"
            time.sleep(min(10, attempt * 2))
    rec["status"] = "failed"
    return rec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_resolve", action="store_true", help="Use slide_id suffix as GDC file_id. Mostly for debugging; normal TCGA filenames need resolution.")
    parser.add_argument("--resolve_only", action="store_true", help="Resolve GDC file_id and write manifest without downloading SVS bytes.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true", help="Only write the planned download manifest; do not download files.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    rows = df[["patient_id", "slide_id"]].drop_duplicates().to_dict("records")
    if args.limit:
        rows = rows[: args.limit]

    manifest_path = args.out_dir / "gdc_download_manifest.csv"
    results = []
    if args.dry_run:
        for r in rows:
            slide_id = str(r["slide_id"])
            out_path = args.out_dir / f"{slide_id}.svs"
            results.append(
                {
                    "patient_id": r.get("patient_id", ""),
                    "slide_id": slide_id,
                    "gdc_file_uuid_suffix": slide_uuid(slide_id),
                    "gdc_file_id": "",
                    "gdc_file_meta": "",
                    "out_path": str(out_path),
                    "status": "dry_run",
                    "bytes": int(out_path.stat().st_size) if out_path.exists() else 0,
                    "error": "",
                }
            )
        pd.DataFrame(results).sort_values(["patient_id", "slide_id"]).to_csv(manifest_path, index=False)
        print({"manifest": str(manifest_path), "rows": len(results), "status_counts": {"dry_run": len(results)}})
        return
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(download_one, r, args.out_dir, args.retries, args.timeout, args.overwrite, args.skip_resolve, args.resolve_only)
            for r in rows
        ]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            print(f"[{i}/{len(rows)}] {rec['status']} {rec['slide_id']} bytes={rec['bytes']} {rec['error']}", flush=True)
            if i % 10 == 0:
                pd.DataFrame(results).to_csv(manifest_path, index=False)
    pd.DataFrame(results).sort_values(["status", "patient_id", "slide_id"]).to_csv(manifest_path, index=False)
    print({"manifest": str(manifest_path), "rows": len(results), "status_counts": pd.Series([r["status"] for r in results]).value_counts().to_dict()})


if __name__ == "__main__":
    main()
