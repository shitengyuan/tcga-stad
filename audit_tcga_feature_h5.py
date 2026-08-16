#!/usr/bin/env python3
"""Audit TCGA-STAD UNI2-h h5 feature files without loading all arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import pandas as pd


def patient_from_slide(slide_id: str) -> str:
    parts = str(slide_id).split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else str(slide_id)


def shape_or_none(dataset) -> list[int] | None:
    if dataset is None:
        return None
    return [int(x) for x in dataset.shape]


def n_patches_from_shape(shape: list[int] | None) -> int | None:
    if not shape:
        return None
    if len(shape) == 3 and shape[0] == 1:
        return int(shape[1])
    if len(shape) == 2:
        return int(shape[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    files = sorted(args.feature_dir.glob("*.h5"))
    for path in files:
        row = {
            "slide_id": path.stem,
            "patient_id": patient_from_slide(path.stem),
            "file_name": path.name,
            "path": str(path),
            "file_size": path.stat().st_size,
            "read_ok": False,
            "features_shape": None,
            "coords_shape": None,
            "coords_key": None,
            "n_patches": None,
            "feature_dim": None,
            "has_coords": False,
            "error": "",
        }
        try:
            with h5py.File(path, "r") as h:
                feature_ds = h.get("features")
                coord_key = "coords" if "coords" in h else "coords_patching" if "coords_patching" in h else None
                coord_ds = h.get(coord_key) if coord_key else None
                fshape = shape_or_none(feature_ds)
                cshape = shape_or_none(coord_ds)
                row.update(
                    {
                        "read_ok": feature_ds is not None,
                        "features_shape": json.dumps(fshape),
                        "coords_shape": json.dumps(cshape),
                        "coords_key": coord_key,
                        "n_patches": n_patches_from_shape(fshape),
                        "feature_dim": int(fshape[-1]) if fshape else None,
                        "has_coords": coord_ds is not None,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - audit should capture all read failures
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    summary = {
        "feature_dir": str(args.feature_dir),
        "n_files": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()) if len(df) else 0,
        "read_ok": int(df["read_ok"].sum()) if len(df) else 0,
        "has_coords": int(df["has_coords"].sum()) if len(df) else 0,
        "feature_dim_counts": df["feature_dim"].value_counts(dropna=False).to_dict() if len(df) else {},
        "patch_count_summary": {
            "min": int(df["n_patches"].min()) if len(df) else None,
            "median": float(df["n_patches"].median()) if len(df) else None,
            "max": int(df["n_patches"].max()) if len(df) else None,
            "sum": int(df["n_patches"].sum()) if len(df) else 0,
        },
        "total_file_size": int(df["file_size"].sum()) if len(df) else 0,
    }
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
