#!/usr/bin/env python3
"""
Build a patient-level TCGA-STAD label table from a restored clinical.csv.

The current checkout does not include clinical.csv, so audit_first_stage.py
reconstructs only the final 289 OOF cohort. When clinical.csv is restored, run
this script to recreate the raw-to-model cohort table and case-flow counts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


M4_MAP = {"STAD_EBV": 0, "STAD_MSI": 1, "STAD_GS": 2, "STAD_CIN": 3}


def site_from_patient(pid: str) -> str:
    parts = str(pid).split("-")
    return parts[1] if len(parts) > 1 else "UNK"


def normalize_subtype(value) -> str:
    text = str(value).strip().upper()
    if text in {"EBV", "MSI", "GS", "CIN", "POLE"}:
        return f"STAD_{text}"
    return text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clinical_csv", type=Path, required=True)
    p.add_argument("--feature_dir", type=Path, default=None, help="Optional TCGA h5 feature dir used to mark missing features.")
    p.add_argument("--out_csv", type=Path, default=Path("results/audit_first_stage/tcga_label_table_from_clinical.csv"))
    p.add_argument("--flow_json", type=Path, default=Path("results/audit_first_stage/tcga_case_flow_from_clinical.json"))
    args = p.parse_args()

    df = pd.read_csv(args.clinical_csv)
    if "patient_id" not in df.columns:
        raise ValueError("clinical_csv must contain patient_id")
    if "subtype" not in df.columns:
        raise ValueError("clinical_csv must contain subtype")

    df = df.copy()
    df["subtype"] = df["subtype"].map(normalize_subtype)
    df["site"] = df["patient_id"].map(site_from_patient)
    if "slide_id" not in df.columns:
        df["slide_id"] = pd.NA

    df["MSI"] = df["subtype"].eq("STAD_MSI").astype(int)
    df["EBV"] = df["subtype"].eq("STAD_EBV").astype(int)
    df["POLE"] = df["subtype"].eq("STAD_POLE").astype(int)
    df["M1_label"] = pd.NA
    df.loc[df["subtype"].isin(["STAD_MSI", "STAD_EBV"]), "M1_label"] = 1
    df.loc[df["subtype"].isin(["STAD_GS", "STAD_CIN"]), "M1_label"] = 0
    df["M4_subtype"] = df["subtype"].str.replace("STAD_", "", regex=False)
    df["M4_label"] = df["subtype"].map(M4_MAP)

    feature_ids = set()
    if args.feature_dir:
        feature_ids = {p.stem for p in args.feature_dir.glob("*.h5")}
    def feature_status(slide_ids) -> str:
        if not feature_ids:
            return "not_checked"
        ids = [s for s in str(slide_ids).split(";") if s]
        if not ids:
            return "no_slide_id"
        present = [s in feature_ids for s in ids]
        if all(present):
            return "all_slides_have_features"
        if any(present):
            return "partial_features"
        return "missing_features"

    df["feature_status"] = df["slide_id"].map(feature_status)
    df["include_status"] = "excluded"
    df["include_exclude_reason"] = "unknown_or_unmapped_subtype"
    include = df["M1_label"].notna() & df["M4_label"].notna()
    df.loc[include, "include_status"] = "included"
    df.loc[include, "include_exclude_reason"] = "mapped subtype in EBV/MSI/GS/CIN"
    df.loc[df["POLE"].eq(1), "include_exclude_reason"] = "POLE excluded from primary subtype tasks"

    cols = [
        "patient_id", "slide_id", "site", "MSI", "EBV", "M1_label",
        "M4_subtype", "M4_label", "POLE", "include_status",
        "include_exclude_reason", "feature_status", "subtype",
    ]
    extra = [c for c in df.columns if c not in cols]
    out = df[cols + extra].sort_values("patient_id")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    flow = {
        "clinical_csv": str(args.clinical_csv),
        "feature_dir": str(args.feature_dir) if args.feature_dir else None,
        "n_raw_patients": int(df["patient_id"].nunique()),
        "subtype_counts_raw": df["subtype"].value_counts(dropna=False).to_dict(),
        "include_status_counts": df["include_status"].value_counts(dropna=False).to_dict(),
        "include_exclude_reason_counts": df["include_exclude_reason"].value_counts(dropna=False).to_dict(),
        "feature_status_counts": df["feature_status"].value_counts(dropna=False).to_dict(),
    }
    args.flow_json.parent.mkdir(parents=True, exist_ok=True)
    args.flow_json.write_text(json.dumps(flow, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.flow_json}")


if __name__ == "__main__":
    main()
