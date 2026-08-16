#!/usr/bin/env python3
"""Attach public GDC TCGA-STAD slide metadata to a clinical label table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def stem_slide_id(file_name: str) -> str:
    text = str(file_name).strip()
    if text.lower().endswith(".svs"):
        return text[:-4]
    return text


def slide_class(slide_id: str) -> str:
    text = str(slide_id).upper()
    if "-DX" in text:
        return "diagnostic"
    if "-TS" in text:
        return "top_slide"
    if "-BS" in text:
        return "bottom_slide"
    return "other"


def semicolon(values) -> str:
    cleaned = [str(v) for v in values if pd.notna(v) and str(v)]
    return ";".join(sorted(dict.fromkeys(cleaned)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clinical_csv", type=Path, required=True)
    parser.add_argument("--gdc_slide_tsv", type=Path, required=True)
    parser.add_argument("--out_clinical_csv", type=Path, required=True)
    parser.add_argument("--out_slide_csv", type=Path, required=True)
    parser.add_argument("--out_audit_json", type=Path, required=True)
    args = parser.parse_args()

    clinical = pd.read_csv(args.clinical_csv)
    slides = pd.read_csv(args.gdc_slide_tsv, sep="\t")

    required_slide_cols = {
        "cases.0.submitter_id",
        "cases.0.samples.0.sample_type",
        "cases.0.samples.0.submitter_id",
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "access",
    }
    missing = sorted(required_slide_cols - set(slides.columns))
    if missing:
        raise ValueError(f"gdc_slide_tsv missing columns: {missing}")

    slide_map = pd.DataFrame(
        {
            "patient_id": slides["cases.0.submitter_id"],
            "sample_id": slides["cases.0.samples.0.submitter_id"],
            "sample_type": slides["cases.0.samples.0.sample_type"],
            "slide_id": slides["file_name"].map(stem_slide_id),
            "file_name": slides["file_name"],
            "file_id": slides["file_id"],
            "file_size": slides["file_size"],
            "md5sum": slides["md5sum"],
            "access": slides["access"],
            "data_format": slides.get("data_format", pd.NA),
            "state": slides.get("state", pd.NA),
        }
    )
    slide_map["slide_class"] = slide_map["slide_id"].map(slide_class)
    slide_map["is_primary_tumor"] = slide_map["sample_type"].eq("Primary Tumor")
    slide_map["is_diagnostic"] = slide_map["slide_class"].eq("diagnostic")

    primary = slide_map[slide_map["is_primary_tumor"]].copy()
    diagnostic = primary[primary["is_diagnostic"]].copy()
    grouped_all = primary.groupby("patient_id")["slide_id"].apply(semicolon)
    grouped_dx = diagnostic.groupby("patient_id")["slide_id"].apply(semicolon)

    out_clinical = clinical.copy()
    out_clinical["gdc_primary_tumor_slide_ids"] = out_clinical["patient_id"].map(grouped_all)
    out_clinical["gdc_diagnostic_slide_ids"] = out_clinical["patient_id"].map(grouped_dx)
    out_clinical["gdc_primary_tumor_slide_count"] = (
        out_clinical["gdc_primary_tumor_slide_ids"].fillna("").map(lambda x: len([s for s in str(x).split(";") if s]))
    )
    out_clinical["gdc_diagnostic_slide_count"] = (
        out_clinical["gdc_diagnostic_slide_ids"].fillna("").map(lambda x: len([s for s in str(x).split(";") if s]))
    )
    out_clinical["slide_id"] = out_clinical["gdc_diagnostic_slide_ids"].where(
        out_clinical["gdc_diagnostic_slide_ids"].notna() & out_clinical["gdc_diagnostic_slide_ids"].ne(""),
        out_clinical["gdc_primary_tumor_slide_ids"],
    )
    out_clinical["slide_id_source"] = "GDC primary tumor Slide Image metadata"
    out_clinical.loc[out_clinical["slide_id"].isna() | out_clinical["slide_id"].eq(""), "slide_id_source"] = (
        "not found in GDC primary tumor Slide Image metadata"
    )

    slide_with_labels = slide_map.merge(
        clinical.drop(columns=["slide_id"], errors="ignore"),
        on="patient_id",
        how="left",
        suffixes=("", "_clinical"),
    )
    slide_with_labels["has_cbioportal_label"] = slide_with_labels["subtype"].notna()

    args.out_clinical_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_slide_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_audit_json.parent.mkdir(parents=True, exist_ok=True)
    out_clinical.sort_values("patient_id").to_csv(args.out_clinical_csv, index=False)
    slide_with_labels.sort_values(["patient_id", "slide_id"]).to_csv(args.out_slide_csv, index=False)

    audit = {
        "clinical_csv": str(args.clinical_csv),
        "gdc_slide_tsv": str(args.gdc_slide_tsv),
        "n_clinical_patients": int(clinical["patient_id"].nunique()),
        "n_gdc_slide_rows": int(len(slide_map)),
        "n_gdc_slide_patients": int(slide_map["patient_id"].nunique()),
        "sample_type_counts": slide_map["sample_type"].value_counts(dropna=False).to_dict(),
        "slide_class_counts": slide_map["slide_class"].value_counts(dropna=False).to_dict(),
        "primary_tumor_slide_rows": int(primary.shape[0]),
        "primary_tumor_slide_patients": int(primary["patient_id"].nunique()),
        "diagnostic_slide_rows": int(diagnostic.shape[0]),
        "diagnostic_slide_patients": int(diagnostic["patient_id"].nunique()),
        "clinical_patients_with_any_primary_tumor_slide": int(out_clinical["gdc_primary_tumor_slide_count"].gt(0).sum()),
        "clinical_patients_with_diagnostic_slide": int(out_clinical["gdc_diagnostic_slide_count"].gt(0).sum()),
        "clinical_patients_without_gdc_primary_tumor_slide": sorted(
            out_clinical.loc[out_clinical["gdc_primary_tumor_slide_count"].eq(0), "patient_id"].tolist()
        ),
        "slide_rows_with_cbioportal_label": int(slide_with_labels["has_cbioportal_label"].sum()),
    }
    args.out_audit_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.out_clinical_csv}")
    print(f"Wrote {args.out_slide_csv}")
    print(f"Wrote {args.out_audit_json}")


if __name__ == "__main__":
    main()
