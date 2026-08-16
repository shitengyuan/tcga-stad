#!/usr/bin/env python3
"""
Build a project-compatible TCGA-STAD clinical.csv from cBioPortal public data.

Source study:
  Stomach Adenocarcinoma (TCGA, Nature 2014), cBioPortal study id
  ``stad_tcga_pub``.

The output is suitable for the current training scripts. It is not guaranteed
to be byte-identical to the historical private ``clinical.csv`` that generated
the old OOF files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SUBTYPE_MAP = {
    "EBV": "STAD_EBV",
    "MSI": "STAD_MSI",
    "GS": "STAD_GS",
    "CIN": "STAD_CIN",
}


def immune_label(subtype: str) -> str:
    if subtype in {"STAD_EBV", "STAD_MSI"}:
        return "IMMUNE_SENSITIVE"
    if subtype in {"STAD_GS", "STAD_CIN"}:
        return "NON_SENSITIVE"
    return "UNKNOWN"


def first_existing(row, names, default=pd.NA):
    for name in names:
        if name in row and pd.notna(row[name]) and row[name] != "":
            return row[name]
    return default


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample_csv", type=Path, default=Path("external_downloads/tcga_stad/cbioportal_stad_tcga_pub_clinical_sample_wide.csv"))
    p.add_argument("--patient_csv", type=Path, default=Path("external_downloads/tcga_stad/cbioportal_stad_tcga_pub_clinical_patient_wide.csv"))
    p.add_argument("--out_csv", type=Path, default=Path("clinical.csv"))
    p.add_argument("--audit_json", type=Path, default=Path("external_downloads/tcga_stad/cbioportal_public_clinical_build_audit.json"))
    args = p.parse_args()

    sample = pd.read_csv(args.sample_csv)
    patient = pd.read_csv(args.patient_csv)
    df = sample.merge(patient, on="patient_id", how="left", suffixes=("_sample", "_patient"))

    out = pd.DataFrame()
    out["patient_id"] = df["patient_id"]
    out["sample_id"] = df.get("sample_id")
    out["slide_id"] = pd.NA
    out["subtype"] = df["MOLECULAR_SUBTYPE"].map(lambda x: SUBTYPE_MAP.get(str(x), pd.NA))
    out["label_immune_sensitive"] = out["subtype"].map(immune_label)
    out["age"] = pd.to_numeric(df.get("AGE"), errors="coerce")
    out["sex"] = df.get("SEX")
    out["histological_diagnosis"] = df.apply(
        lambda r: first_existing(r, ["WHO_CLASS", "LAUREN_CLASS", "CANCER_TYPE_DETAILED"]),
        axis=1,
    )
    out["ajcc_pathologic_tumor_stage"] = df.apply(
        lambda r: first_existing(r, ["TNMSTAGE", "PATH_T_STAGE"]),
        axis=1,
    )
    out["primary_site_patient"] = df.get("ANATOMIC_REGION")
    out["lymph_node_examined_count"] = pd.NA
    out["os_months"] = pd.to_numeric(df.get("OS_MONTHS"), errors="coerce")
    out["os_status"] = df.get("OS_STATUS")
    out["dfs_months"] = pd.to_numeric(df.get("DFS_MONTHS"), errors="coerce")
    out["dfs_status"] = df.get("DFS_STATUS")

    # Source audit columns retained for downstream review.
    out["source_study"] = "cbioportal:stad_tcga_pub"
    out["source_sample_id"] = df.get("sample_id")
    out["source_molecular_subtype"] = df.get("MOLECULAR_SUBTYPE")
    out["source_msi_status"] = df.get("MSI_STATUS")
    out["source_ebv_present"] = df.get("EBV_PRESENT")
    out["source_lauren_class"] = df.get("LAUREN_CLASS")
    out["source_tnmstage"] = df.get("TNMSTAGE")
    out["source_anatomic_region"] = df.get("ANATOMIC_REGION")
    out["source_path_t_stage"] = df.get("PATH_T_STAGE")
    out["source_path_n_stage"] = df.get("PATH_N_STAGE")
    out["source_path_m_stage"] = df.get("PATH_M_STAGE")
    out["source_mutation_count"] = df.get("MUTATION_COUNT")
    out["source_tmb_nonsynonymous"] = df.get("TMB_NONSYNONYMOUS")

    out = out.sort_values("patient_id")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    audit = {
        "source": "cBioPortal stad_tcga_pub public API",
        "sample_csv": str(args.sample_csv),
        "patient_csv": str(args.patient_csv),
        "out_csv": str(args.out_csv),
        "n_rows": int(len(out)),
        "subtype_counts": out["subtype"].value_counts(dropna=False).to_dict(),
        "immune_label_counts": out["label_immune_sensitive"].value_counts(dropna=False).to_dict(),
        "missing_rates": {c: float(out[c].isna().mean()) for c in out.columns},
        "important_limitations": [
            "slide_id is not present in cBioPortal clinical data and remains NA until WSI/feature filenames are joined.",
            "POLE subtype is not represented in cBioPortal stad_tcga_pub MOLECULAR_SUBTYPE.",
            "This file is public-source reconstructed clinical.csv, not the historical private clinical.csv used for old OOF training.",
        ],
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_csv} rows={len(out)}")
    print(f"Wrote {args.audit_json}")


if __name__ == "__main__":
    main()
