"""Generate descriptive baseline summaries from frozen cohort files.

This script does not modify any locked analysis file. It reports availability
and descriptive values after patient-level ID joins used by the manuscript.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def median_iqr(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "n_available": int(values.size),
        "median": round(float(values.median()), 2),
        "q1": round(float(values.quantile(0.25)), 2),
        "q3": round(float(values.quantile(0.75)), 2),
    }


def count_percent(mask: pd.Series, denominator: int) -> dict[str, float | int]:
    count = int(mask.fillna(False).sum())
    return {"n": count, "percent": round(100 * count / denominator, 1)}


def stage_group(value: object) -> str | None:
    text = str(value).upper()
    match = re.search(r"(?:STAGE[_\s-]*)?(IV|III|II|I)(?:[ABC])?", text)
    return match.group(1) if match else None


def stages(series: pd.Series, denominator: int) -> dict[str, object]:
    grouped = series.map(stage_group)
    counts = {stage: int((grouped == stage).sum()) for stage in ("I", "II", "III", "IV")}
    return {
        "n_available": int(grouped.notna().sum()),
        "counts": counts,
        "percents": {stage: round(100 * count / denominator, 1) for stage, count in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcga-manifest", type=Path, required=True)
    parser.add_argument("--tcga-clinical", type=Path, required=True)
    parser.add_argument("--cptac-predictions", type=Path, required=True)
    parser.add_argument("--cptac-clinical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tcga_manifest = pd.read_csv(args.tcga_manifest)
    tcga_clinical = pd.read_csv(args.tcga_clinical)
    cptac_predictions = pd.read_csv(args.cptac_predictions)
    cptac_clinical = pd.read_csv(args.cptac_clinical)

    tcga = tcga_manifest.merge(tcga_clinical, on="patient_id", how="left", validate="one_to_one")
    cptac = cptac_predictions.merge(
        cptac_clinical,
        left_on="patient_id",
        right_on="case_submitter_id",
        how="left",
        validate="one_to_one",
    )

    def summarise_tcga(frame: pd.DataFrame) -> dict[str, object]:
        n = len(frame)
        return {
            "n": n,
            "clinical_match_n": int(frame["age"].notna().sum()),
            "age_years": median_iqr(frame["age"]),
            "female": count_percent(frame["sex"].str.lower().eq("female"), n),
            "lauren_intestinal": count_percent(frame["source_lauren_class"].str.lower().eq("intestinal"), n),
            "pathologic_stage": stages(frame["ajcc_pathologic_tumor_stage"], n),
        }

    def summarise_cptac(frame: pd.DataFrame) -> dict[str, object]:
        n = len(frame)
        return {
            "n": n,
            "clinical_match_n": int(frame["age_at_index"].notna().sum()),
            "age_years": median_iqr(frame["age_at_index"]),
            "female": count_percent(frame["gender"].str.lower().eq("female"), n),
            "lauren_intestinal": count_percent(frame["lauren_label"].str.lower().eq("intestinal"), n),
            "pathologic_stage": stages(frame["ajcc_pathologic_stage"], n),
        }

    summary = {"TCGA_246": summarise_tcga(tcga), "CPTAC_156": summarise_cptac(cptac)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
