# CPU Supplement Materials

Generated from existing CSV/JSON outputs only. No GPU, WSI loading, feature extraction, or LLM calls were used.

## Newly Added

- `errors_*_oof.csv`: TCGA OOF TP/TN/FP/FN and M4 misclassification lists.
- `error_case_summary.json`: error counts by model.
- `subgroup_metrics_by_site_and_subtype.csv`: subgroup metrics by TCGA site and molecular subtype.
- `calibration_curve_data_oof.csv`: calibration curve raw data.
- `decision_curve_data_oof.csv`: decision curve raw data.
- `paired_model_comparison_M1_vs_M5.json`: paired bootstrap comparison of M1 vs M5 AUC.
- `agent_no_leakage_inputs_289.csv`: Agent input table with true labels/subtypes removed.
- `agent_no_leakage_output_schema.json` and `agent_no_leakage_prompt.txt`: no-leakage Agent contract.
- `model_registry_current.json`: current model artifact registry.
- `cptac_errors_*.csv` and `cptac_error_summary.json`: CPTAC fixed-threshold error lists.

## Still Missing

- `clinical.csv`, TCGA h5 features, original fold files, per-fold weights, training logs.
- WSI thumbnails, attention heatmaps, cluster overlays, representative patches.
- Formal no-leakage LLM rerun outputs. This package only prepares safe inputs/prompt/schema.

## Re-run

```bash
cd /share/home/shitengyuan_lustre/medical/tcga-stad
/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python generate_cpu_supplement.py
```
