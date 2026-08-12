# First-stage Audit Package

Generated from commit `59bea3917395ca83bbbfb3771ea97ae954bdd9f1`.

## Files
- `final_289_patient_cohort.csv`: reconstructed TCGA 289-patient OOF cohort.
- `cv_folds_reconstructed.csv`: reconstructed folds using original seed/policy.
- `cv_fold_reconstruction_summary.json`: leakage checks for patient/site folds.
- `metrics_M1-M5_oof_recomputed.json`: OOF metrics recomputed from the 289 rows.
- `m4_class_mapping_audit.*`: confirms M4 class order.
- `label_sources_and_mapping.json`: label source and mapping notes.
- `agent_leakage_audit_and_no_leakage_plan.json`: current Agent caveats.
- `cptac_*`: external CPTAC cohort, predictions, exclusions and fixed-threshold metrics.
- `environment_minimal.yml` and `reproducibility_manifest.json`: minimal reproducibility metadata.
- `../../build_tcga_label_table.py`: reusable script to regenerate TCGA label/cohort tables after `clinical.csv` is restored.
- `../../reports/第一阶段审计交付清单.md`: Chinese delivery checklist summarizing available and missing materials.

## Key Caveats
- `clinical.csv` and internal TCGA feature h5 files are absent from this checkout, so `slide_id`, `POLE`, and original TCGA raw label fields cannot be fully restored here.
- CV folds are reconstructed from current 289 OOF patients. The original fold files were not saved.
- M1-M4 `models/*.pt` are saved by the training script as the last fold model per task, not per-fold weights and not proven full-data retraining weights.
- Existing Agent panel output is development-only because the prompt included true subtype and the visual module was disabled.

## M4 Mapping
Class order: `{'0': 'EBV', '1': 'MSI', '2': 'GS', '3': 'CIN'}`. Therefore `prob_c1` means MSI.

## Fold Check
Same patient crosses validation folds within a repeat: `False`.
Same site crosses validation folds within a repeat: `False`.

## Recompute Commands
```bash
cd /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/tcga-stad
/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python audit_first_stage.py --device cpu

# After restoring clinical.csv, regenerate the raw TCGA label/cohort table:
/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python build_tcga_label_table.py \
  --clinical_csv clinical.csv \
  --feature_dir tcga_stad_uni2h/TCGA-STAD/features
```
