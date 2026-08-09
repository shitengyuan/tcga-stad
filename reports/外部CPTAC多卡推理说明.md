# CPTAC external multi-GPU inference

## Environment

Required Python packages:

```bash
pip install torch torchvision timm openslide-python pandas numpy pillow tqdm scikit-learn
```

System library is also required:

```bash
# Ubuntu/Debian
apt-get install -y openslide-tools

# Conda alternative
conda install -c conda-forge openslide
```

## Smoke test

Run two slides with few sampled patches first:

```bash
cd /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/tcga-stad
torchrun --standalone --nproc_per_node=1 run_cptac_external_multigpu.py \
  --svs_dir /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/dataset/cptac-stad-histopathology \
  --uni_weights /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/uni2-h-weights/pytorch_model.bin \
  --model_dir models \
  --out_dir results/external_cptac_smoke \
  --max_slides 2 \
  --max_patches 128 \
  --batch_size 16
```

## Formal inference

Example for 4 GPUs:

```bash
cd /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/tcga-stad
torchrun --standalone --nproc_per_node=4 run_cptac_external_multigpu.py \
  --svs_dir /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/dataset/cptac-stad-histopathology \
  --uni_weights /gpfsdata/home/shitengyuan/shitengyuan_lustre/medical/uni2-h-weights/pytorch_model.bin \
  --model_dir models \
  --out_dir results/external_cptac \
  --max_patches 8192 \
  --batch_size 64
```

Outputs:

- `external_cptac_slide_predictions.csv`: one row per SVS.
- `external_cptac_patient_predictions.csv`: patient-level mean aggregation.
- `errors.rank*.json`: per-rank slide failures, if any.

If CPTAC labels are ready, prepare a CSV with `patient_id` and optional binary columns
`immune_sensitive`, `msi`, `ebv`, then add:

```bash
  --labels_csv /path/to/cptac_labels.csv
```

The script will write `external_cptac_metrics.json`.
