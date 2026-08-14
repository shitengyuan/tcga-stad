# GPU交付脚本说明

脚本：`run_gpu_deliverables.sh`

用途：在有GPU的环境中，一次性生成CPTAC外部验证相关的GPU依赖材料，并刷新后续审计表、CPU补充表和轻量交付包。

## 默认执行内容

默认命令：

```bash
cd /share/home/shitengyuan_lustre/medical/tcga-stad
bash run_gpu_deliverables.sh
```

默认会执行：

1. 检查Python、PyTorch、CUDA、GPU数量、UNI2-h权重、ABMIL分类头和CPTAC SVS目录。
2. 使用4卡 `torchrun` 提取CPTAC UNI2-h特征。
3. 使用4卡对已保存特征进行M1-M4推理。
4. 生成CPTAC患者级/切片级预测、图表和有标签指标。
5. 刷新 `results/audit_first_stage/`。
6. 刷新 `results/cpu_supplement/`。
7. 打包轻量交付材料到 `results/gpu_deliverables/package/`。

## 默认输出

特征目录：

- `results/external_cptac_features_20x256/`

外部验证推理目录：

- `results/external_cptac_feature_infer_20x256_4gpu/`

审计目录：

- `results/audit_first_stage/`

脚本在刷新审计前，会把 `INFER_DIR` 里的 `external_feature_slide_predictions.csv`、`external_feature_patient_predictions.csv` 和 `external_feature_errors.json` 同步到 `results/audit_first_stage/cptac_193_feature_inference/`，确保审计表使用最新CPTAC推理结果。

CPU补充目录：

- `results/cpu_supplement/`

运行日志和打包目录：

- `results/gpu_deliverables/logs/`
- `results/gpu_deliverables/gpu_deliverables_run_config.json`
- `results/gpu_deliverables/package/`

## 重要参数

默认4卡：

```bash
NUM_GPUS=4 bash run_gpu_deliverables.sh
```

默认每张切片最多8192个patch，和当前已有外部验证设置一致：

```bash
MAX_PATCHES=8192 bash run_gpu_deliverables.sh
```

如果中途在特征提取末尾遇到 `dist.barrier()`、`NCCL communicator`、`TCPStore Socket Timeout` 一类错误，原因通常是不同GPU处理切片耗时差异太大，先完成的rank在同步点等待超时。当前版本已经取消特征提取阶段的分布式barrier，改为各rank独立写文件，`torchrun` 全部结束后由总控脚本合并 `feature_manifest.rank*.csv`。

中断后直接重跑即可，已有完整 `.pt/.h5` 特征默认会跳过：

```bash
bash run_gpu_deliverables.sh
```

如果要按“全部采样组织patch”正式重提特征：

```bash
MAX_PATCHES=0 bash run_gpu_deliverables.sh
```

只跑推理，不重新提特征：

```bash
RUN_CPTAC_EXTRACT=0 bash run_gpu_deliverables.sh
```

只重画图、刷新审计和打包：

```bash
RUN_CPTAC_EXTRACT=0 RUN_CPTAC_INFER=0 bash run_gpu_deliverables.sh
```

把特征文件本身也放进tar包。默认不打包 `.pt/.h5`，因为体积可能很大：

```bash
PACKAGE_FEATURES=1 bash run_gpu_deliverables.sh
```

## 路径覆盖

如果在其他机器跑，可以显式指定路径：

```bash
ROOT=/path/to/tcga-stad \
PY=/path/to/env/bin/python \
SVS_DIR=/path/to/cptac-stad-histopathology \
UNI_WEIGHTS=/path/to/uni2-h-weights/pytorch_model.bin \
MODEL_DIR=/path/to/models \
NUM_GPUS=4 \
bash run_gpu_deliverables.sh
```

## 内部模型重训

默认不重训内部M1-M4/M6，因为当前训练脚本会覆盖：

- `models/M1_immune_sensitive.pt`
- `models/M2_msi.pt`
- `models/M3_ebv.pt`
- `models/M4_subtype4.pt`
- `results/oof_preds_*.csv`
- `results/metrics_*.json`

而且当前 `src/train_multitask.py` 只保存每个任务最后一个fold checkpoint，不保存每折正式权重。

如果确认要按当前代码重训并允许覆盖：

```bash
RUN_INTERNAL_TRAIN=1 CONFIRM_OVERWRITE_MODELS=1 bash run_gpu_deliverables.sh
```

重训还需要当前目录存在：

- `clinical.csv`
- `tcga_stad_uni2h/TCGA-STAD/features/`

## 运行后应该检查的文件

关键预测：

- `results/external_cptac_feature_infer_20x256_4gpu/external_feature_slide_predictions.csv`
- `results/external_cptac_feature_infer_20x256_4gpu/external_feature_patient_predictions.csv`
- `results/external_cptac_feature_infer_20x256_4gpu/external_feature_errors.json`

关键审计：

- `results/audit_first_stage/cptac_slide_cohort_and_exclusions.csv`
- `results/audit_first_stage/cptac_patient_cohort_labels_predictions_fixed_threshold.csv`
- `results/audit_first_stage/cptac_fixed_threshold_metrics_bootstrap.json`
- `results/cpu_supplement/cptac_error_summary.json`

日志：

- `results/gpu_deliverables/logs/00_preflight.log`
- `results/gpu_deliverables/logs/01_extract_cptac_uni2h_20x256.log`
- `results/gpu_deliverables/logs/02_infer_cptac_features_4gpu.log`
- `results/gpu_deliverables/logs/03_plot_cptac_feature_results.log`
- `results/gpu_deliverables/logs/04_audit_first_stage.log`
- `results/gpu_deliverables/logs/05_generate_cpu_supplement.log`

## 脚本不能自动补齐的材料

这些材料需要原始数据、代码改造或人工/LLM复核，不能仅靠当前GPU脚本自动得到：

- 内部TCGA原始 `clinical.csv`
- 原始CV fold落盘文件
- M1-M4每折正式权重、best epoch和完整训练日志
- M5实际拟合pipeline对象
- M6完整可复现实验登记
- WSI缩略图、attention heatmap、cluster overlay和代表patch证据包
- 真实无泄漏Agent全量重跑结果
- cluster命名的病理医师验证记录
