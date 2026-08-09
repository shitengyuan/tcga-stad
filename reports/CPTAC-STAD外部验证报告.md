# CPTAC-STAD 外部验证报告

日期：2026-08-09  
训练队列：TCGA-STAD  
外部验证队列：CPTAC-STAD  
特征目录：`results/external_cptac_features_20x256`  
推理目录：`results/external_cptac_feature_infer_20x256_4gpu`

## 1. 数据与标签

本报告基于当前已经完成 UNI2-h 特征提取并完成模型推理的 CPTAC-STAD 病理切片结果，评估 TCGA-STAD 训练模型在 CPTAC-STAD 队列上的外部泛化表现。

输入结果：

| 文件 | 路径 |
|---|---|
| 患者级预测 | `results/external_cptac_feature_infer_20x256_4gpu/external_feature_patient_predictions.csv` |
| 切片级预测 | `results/external_cptac_feature_infer_20x256_4gpu/external_feature_slide_predictions.csv` |
| 推理错误记录 | `results/external_cptac_feature_infer_20x256_4gpu/external_feature_errors.json` |

标签来自 2026 年 CPTAC-STAD 主论文补充表：

| 文件 | 路径 |
|---|---|
| 原始补充表 | `../dataset/cptac-stad-histopathology/labels/cptac_stad_2026_supplement/1-s2.0-S2666379126001734-mmc2.xlsx` |
| QC-pass 四分型标签 | `../dataset/cptac-stad-histopathology/labels/cptac_stad_2026_tcga_subtype_labels_qc_pass.csv` |

使用的标签字段：

| 字段 | 含义 |
|---|---|
| `Genomic_subtype` | TCGA 四分型，`EBV_Pos` 映射为 `EBV`，`MSI_High` 映射为 `MSI` |
| `immune_sensitive` | `EBV` 或 `MSI` 为阳性，`GS` 或 `CIN` 为阴性 |
| `msi` | MSI vs non-MSI |
| `ebv` | EBV vs non-EBV |

当前覆盖情况：

| 项目 | 数量 |
|---|---:|
| 已推理患者 | 45 |
| 已推理切片 | 174 |
| 推理错误 | 0 |
| CPTAC-STAD QC-pass 有标签患者 | 157 |
| 预测与 QC-pass 标签均存在的可评估患者 | 42 |

有 3 个已推理患者没有 QC-pass 分子分型标签，因此未纳入监督评估：

```text
C3L-05521
C3L-06136
C3L-06767
```

42 个可评估患者的标签分布：

| 分型 | 患者数 |
|---|---:|
| EBV | 7 |
| MSI | 7 |
| GS | 13 |
| CIN | 15 |

## 2. 默认阈值外部验证结果

二分类任务默认阈值为 `0.5`。这是最接近真实外部验证的结果，因为阈值没有在 CPTAC-STAD 上重新优化。

| 任务 | n | 阳性数 | AUC | AP | Accuracy | Balanced Accuracy | F1 | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EBV/MSI vs 其他 | 42 | 14 | 0.883 | 0.850 | 0.905 | 0.911 | 0.867 | 0.812 | 0.929 | 0.893 |
| MSI | 42 | 7 | 0.800 | 0.592 | 0.786 | 0.814 | 0.571 | 0.429 | 0.857 | 0.771 |
| EBV | 42 | 7 | 0.988 | 0.957 | 0.881 | 0.643 | 0.444 | 1.000 | 0.286 | 1.000 |
| 四分类 EBV/MSI/GS/CIN | 42 | - | 0.766 macro OvR | - | 0.500 | - | 0.449 macro | - | - | - |

默认阈值混淆矩阵如下，行是真实标签，列是预测标签。

EBV/MSI vs 其他：

```text
[[25, 3],
 [ 1,13]]
```

MSI：

```text
[[27, 8],
 [ 1, 6]]
```

EBV：

```text
[[35, 0],
 [ 5, 2]]
```

四分类类别顺序为 `[EBV, MSI, GS, CIN]`：

```text
[[ 1, 3, 0, 3],
 [ 0, 5, 1, 1],
 [ 0, 2, 4, 7],
 [ 0, 0, 4,11]]
```

默认阈值下的主要发现：

- EBV/MSI 联合终点在 CPTAC-STAD 上表现最好，`AUC=0.883`，`balanced accuracy=0.911`，`F1=0.867`，说明 TCGA-STAD 训练得到的免疫敏感型病理信号具有较好的外部泛化能力。
- MSI 单独预测有一定泛化能力，`AUC=0.800`，但默认阈值下假阳性偏多，precision 只有 `0.429`。
- EBV 单独预测的排序能力很强，`AUC=0.988`、`AP=0.957`，但默认阈值 `0.5` 明显过高，只检出 2/7 个 EBV 患者，recall 为 `0.286`。
- 直接四分类表现不足，`accuracy=0.500`，`macro-F1=0.449`，不建议把直接四分类头作为当前外部验证的主要结论。

## 3. 阈值调整后的结果

下面结果是在当前 42 个带标签 CPTAC-STAD 患者上选择阈值得到的，适合用于模型校准分析。由于阈值使用了外部验证集标签进行优化，因此应作为探索性分析或校准后结果报告，不能替代默认阈值外部验证结果。

按 balanced accuracy 最优选择阈值：

| 任务 | 调整后阈值 | Accuracy | Balanced Accuracy | F1 | Precision | Recall | Specificity | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EBV/MSI vs 其他 | 0.494 | 0.905 | 0.911 | 0.867 | 0.812 | 0.929 | 0.893 | `[[25,3],[1,13]]` |
| MSI | 0.521 | 0.810 | 0.829 | 0.600 | 0.462 | 0.857 | 0.800 | `[[28,7],[1,6]]` |
| EBV | 0.088 | 0.929 | 0.957 | 0.824 | 0.700 | 1.000 | 0.914 | `[[32,3],[0,7]]` |

EBV 任务另有一个 F1 最优阈值：

| 任务 | 阈值 | Accuracy | Balanced Accuracy | F1 | Precision | Recall | Specificity | 混淆矩阵 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| EBV | 0.155 | 0.976 | 0.929 | 0.923 | 1.000 | 0.857 | 1.000 | `[[35,0],[1,6]]` |

阈值调整后的新结论：

- EBV/MSI 联合终点几乎不需要重新校准，最优阈值 `0.494` 与默认阈值 `0.5` 基本一致，性能不变。这说明该联合二分类模型的外部概率校准相对稳定。
- MSI 阈值从 `0.5` 调整到 `0.521` 后，balanced accuracy 从 `0.814` 提升到 `0.829`，但提升幅度有限；MSI 的主要问题仍是假阳性较多。
- EBV 是最需要重新校准的任务。默认阈值下 EBV recall 只有 `0.286`；当阈值降到 `0.088` 后，recall 达到 `1.000`，specificity 仍有 `0.914`。
- 如果研究目标是筛查 EBV 患者，建议使用 EBV 阈值 `0.088`，优先保证不漏诊。
- 如果研究目标是获得高置信度 EBV 阳性病例，建议使用 EBV 阈值 `0.155`，此时 precision 为 `1.000`，recall 为 `0.857`，F1 为 `0.923`。

## 4. 分层四分类策略

考虑到 EBV 和 MSI 的二分类头强于直接四分类头，进一步测试了一个分层规则：

```text
if ebv_prob >= 0.087825:
    predict EBV
elif msi_prob >= 0.520582:
    predict MSI
else:
    predict argmax(GS, CIN) using the M4 four-class head
```

结果如下：

| 方法 | Accuracy | Macro F1 |
|---|---:|---:|
| 直接四分类头 | 0.500 | 0.449 |
| 分层阈值策略 | 0.571 | 0.580 |

分层策略混淆矩阵，类别顺序为 `[EBV, MSI, GS, CIN]`：

```text
[[7,0,0,0],
 [2,4,0,1],
 [0,3,4,6],
 [1,1,4,9]]
```

分层策略各类别指标：

| 类别 | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| EBV | 7 | 0.700 | 1.000 | 0.824 |
| MSI | 7 | 0.500 | 0.571 | 0.533 |
| GS | 13 | 0.500 | 0.308 | 0.381 |
| CIN | 15 | 0.562 | 0.600 | 0.581 |

分层四分类结论：

- 分层阈值策略优于直接四分类头，macro-F1 从 `0.449` 提升到 `0.580`。
- 主要收益来自 EBV：直接四分类只正确识别 1/7 个 EBV 患者，分层策略可以识别 7/7 个 EBV 患者。
- GS 与 CIN 仍然是主要混淆来源，说明当前模型对非 EBV/MSI 的组织学-分子亚型区分能力仍有限。
- 对当前数据，更合理的使用方式是先报告 EBV/MSI 联合终点，再报告 EBV、MSI 二分类，四分类只作为探索性分析。

## 5. 图表与结果文件

主要图表：

| 图表 | 路径 |
|---|---|
| 预测概率分布 | [fig1_probability_histograms.png](../results/external_cptac_feature_infer_20x256_4gpu/figures/fig1_probability_histograms.png) |
| 患者预测计数 | [fig2_patient_prediction_counts.png](../results/external_cptac_feature_infer_20x256_4gpu/figures/fig2_patient_prediction_counts.png) |
| 患者概率热图 | [fig3_patient_probability_heatmap.png](../results/external_cptac_feature_infer_20x256_4gpu/figures/fig3_patient_probability_heatmap.png) |
| 切片-患者一致性 | [fig4_slide_patient_consistency.png](../results/external_cptac_feature_infer_20x256_4gpu/figures/fig4_slide_patient_consistency.png) |
| 覆盖度分布 | [fig5_coverage_histograms.png](../results/external_cptac_feature_infer_20x256_4gpu/figures/fig5_coverage_histograms.png) |
| 默认阈值 ROC/PR | [fig6_supervised_roc_pr_qc_labels.png](../results/external_cptac_feature_infer_20x256_4gpu/figures/fig6_supervised_roc_pr_qc_labels.png) |
| 默认阈值混淆矩阵 | [fig7_supervised_confusion_matrices_qc_labels.png](../results/external_cptac_feature_infer_20x256_4gpu/figures/fig7_supervised_confusion_matrices_qc_labels.png) |
| 按真实标签的概率分布 | [fig8_supervised_probability_by_label_qc_labels.png](../results/external_cptac_feature_infer_20x256_4gpu/figures/fig8_supervised_probability_by_label_qc_labels.png) |
| 阈值扫描曲线 | [fig9_threshold_sweeps_qc_labels.png](../results/external_cptac_feature_infer_20x256_4gpu/figures/fig9_threshold_sweeps_qc_labels.png) |
| 调整阈值后混淆矩阵 | [fig10_threshold_tuned_confusion_matrices_qc_labels.png](../results/external_cptac_feature_infer_20x256_4gpu/figures/fig10_threshold_tuned_confusion_matrices_qc_labels.png) |

主要表格：

| 表格 | 路径 |
|---|---|
| 默认阈值监督评估指标 | [cptac_supervised_metrics_qc_labels.csv](../results/external_cptac_feature_infer_20x256_4gpu/figures/cptac_supervised_metrics_qc_labels.csv) |
| 调整阈值后评估指标 | [cptac_threshold_tuned_metrics_qc_labels.csv](../results/external_cptac_feature_infer_20x256_4gpu/figures/cptac_threshold_tuned_metrics_qc_labels.csv) |
| 患者级预测与 QC 标签合并表 | [cptac_patient_predictions_with_qc_labels.csv](../results/external_cptac_feature_infer_20x256_4gpu/figures/cptac_patient_predictions_with_qc_labels.csv) |
| 患者级预测与调整阈值后标签 | [cptac_patient_predictions_with_qc_labels_threshold_tuned.csv](../results/external_cptac_feature_infer_20x256_4gpu/figures/cptac_patient_predictions_with_qc_labels_threshold_tuned.csv) |

## 6. 总结

当前结果支持以下结论：

1. TCGA-STAD 训练模型在 CPTAC-STAD 上对 EBV/MSI 联合终点具有较好的外部泛化能力。默认阈值下，42 个可评估患者的 `AUC=0.883`，`balanced accuracy=0.911`，`F1=0.867`。
2. EBV 单独任务的判别排序能力很强，但存在明显跨队列阈值漂移。默认阈值 `0.5` 漏检较多，经过阈值校准后性能显著改善。
3. MSI 单独任务有中等泛化能力，但假阳性仍较多，后续需要更多样本和更稳定的校准策略。
4. 直接四分类 EBV/MSI/GS/CIN 当前不够稳健，不宜作为主要外部验证终点。
5. 分层策略比直接四分类更合理：先用 EBV、MSI 二分类头识别可分性较强的亚型，再在剩余病例中区分 GS/CIN，可以把四分类 macro-F1 从 `0.449` 提高到 `0.580`。

需要注意的是，目前只评估了 42 个已经完成特征提取并成功匹配 QC-pass 标签的患者。最终外部验证应在全部可用 CPTAC-STAD 肿瘤 WSI 完成特征提取后重新统计，并建议补充患者级 bootstrap 95% CI。
