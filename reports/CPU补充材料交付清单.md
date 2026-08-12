# CPU补充材料交付清单

生成日期：2026-08-13  
项目目录：`/share/home/shitengyuan_lustre/medical/tcga-stad`  
基于版本：`f23ca86eea293e8094b0e874f66a84c5a8689830`

本次只补充当前CPU环境可以完成的材料，不重新提取WSI特征、不做GPU推理、不调用LLM、不重新训练模型。

## 一、本次新增文件

脚本：

- `generate_cpu_supplement.py`

输出目录：

- `results/cpu_supplement/`

主要输出：

- `errors_M1_immune_sensitive_oof.csv`
- `errors_M2_msi_oof.csv`
- `errors_M3_ebv_oof.csv`
- `errors_M4_subtype4_oof.csv`
- `errors_M5_clinical_oof.csv`
- `error_case_summary.json`
- `subgroup_metrics_by_site_and_subtype.csv`
- `subgroup_metrics_summary.json`
- `calibration_curve_data_oof.csv`
- `decision_curve_data_oof.csv`
- `calibration_dca_summary.json`
- `paired_model_comparison_M1_vs_M5.json`
- `agent_no_leakage_inputs_289.csv`
- `agent_no_leakage_output_schema.json`
- `agent_no_leakage_prompt.txt`
- `agent_no_leakage_package_summary.json`
- `model_registry_current.json`
- `cptac_errors_M1_immune_sensitive.csv`
- `cptac_errors_M2_msi.csv`
- `cptac_errors_M3_ebv.csv`
- `cptac_error_summary.json`
- `cpu_supplement_manifest.json`
- `README.md`

## 二、已补充内容

### 1. 289例内部OOF错例表

已为M1、M2、M3、M5生成患者级TP/TN/FP/FN清单；已为M4生成四分类正确/错误清单。

当前错误数：

- M1 immune-sensitive：40例错误，TN=198、TP=51、FN=22、FP=18
- M2 MSI：34例错误，TN=224、TP=31、FP=18、FN=16
- M3 EBV：22例错误，TN=261、FN=20、TP=6、FP=2
- M5 clinical：123例错误，TN=126、FP=90、TP=40、FN=33
- M4 subtype4：81例错误，正确208例

M4类别顺序继续固定为：

```text
prob_c0 = EBV
prob_c1 = MSI
prob_c2 = GS
prob_c3 = CIN
```

### 2. 亚组指标

已基于现有289例队列表生成：

- 按 `site` 的M1/M2/M3/M5指标
- 按 `M4_subtype` 的M1/M2/M3/M5指标

文件：`results/cpu_supplement/subgroup_metrics_by_site_and_subtype.csv`

限制：当前没有 `clinical.csv`，因此还不能补Lauren分型、部位、分期、切片质量等亚组。

### 3. 校准曲线和决策曲线原始数据

已生成：

- `calibration_curve_data_oof.csv`
- `decision_curve_data_oof.csv`

覆盖M1、M2、M3、M5。决策曲线阈值为0.01到0.99。

### 4. M1与M5配对比较

已基于289例患者级OOF做M1 vs M5的配对bootstrap AUC差异。

结果：

- M1 AUC：0.8986
- M5 AUC：0.6201
- AUC差值均值：0.2790
- bootstrap 95% CI：0.2024到0.3555
- bootstrap双侧p值：0.0

文件：`results/cpu_supplement/paired_model_comparison_M1_vs_M5.json`

说明：这里是患者级配对bootstrap，不是DeLong检验。

### 5. 无泄漏Agent输入包

已生成不含真实标签和真实分子亚型的Agent输入表：

- `agent_no_leakage_inputs_289.csv`

已移除：

- 真实MSI
- 真实EBV
- 真实TCGA subtype / M4 subtype
- label
- POLE

同时补充：

- `agent_no_leakage_prompt.txt`
- `agent_no_leakage_output_schema.json`
- `agent_no_leakage_package_summary.json`

当前规则：

- 当前panel没有视觉模块。
- 禁止输出未经图像支持的TIL、CIN、MSI形态学结论。
- Agent只能解释模型间一致性/冲突和建议验证项目。
- 最终预测固定来自M1 0.5阈值，Agent不得修改M1结果。

### 6. 当前模型登记表

已生成：`model_registry_current.json`

记录：

- 当前git commit
- M1-M4权重路径
- M1-M4权重sha256
- 任务类型和类别数
- M4类别顺序
- 当前缺失的正式模型材料

结论维持第一阶段判断：当前 `models/M1-M4.pt` 是训练脚本保存的每个任务最后一个fold checkpoint，不是完整每折权重登记，也不能证明是全数据重训权重。

### 7. CPTAC错例和排除汇总

已基于第一阶段CPTAC患者级结果生成固定阈值错例表：

- `cptac_errors_M1_immune_sensitive.csv`
- `cptac_errors_M2_msi.csv`
- `cptac_errors_M3_ebv.csv`
- `cptac_error_summary.json`

当前CPTAC统计：

- manifest：204张切片
- 已推理特征：193张切片
- 缺特征排除：11张切片
- 患者级预测：50例
- 有标签可评价：47例

CPTAC固定阈值0.5错误数：

- M1：TN=28、TP=13、FP=4、FN=2
- M2：TN=30、FP=9、TP=6、FN=2
- M3：TN=40、FN=5、TP=2

## 三、仍缺材料

这些内容不能仅靠当前CPU环境和已有CSV/JSON补齐：

- 内部TCGA原始 `clinical.csv`
- TCGA全部h5特征和真实患者-切片映射
- 原始CV fold落盘文件
- M1-M4每折正式权重、best epoch、随机种子登记、训练日志
- M5实际拟合pipeline对象
- M6生存模型权重、OOF风险表和评价脚本
- WSI缩略图、attention heatmap、cluster overlay、代表patch
- 真实无泄漏Agent全量重跑结果
- Agent重复运行稳定性结果
- cluster中心、代表patch、病理验证记录
- Lauren分型、部位、分期、切片质量等临床/病理亚组

## 四、重跑命令

```bash
cd /share/home/shitengyuan_lustre/medical/tcga-stad
/gpfsdata/home/shitengyuan/miniconda3/envs/gastric_msi_pathai/bin/python generate_cpu_supplement.py
```

