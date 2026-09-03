# 计算机改进部分逐项完成进展

生成时间：2026-09-02T19:34:55  
项目目录：`/share/home/shitengyuan_lustre/medical/tcga-stad`  
输出目录：`/share/home/shitengyuan_lustre/medical/tcga-stad/results/locked_release_20260902`  
Git HEAD：`288fcea34294880d4e29831df398cce6bdc34176`  

## 已补齐的核心交付

1. 已冻结 TCGA 246 例主队列、231 例中心隔离 OOF、CPTAC 156 例外部验证队列。
2. 已按患者级 bootstrap 重算 TCGA 231 和 CPTAC 156 的 AUROC、AP、阈值指标、Brier、校准截距/斜率 CI。
3. 已生成 ROC/PR、校准曲线、临床检测容量曲线。
4. 已按同一 231 例和同一 fold 重算 Clinical-only，并生成图像+临床 late-fusion/stacking 探索版。
5. 已生成 UNI2-h mean pooling 和 max pooling 最小基线。
6. 已生成错误病例和低置信病例的病理审阅模板。

## 当前主线指标摘要

TCGA 231 OOF M1：AUROC=0.905，AP=0.839，Sensitivity=0.677，Specificity=0.941。  
CPTAC 156 external M1：AUROC=0.888，AP=0.812，Sensitivity=0.860，Specificity=0.885。  
Matched Clinical-only 231：AUROC=0.619，AP=0.335。  
Image+clinical late-fusion 231：AUROC=0.835，AP=0.709。  
UNI2-h mean pooling M1：AUROC=0.886，AP=0.785。  
UNI2-h max pooling M1：AUROC=0.796，AP=0.645。  


## 仍不能自动交付的内容

第二编码器、Transformer aggregator 需要额外特征或GPU训练；病理形态学解释需要病理医生基于模板盲评，不能由当前脚本替代。late-fusion 文件只能作为探索性图像分数+临床融合，不等同于端到端图像特征融合网络。
