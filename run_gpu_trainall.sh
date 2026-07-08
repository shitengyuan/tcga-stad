#!/bin/bash
# 容器端 GPU 全量训练: M1-M4 + M6, 保留模型权重 + OOF 预测
# 在容器内执行: bash run_gpu_trainall.sh
set -e
cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-platcv/shitengyuan/TCGA-STAD

# 用绝对路径 python (base conda, 含 torch+h5py+lifelines), 避免 conda activate 在 non-interactive 失效
PY=/home/sankuai/conda/bin/python
export PYTHONUNBUFFERED=1
mkdir -p results models

echo "=========== GPU 全量训练开始 $(date) ==========="
echo "Python: $($PY --version)"
echo "设备: $($PY -c 'import torch; print(torch.cuda.get_device_name(0))')"

# M1-M4: 全量 patch (max_patches=999999), 30 epoch, 5fold×3repeat
for TASK in immune_sensitive msi ebv subtype4; do
    echo "=========== $TASK $(date) ==========="
    $PY -m src.train_multitask \
        --task $TASK --device cuda \
        --max_patches 999999 --epochs 30 --n_folds 5 --n_repeats 3 --n_boot 1000 \
        2>&1 | tee results/log_gpu_${TASK}.txt
done

# M6 生存: 全量 patch
echo "=========== survival $(date) ==========="
$PY -m src.train_survival \
    --device cuda --max_patches 999999 --epochs 20 --n_folds 5 \
    2>&1 | tee results/log_gpu_survival.txt

echo "=========== 全部完成 $(date) ==========="
echo "=== 汇总 ==="
for m in M1_immune_sensitive M2_msi M3_ebv M4_subtype4 M6_survival; do
    f=results/metrics_${m}.json
    [ -f "$f" ] && $PY -c "import json; d=json.load(open('$f')); print(f\"$m: AUC={d.get('oof_auc',d.get('oof_cindex','?')):.3f}\")"
done

echo "=========== GPU 全量训练开始 $(date) ==========="
echo "设备: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"

# M1-M4: 全量 patch (max_patches=999999), 30 epoch, 5fold×3repeat
for TASK in immune_sensitive msi ebv subtype4; do
    echo "=========== $TASK $(date) ==========="
    python -m src.train_multitask \
        --task $TASK --device cuda \
        --max_patches 999999 --epochs 30 --n_folds 5 --n_repeats 3 --n_boot 1000 \
        2>&1 | tee results/log_gpu_${TASK}.txt
done

# M6 生存: 全量 patch
echo "=========== survival $(date) ==========="
python -m src.train_survival \
    --device cuda --max_patches 999999 --epochs 20 --n_folds 5 \
    2>&1 | tee results/log_gpu_survival.txt

echo "=========== 全部完成 $(date) ==========="
echo "=== 汇总 ==="
for m in M1_immune_sensitive M2_msi M3_ebv M4_subtype4 M6_survival; do
    f=results/metrics_${m}.json
    [ -f "$f" ] && python -c "import json; d=json.load(open('$f')); print(f\"$m: AUC={d.get('oof_auc',d.get('oof_cindex','?')):.3f}\")"
done
