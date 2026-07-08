"""
feature_loader.py
══════════════════
加载 UNI2-h patch 特征 (h5) 并与 clinical.csv 对齐。

h5 结构 (per slide):
  features:        (1, N, 1536) float32  — UNI2-h patch 特征
  coords_patching: (N, 2) int64           — patch 坐标 (去掉外层 batch 维)

用法:
  from src.feature_loader import FeatureLoader
  fl = FeatureLoader(feature_dir, clinical_csv)
  dataset = fl.build_dataset(label_filter=True)  # 仅保留可建模样本
  # dataset[pid] = {"features": (N,1536) float32, "coords": (N,2) int64,
  #                 "label": 0/1, "subtype": str, ...}
"""
from __future__ import annotations
import os
import csv
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd


class FeatureLoader:
    """h5 特征加载 + clinical.csv 对齐。

    Parameters
    ----------
    feature_dir : str
        含 *.h5 的目录 (tcga_stad_uni2h/TCGA-STAD/features)
    clinical_csv : str
        clinical.csv 路径
    """

    FEAT_DIM = 1536

    def __init__(self, feature_dir: str, clinical_csv: str):
        self.feature_dir = Path(feature_dir)
        self.clinical = pd.read_csv(clinical_csv).set_index("patient_id")
        # slide_id -> h5 路径
        self.slide_paths = {f.stem: f for f in self.feature_dir.glob("*.h5")}

    # ── 公开接口 ─────────────────────────────────────────────

    def list_slides(self) -> List[str]:
        return sorted(self.slide_paths.keys())

    def load_slide(self, slide_id: str) -> Dict[str, np.ndarray]:
        """加载单个 slide: features (N,1536), coords (N,2)。"""
        path = self.slide_paths[slide_id]
        try:
            with h5py.File(path, "r") as h:
                feat = h["features"][0].astype(np.float32)        # (N,1536)
                coords = h["coords_patching"][:].astype(np.int64)  # (N,2)
        except Exception as e:
            logger.warning(f"slide {slide_id} 加载失败({e}), 返回空")
            return {"features": np.zeros((1, self.FEAT_DIM), dtype=np.float32),
                    "coords": np.zeros((1, 2), dtype=np.int64)}
        return {"features": feat, "coords": coords}

    def build_dataset(
        self,
        task: str = "immune_sensitive",
        max_patches: Optional[int] = None,
        seed: int = 42,
    ) -> Dict[str, dict]:
        """构建 patient-level 数据集。

        Parameters
        ----------
        task : str
            "immune_sensitive" : MSI∪EBV vs 非 (排除POLE/NO_SUBTYPE), 二分类
            "msi"              : MSI-H vs MSS, 二分类
            "ebv"              : EBV+ vs EBV-, 二分类
            "subtype4"         : EBV/MSI/GS/CIN 四分类
        max_patches : int | None
            每个 slide 最多保留的 patch 数
        """
        rng = np.random.default_rng(seed)
        dataset = {}
        for pid, row in self.clinical.iterrows():
            subtype = str(row["subtype"])
            label, keep = self._make_label(task, subtype,
                                           row.get("label_immune_sensitive", ""))
            if not keep:
                continue
            slide_ids = [s for s in str(row["slide_id"]).split(";") if s]
            slide_ids = [s for s in slide_ids if s in self.slide_paths]
            if not slide_ids:
                continue
            data = self.load_slide(slide_ids[0])
            feat, coords = data["features"], data["coords"]
            if max_patches is not None and feat.shape[0] > max_patches:
                idx = rng.choice(feat.shape[0], max_patches, replace=False)
                idx.sort()
                feat, coords = feat[idx], coords[idx]
            dataset[pid] = {
                "features": feat,
                "coords": coords,
                "label": label,
                "subtype": subtype,
                "slide_id": slide_ids[0],
                "n_patches": feat.shape[0],
            }
        return dataset

    @staticmethod
    def _make_label(task: str, subtype: str, immune_label: str):
        """根据任务构造标签, 返回 (label, keep)。"""
        if task == "immune_sensitive":
            if immune_label == "IMMUNE_SENSITIVE": return 1, True
            if immune_label == "NON_SENSITIVE": return 0, True
            return None, False   # POLE/NO_SUBTYPE 排除
        elif task == "msi":
            if subtype == "STAD_MSI": return 1, True
            if subtype in ("STAD_GS", "STAD_CIN", "STAD_EBV"): return 0, True
            return None, False   # POLE/NA 排除 (EBV 是 MSS, 归入 MSS 对照)
        elif task == "ebv":
            if subtype == "STAD_EBV": return 1, True
            if subtype in ("STAD_MSI", "STAD_GS", "STAD_CIN"): return 0, True
            return None, False
        elif task == "subtype4":
            mapping = {"STAD_EBV": 0, "STAD_MSI": 1, "STAD_GS": 2, "STAD_CIN": 3}
            if subtype in mapping: return mapping[subtype], True
            return None, False
        raise ValueError(f"未知任务: {task}")


    def get_site(self, pid: str) -> str:
        """TCGA barcode 第 2 段 = tissue source site (用于 site-stratified CV)。"""
        return pid.split("-")[1] if "-" in pid else "UNK"

    # ── 统计 ─────────────────────────────────────────────────

    def summary(self, dataset: Dict[str, dict]) -> str:
        import numpy as np
        from collections import Counter
        n = len(dataset)
        labels = Counter(v["label"] for v in dataset.values())
        patch_counts = np.array([v["n_patches"] for v in dataset.values()])
        sites = len(set(self.get_site(p) for p in dataset))
        lines = [
            f"样本数: {n}  (标签分布: {dict(labels)})",
            f"patch 数: 中位 {int(np.median(patch_counts))}, "
            f"范围 [{patch_counts.min()}, {patch_counts.max()}], "
            f"总 {patch_counts.sum():,}",
            f"tissue source sites: {sites}",
        ]
        return "\n".join(lines)
