"""
abmil.py
═════════
Attention-Based MIL (Ilse et al. 2018) for 免疫敏感二分类。

patch features (N,1536) -> gated attention -> patient-level prob

CPU 可训练 (轻量: 1536->256->128->1)。
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    """Gated attention pooling + 分类头。

    Parameters
    ----------
    in_dim : int  (UNI2-h = 1536)
    hidden : int  (attention MLP 隐层)
    dropout : float
    """

    def __init__(self, in_dim: int = 1536, hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        # 特征投影
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        # gated attention (Ilse 2018, eq. 8)
        self.att_V = nn.Linear(hidden, 128)
        self.att_U = nn.Linear(hidden, 128)
        self.att_w = nn.Linear(128, 1)
        # 分类头
        self.clf = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (N, in_dim) patch features of one slide
        returns: logits (1,), attention (N,)
        """
        x = self.proj(x)                       # (N, hidden)
        v = torch.tanh(self.att_V(x))          # (N, 128)
        u = torch.sigmoid(self.att_U(x))       # (N, 128)  gating
        a = self.att_w(v * u).squeeze(-1)      # (N,)
        a = F.softmax(a, dim=0)                # attention weights
        z = (x * a.unsqueeze(-1)).sum(0)       # (hidden,)  weighted pooling
        logit = self.clf(z)                    # (1,)
        return logit.squeeze(0).unsqueeze(0), a  # (1,) 保持 batch 维


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
