# -*- coding: utf-8 -*-
"""异常分数：只学“正常长什么样”，然后回答“这只离正常有多远”。

为什么需要它（答辩要点）
----------------------
分类器只在“见过的 4 类”里选，遇到没学过的缺陷（比如整片偏色、裂纹）也会硬塞一个类，
而且可能塞成 normal —— 缺陷就流出去了。
异常分数不问“是哪一类”，只问“离正常有多远”，所以**没见过的新缺陷也能报警**。

实现
----
对每个特征算稳健 z 分数：    z = (x - median_normal) / scale
其中 scale 取「1.4826 × MAD」与「到该特征阈值的距离的一半」中的较大者。
    · 用 MAD 而不是标准差：抗离群点，40 张正常样本就能估稳；
    · 为什么还要和“到阈值的距离”取大：正常品几乎一模一样时 MAD ≈ 0，
      直接除会让任何微小波动变成天文数字（这个坑第一次跑就踩到了）。
最后取 4 个特征里最大的 z，压成 0~1：score = 1 - exp(-max_z / 2)
    max_z = 0（和正常品一样）      -> 0.00
    max_z = 1.4（刚好到阈值附近）  -> 0.50
    max_z = 4（明显异常）          -> 0.86
    max_z = 8（远得离谱）          -> 0.98
    max_z = 30（到上限）           -> 0.999999
上限 z_cap=30 只是为了数值稳定；缺陷样本的 z 常在 10~40，截到 30 不影响排序与阈值。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

FEATURES = ("notch", "dark", "purity", "peak")


def _median_mad(values: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, mad


@dataclass
class Baseline:
    """正常品基线：每个特征的中位数 med、离散度 mad、以及打分用的 scale。"""

    medians: Dict[str, float] = field(default_factory=dict)
    mads: Dict[str, float] = field(default_factory=dict)
    scales: Dict[str, float] = field(default_factory=dict)
    n_samples: int = 0
    z_cap: float = 30.0

    # ---------------- 构建 ----------------
    @classmethod
    def fit(cls, feature_dicts: List[Dict[str, float]],
            thresholds: Dict[str, float], z_cap: float = 30.0) -> "Baseline":
        """只用 train 的正常品拟合；thresholds 来自同样只用 train 的 tune.py。"""
        b = cls(z_cap=z_cap)
        for k in FEATURES:
            v = np.asarray([d[k] for d in feature_dicts], dtype=np.float64)
            med, mad = _median_mad(v)
            dist_to_thr = max(float(thresholds.get(k, 0.0)) - med, 1e-9)
            b.medians[k] = med
            b.mads[k] = mad
            b.scales[k] = max(1.4826 * mad, dist_to_thr / 2.0)
        b.n_samples = len(feature_dicts)
        return b

    # ---------------- 打分 ----------------
    def zs(self, feat: Dict[str, float]) -> Dict[str, float]:
        """每个特征“离正常有多远”，单位是“半个阈值距离”。"""
        return {k: (feat[k] - self.medians[k]) / max(self.scales[k], 1e-12) for k in FEATURES}

    def score(self, feat: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """返回 (0~1 的异常分, 各特征 z 字典)。"""
        z = self.zs(feat)
        max_z = float(min(max(0.0, max(z.values())), self.z_cap))
        return 1.0 - float(np.exp(-max_z / 2.0)), z

    def worst_feature(self, feat: Dict[str, float]) -> str:
        """哪一项最异常——界面和日志里用来解释“为什么报警”。"""
        z = self.zs(feat)
        return max(z, key=z.get)

    # ---------------- 存取 ----------------
    def to_dict(self) -> dict:
        return {"medians": self.medians, "mads": self.mads, "scales": self.scales,
                "n_samples": self.n_samples, "z_cap": self.z_cap}

    @classmethod
    def from_dict(cls, d: dict) -> "Baseline":
        return cls(medians=dict(d["medians"]), mads=dict(d.get("mads", {})),
                   scales=dict(d["scales"]), n_samples=int(d.get("n_samples", 0)),
                   z_cap=float(d.get("z_cap", 30.0)))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Baseline":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
