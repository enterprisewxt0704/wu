# -*- coding: utf-8 -*-
"""全工程共用的数据对象。

把“结论、质量、模型输出”定义成 dataclass，而不是到处传 dict：
    1. IDE 能补全，写错字段名会立刻报错；
    2. 记录 CSV / JSON / 界面 / 测试用的是同一套字段；
    3. 以后加字段（比如相机曝光、编码器计数）只改这里一处。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 四类的顺序必须与 model.pth 的 class_names 完全一致，不能改。
CLASSES = ("normal", "notch", "scratch", "stain")
CLASS_CN = {"normal": "正常", "notch": "缺口", "scratch": "划痕", "stain": "污渍"}
DEFECT_CLASSES = ("notch", "scratch", "stain")

ACTIONS = ("PASS", "RECHECK", "REJECT")
ACTION_CN = {"PASS": "放行", "RECHECK": "复检", "REJECT": "剔除"}

FEATURES = ("notch", "dark", "purity", "peak")
FEATURE_CN = {
    "notch": "凹陷率（缺口）",
    "dark": "暗斑面积（污渍）",
    "purity": "直线纯度（划痕）",
    "peak": "直线带点数",
}


@dataclass
class QualityReport:
    """质量门结果：先保证“图是能用的”，再谈模型判得准不准。"""

    ok: bool = True
    reasons: List[str] = field(default_factory=list)
    mean_lum: float = 0.0
    sharpness: float = 0.0
    center_offset: float = 0.0
    width: int = 0
    height: int = 0
    resized: bool = False

    def summary(self) -> str:
        if self.ok:
            return "质量正常"
        return "；".join(self.reasons)

    def to_dict(self) -> Dict:
        return {
            "quality_ok": int(self.ok),
            "quality_reason": self.summary(),
            "mean_lum": round(self.mean_lum, 2),
            "sharpness": round(self.sharpness, 2),
            "center_offset": round(self.center_offset, 4),
            "width": self.width,
            "height": self.height,
            "resized": int(self.resized),
        }


@dataclass
class ModelOutput:
    """一个分类模型的输出。特征模型和深度模型共用，界面上并排显示。"""

    name: str
    available: bool = True
    probs: Dict[str, float] = field(default_factory=dict)
    pred: str = ""
    conf: float = 0.0
    defect_score: float = 0.0
    latency_ms: float = 0.0
    error: str = ""

    @classmethod
    def unavailable(cls, name: str, error: str) -> "ModelOutput":
        return cls(name=name, available=False, error=error)

    def topk(self, k: int = 2) -> List[tuple]:
        return sorted(self.probs.items(), key=lambda kv: kv[1], reverse=True)[:k]


@dataclass
class FusionResult:
    """融合后的“系统结论”，decision 之前的最后一步。"""

    mode: str
    risk: float
    pred: str
    conf: float
    proba: Dict[str, float]
    action: str
    reason: str
    is_unknown: bool = False
    detail: str = ""
    rule: str = ""
    agreement: bool = False
    used_deep: bool = False
