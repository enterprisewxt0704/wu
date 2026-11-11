# -*- coding: utf-8 -*-
"""决策：把「异常分 + 分类结果 + 阈值」变成三选一的处置。

这是整个项目里唯一一处“拍板”的地方，所以单独成文件、写得尽量短，
方便答辩时逐行讲清楚“模型回答像什么，系统回答现在做什么”。

三选一（与课件一致）
------------------
PASS     正常放行
RECHECK  不确定，交人工复检（**不是废品通道**）
REJECT   确认缺陷，剔除/分流

判定顺序
--------
1. 异常分 < low  且 判为正常 且 正常概率 ≥ pass_normal_prob  -> PASS
   （三个条件都满足才敢放行；分类器说“不是正常”时，哪怕异常分低也进复检）
2. low ≤ 异常分 < high                                       -> RECHECK
3. 异常分 ≥ high                                             -> REJECT
   其中：分类器判为正常、或置信度低于 conf_min 时，标记为“未知缺陷”
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class Decision:
    action: str                 # PASS / RECHECK / REJECT
    reason: str                 # 一句人话，写进记录与界面
    is_unknown: bool = False    # 是否“异常但说不清是哪一类”
    detail: str = ""            # 更细的判定依据（阈值比较过程）


# 特征的中文名（界面与记录都用中文，答辩时不用解释英文）
FEATURE_CN = {"notch": "凹陷率", "dark": "暗斑面积",
              "purity": "直线纯度", "peak": "直线带点数"}


def decide(score: float, pred: str, proba: Dict[str, float], cfg: Dict,
           worst: str = "") -> Decision:
    """pure function：同样的输入永远得到同样的处置，方便单元测试与回放。"""
    low = float(cfg["low"])
    high = float(cfg["high"])
    conf_min = float(cfg.get("conf_min", 0.6))
    pass_np = float(cfg.get("pass_normal_prob", 0.9))
    conf = float(max(proba.values()))
    p_normal = float(proba.get("normal", 0.0))
    worst_txt = f"（最异常的特征是{FEATURE_CN.get(worst, worst)}）" if worst else ""

    if score < low:
        if pred == "normal" and p_normal >= pass_np:
            return Decision("PASS", "异常分低于下限，且判为正常", False,
                            f"异常分 {score:.3f} < low {low:.2f}；正常概率 {p_normal:.2f}")
        return Decision("RECHECK",
                        f"异常分不高，但分类器判为「{pred}」，交复检确认", False,
                        f"异常分 {score:.3f} < low {low:.2f}，但预测 {pred}（置信 {conf:.2f}）")
    if score < high:
        return Decision("RECHECK", f"落在不确定区{worst_txt}，交人工复检", False,
                        f"low {low:.2f} ≤ 异常分 {score:.3f} < high {high:.2f}")
    # 异常分很高 —— 一定不能放行
    if pred == "normal" or conf < conf_min:
        return Decision("REJECT", f"异常分很高{worst_txt}，但四类都说不准，记为「未知缺陷」",
                        True, f"异常分 {score:.3f} ≥ high {high:.2f}；预测 {pred}（置信 {conf:.2f}）")
    return Decision("REJECT", f"确认缺陷：{pred}", False,
                    f"异常分 {score:.3f} ≥ high {high:.2f}；预测 {pred}（置信 {conf:.2f}）")
