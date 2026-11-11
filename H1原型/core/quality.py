# -*- coding: utf-8 -*-
"""质量门：在模型之前先判断“这张图能不能用”。

H1 明确要求处理“没有图像 / 输入异常”。真实产线里最危险的不是模型判错，
而是相机没拍到、镜头糊了、瓶盖只进来半个，模型还硬给一个 PASS。
所以任何一项质量检查不过，直接 RECHECK，绝不放行。

检查项
------
曝光     平均亮度太低/太高          -> 可能没打光或过曝
清晰度   梯度方差太低              -> 镜头脏/失焦/运动模糊
尺寸     太小或太大                -> 相机分辨率/ROI 配错
居中     瓶盖圆心离画面中心太远    -> 瓶盖没到位，特征不可信
重复触发 由 LineController 的防抖逻辑处理，不在这里
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .types import QualityReport


def _as_rgb_u8(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"需要 HxWx3 的图像，收到 {a.shape}")
    return a


def sharpness_score(gray: np.ndarray) -> float:
    """清晰度 = 水平/垂直一阶差分方差之和。

    没有 OpenCV 的 Laplacian，用最简单的梯度方差；数值越大越清晰。
    课程数据集 512×512 合成图最低约 108，真实失焦图通常 <20，所以阈值 30 很宽。
    """
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    return float(np.var(gx) + np.var(gy))


def assess(image: np.ndarray, cfg: Dict) -> QualityReport:
    """只看图像本身的质量项；居中检查要等特征模块算出圆心后再补。"""
    qcfg = cfg.get("quality", {})
    a = _as_rgb_u8(image)
    h, w = a.shape[:2]
    gray = a.astype(np.float32).mean(axis=2)
    rep = QualityReport(
        width=w,
        height=h,
        mean_lum=float(gray.mean()),
        sharpness=sharpness_score(gray),
    )

    if min(h, w) < int(qcfg.get("min_side", 256)):
        rep.reasons.append(f"图像太小（{w}×{h}，至少 {qcfg['min_side']} 像素）")
    if max(h, w) > int(qcfg.get("max_side", 4096)):
        rep.reasons.append(f"图像太大（{w}×{h}）")

    if rep.mean_lum < float(qcfg.get("mean_lum_min", 20)):
        rep.reasons.append(f"曝光不足（平均亮度 {rep.mean_lum:.0f}）")
    if rep.mean_lum > float(qcfg.get("mean_lum_max", 245)):
        rep.reasons.append(f"过曝（平均亮度 {rep.mean_lum:.0f}）")

    if rep.sharpness < float(qcfg.get("sharpness_min", 30)):
        rep.reasons.append(f"图像模糊（清晰度 {rep.sharpness:.0f}）")

    if not (0.5 <= w / max(h, 1) <= 2.0):
        rep.reasons.append(f"长宽比异常（{w}×{h}）")

    size = cfg.get("deep", {}).get("input_size", 512)
    if (w, h) != (size, size):
        rep.resized = True
        if not qcfg.get("allow_resize", True):
            rep.reasons.append(f"尺寸不是 {size}×{size}，且配置不允许缩放")

    rep.ok = not rep.reasons
    return rep


def check_center(rep: QualityReport, fv, cfg: Dict) -> QualityReport:
    """用特征模块算出的圆心/半径，检查瓶盖是否大致在画面中央。

    偏移量用“像素距离 / 半径”归一化，换分辨率不影响阈值。
    """
    qcfg = cfg.get("quality", {})
    if not fv or fv.radius <= 1:
        return rep
    h, w = fv.image_shape
    dx = fv.cx - w / 2.0
    dy = fv.cy - h / 2.0
    off = float(np.hypot(dx, dy) / max(fv.radius, 1e-6))
    rep.center_offset = off
    if off > float(qcfg.get("center_offset_max", 0.35)):
        rep.reasons.append(f"瓶盖偏离中心（偏移 {off:.2f}R）")
        rep.ok = False
    return rep
