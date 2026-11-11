# -*- coding: utf-8 -*-
"""图像测量：把一张瓶盖照片变成 4 个可解释的数字。

设计原则
--------
1. 只用 numpy + Pillow，不依赖 OpenCV / torch，现场机器上随时能跑。
2. 每一步都能在界面上画出来（圆心、半径、摊平图、亮暗像素），便于答辩解释。
3. 同一份图像永远得到同一个数：不引入随机数，不做数据增强。

处理链（与课件第 2 讲的“七步链”对应）
------------------------------------
    彩色图 -> 色度分割(相当于二值化) -> 圆心/半径(相当于轮廓) -> 极坐标摊平
           -> 按行减中位数(相当于去掉公共环纹) -> 三个特征(相当于特征计算)

四个输出
--------
notch   凹陷率   : 外轮廓半径曲线相对中位数最大下凹比例      （缺口敏感）
dark    暗斑面积 : 内部“又暗又不鲜艳”的像素占比            （污渍敏感）
purity  直线纯度 : 亮像素最长直线带点数 / sqrt(亮像素数)     （划痕敏感）
peak    直线带点数: 上述最长带的绝对点数                    （配合 purity 门槛）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# 默认参数：与 config.json 的 "features" 段一一对应，单独跑本模块时用默认值
# ----------------------------------------------------------------------------
DEFAULTS: Dict[str, float] = {
    "chroma_thresh": 45.0,      # 色度 > 45 视为“瓶盖本体”（背景是灰的，色度很低）
    "radius_percentile": 99.5,  # 用 99.5% 分位半径抗离群点
    "unwrap_angles": 720,       # 摊平图横向分辨率（角度）
    "unwrap_radii": 160,        # 摊平图纵向分辨率（半径）
    "r_min": 0.10,              # 只取半径 10%~86% 的环带（避开圆心高光和边缘阴影）
    "r_max": 0.86,
    "rim_bins": 720,            # 外轮廓曲线的角度分辨率
    "dark_split": -0.10,        # 亮度低于同行中位数 10% 才算“暗”
    "dark_chroma_ratio": 0.8,   # 且色度低于全图中位数的 80%（排除“深色盖”被整体误判）
    "bright_split": 0.08,       # 亮度高于同行中位数 8% 才算“亮”
    "inner_r": 0.72,            # 只在半径 72% 以内找缺陷（边缘环纹不算）
    "line_step": 1.5,           # 直线带投票的分箱宽度（像素）
    "line_angles": 90,          # 投影方向数（180°/90 = 每 2°一个方向）
}


@dataclass
class FeatureVector:
    """一只瓶盖的全部测量结果。"""

    notch: float = 0.0
    dark: float = 0.0
    purity: float = 0.0
    peak: float = 0.0
    # 以下为调试/可视化用的中间结果，不参与判断
    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)
    debug: Dict[str, object] = field(default_factory=dict)

    def as_features(self) -> Dict[str, float]:
        """判断模块只认这 4 个数，保证特征与判断解耦。"""
        return {"notch": self.notch, "dark": self.dark,
                "purity": self.purity, "peak": self.peak}


# ----------------------------------------------------------------------------
# 基础图像操作
# ----------------------------------------------------------------------------
def load_rgb(path: str) -> np.ndarray:
    """读图并转 float32 RGB（0-255）。固定尺寸不缩放：坐标即像素坐标。"""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def chroma(a: np.ndarray) -> np.ndarray:
    """色度 = max(R,G,B) - min(R,G,B)。灰色背景接近 0，彩色瓶盖显著大于 0。"""
    return a.max(axis=2) - a.min(axis=2)


def luminance(a: np.ndarray) -> np.ndarray:
    """亮度（BT.601 权重），用于“暗斑/亮线”判断。"""
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def disc_geometry(a: np.ndarray, p: Dict[str, float]) -> Tuple[float, float, float]:
    """返回瓶盖圆心 (cx, cy) 与半径 R（像素）。

    做法：取色度 > 阈值的像素（瓶盖本体），用质心当圆心，
    用 99.5% 分位到质心的距离当半径——分位数比最大值稳，不怕个别噪点。
    """
    ys, xs = np.nonzero(chroma(a) > p["chroma_thresh"])
    if xs.size < 100:                     # 兜底：图像几乎无彩色，退化为整图中心
        h, w = a.shape[:2]
        return w / 2.0, h / 2.0, min(h, w) * 0.35
    cx, cy = float(xs.mean()), float(ys.mean())
    r = np.hypot(xs - cx, ys - cy)
    return cx, cy, float(np.percentile(r, p["radius_percentile"]))


def rim_profile(a: np.ndarray, cx: float, cy: float, R: float,
                p: Dict[str, float]) -> np.ndarray:
    """外轮廓半径曲线：每个角度扇区里“最远的瓶盖像素”到圆心的距离 / R。

    缺口处这一圈的半径会突然变短，所以曲线上去就是一个“坑”。
    返回长度为 rim_bins 的数组，值约在 0.9~1.0 之间。
    """
    nb = int(p["rim_bins"])
    ys, xs = np.nonzero(chroma(a) > p["chroma_thresh"])
    ang = (np.arctan2(ys - cy, xs - cx) + math.pi) / (2 * math.pi)
    bins = np.minimum((ang * nb).astype(np.int32), nb - 1)
    prof = np.zeros(nb, dtype=np.float32)
    np.maximum.at(prof, bins, (np.hypot(xs - cx, ys - cy) / max(R, 1e-6)).astype(np.float32))
    # 极少数扇区没有像素（被缺口完全切掉）时，用邻域中位数补上，避免误判为“半径=0”
    empty = prof <= 0
    if empty.any() and (~empty).any():
        idx = np.arange(nb)
        prof[empty] = np.interp(idx[empty], idx[~empty], prof[~empty])
    return prof


def unwrap(a: np.ndarray, cx: float, cy: float, R: float,
           p: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把圆盘摊平成矩形（极坐标重采样）。

    返回
    ----
    lum    (nr, nt) 亮度
    ch     (nr, nt) 色度
    xs, ys (nr, nt) 每个采样点在原图中的整数坐标（用于把亮像素画回去）
    rr     (nr,)    每个半径行的 r/R
    """
    nt, nr = int(p["unwrap_angles"]), int(p["unwrap_radii"])
    theta = (np.arange(nt) + 0.5) / nt * 2 * math.pi - math.pi
    rr = p["r_min"] + (p["r_max"] - p["r_min"]) * (np.arange(nr) + 0.5) / nr
    TH, RR = np.meshgrid(theta, rr)                      # (nr, nt)
    xs = np.clip((cx + RR * R * np.cos(TH)).astype(np.int32), 0, a.shape[1] - 1)
    ys = np.clip((cy + RR * R * np.sin(TH)).astype(np.int32), 0, a.shape[0] - 1)
    px = a[ys, xs]
    return luminance(px), chroma(px), xs, ys, rr


def residual_map(lum: np.ndarray) -> np.ndarray:
    """去掉公共环纹：先按图中位数归一化，再在每一半径行内减去该行中位数。

    环纹/刻度在每个角度上都在同一半径位置——所以它们对“行中位数”没有贡献；
    减去之后，圆盘变成一片 0 附近的灰，只有这一只瓶盖上独有的东西（缺陷）留下。
    """
    ln = lum / max(float(np.median(lum)), 1e-6)
    return ln - np.median(ln, axis=1)[:, None]


def line_vote(xs: np.ndarray, ys: np.ndarray, p: Dict[str, float]) -> Tuple[float, float]:
    """直线带投票（Hough 的极简版）：把点往各个方向投影，看最多能有多少点落进同一条窄带。

    返回 (峰值点数, 该方向法向角)。
    """
    if xs.size < 15:
        return 0.0, 0.0
    step = float(p["line_step"])
    best, best_phi = 0.0, 0.0
    for k in range(int(p["line_angles"])):
        phi = math.pi * k / int(p["line_angles"])
        s = xs * math.cos(phi) + ys * math.sin(phi)
        b = np.rint(s / step).astype(np.int32)
        b -= b.min()
        v = float(np.bincount(b).max())
        if v > best:
            best, best_phi = v, phi
    return best, best_phi


# ----------------------------------------------------------------------------
# 对外主函数
# ----------------------------------------------------------------------------
def measure_image(path: str, params: Optional[Dict[str, float]] = None,
                  keep_debug: bool = True) -> FeatureVector:
    """测量一张图（从文件），返回 FeatureVector。"""
    return measure_array(load_rgb(path), params, keep_debug)


def measure_array(a: np.ndarray, params: Optional[Dict[str, float]] = None,
                  keep_debug: bool = True) -> FeatureVector:
    """测量一帧内存里的图像。以后接相机时直接传 ndarray 即可，不用改判断逻辑。"""
    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULTS})
    a = np.asarray(a, dtype=np.float32)
    cx, cy, R = disc_geometry(a, p)
    prof = rim_profile(a, cx, cy, R, p)

    # --- 特征 1：凹陷率（缺口）
    med = float(np.median(prof))
    notch = float(((med - prof) / max(med, 1e-6)).max()) if med > 0 else 0.0

    # --- 特征 2/3/4：暗斑面积、直线峰值、直线纯度
    lum, ch, xs, ys, rr = unwrap(a, cx, cy, R, p)
    resid = residual_map(lum)
    inner = (rr < p["inner_r"])[:, None]
    ch_med = float(np.median(ch))
    dark = (resid < p["dark_split"]) & (ch < p["dark_chroma_ratio"] * ch_med) & inner
    bright = (resid > p["bright_split"]) & inner

    # 注意：dark 的比例按“整幅摊平图”为分母统计（历史口径，阈值与之一致）
    dark_frac = float(dark.sum() / dark.size)
    peak, phi = line_vote(xs[bright], ys[bright], p)
    purity = float(peak / max(math.sqrt(float(bright.sum())) * 1.6, 1.0))

    fv = FeatureVector(notch=notch, dark=dark_frac, purity=purity, peak=float(peak),
                       cx=cx, cy=cy, radius=R, image_shape=a.shape[:2])
    if keep_debug:
        # 只放“画图需要”的数据，避免把大数组带进结果对象
        fv.debug = {
            "rim": prof,
            "bright_pts": (xs[bright], ys[bright]),
            "bright_dir": phi,
            "dark_ratio": float(dark.mean()),
            "image": a.astype(np.uint8),
            "unwrap_img": render_unwrap(resid, dark, bright),
        }
    return fv


def render_unwrap(resid: np.ndarray, dark: np.ndarray, bright: np.ndarray) -> np.ndarray:
    """把“去掉环纹后的偏差图”画成一张小图，供界面显示。

    灰 = 正常（偏差接近 0）；红 = 比周围亮；蓝 = 比周围暗。
    上边是圆心附近，下边是边缘——一眼能看出缺陷落在哪个半径上。
    """
    base = np.clip(128.0 + resid * 320.0, 0, 255)
    img = np.repeat(base[:, :, None], 3, axis=2)
    img[bright] = (225, 70, 55)
    img[dark] = (40, 90, 200)
    return img.astype(np.uint8)


def describe(fv: FeatureVector, thr: Dict[str, float]) -> str:
    """人话版一行摘要，方便命令行/日志使用。"""
    return (f"凹陷率 {fv.notch:.4f}(阈 {thr.get('notch', 0):.4f}) "
            f"暗斑 {fv.dark:.4f}(阈 {thr.get('dark', 0):.4f}) "
            f"纯度 {fv.purity:.2f}(阈 {thr.get('purity', 0):.2f}) "
            f"峰值 {fv.peak:.0f}(阈 {thr.get('peak', 0):.0f})")
