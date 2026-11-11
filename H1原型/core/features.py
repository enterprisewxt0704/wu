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

性能（v2，目标 <10 ms/张）
-------------------------
· 全程 uint8 读图、int16 做色度，不做整幅 float 转换；
· 色度掩膜只算一次，圆心/半径/外轮廓共用（v1 里每步都重算一遍）；
· 外轮廓改成“沿射线由外向内找第一个瓶盖像素”，替掉 arctan2 + maximum.at；
· 摊平用「展平索引一次取点」（flat[Y*W+X]），cos/sin 表按分辨率缓存；
· 直线投票合成一次 bincount（90 个方向一起投），替掉 90 次 Python 循环；
· 界面要用的调试图只在 keep_debug=True 时才生成。
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
    "geom_stride": 1,           # 找圆心/外轮廓时把图隔点取样（1=原分辨率，2=半分辨率但外轮廓会变粗）
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
    """读图并转 uint8 RGB（0-255）。

    返回 uint8 而不是 float32：512x512 的图省掉一次 786k 元素的浮点转换，
    后面真正需要浮点的地方（摊平图，只有 11 万点）再转。
    """
    with Image.open(path) as im:
        if im.mode == "RGB":
            return np.asarray(im, dtype=np.uint8)
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _as_u8(a: np.ndarray) -> np.ndarray:
    """统一成 uint8 RGB（外部传 float / 灰度进来也能用）。"""
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"需要 HxWx3 的图像，收到 {a.shape}")
    return a


def _max3(a: np.ndarray) -> np.ndarray:
    """逐像素取 R/G/B 最大值。

    注意：**不要**写 a.max(axis=2)。numpy 对“最后一维只有 3 个数”的归约会退化成
    慢速循环（实测 3.1 ms），三次两两 maximum 只要 0.2 ms。
    """
    return np.maximum(np.maximum(a[..., 0], a[..., 1]), a[..., 2])


def _min3(a: np.ndarray) -> np.ndarray:
    """逐像素取 R/G/B 最小值（理由同 _max3）。"""
    return np.minimum(np.minimum(a[..., 0], a[..., 1]), a[..., 2])


_LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)   # 亮度权重（Rec.601）


def chroma(a: np.ndarray) -> np.ndarray:
    """色度 = max(R,G,B) - min(R,G,B)。灰色背景接近 0，彩色瓶盖显著大于 0。

    用 int16 做差（uint8 相减会溢出回绕），比 float32 版本快且结果相同。
    """
    a = np.asarray(a)
    return _max3(a).astype(np.int16) - _min3(a).astype(np.int16)


def luminance(a: np.ndarray) -> np.ndarray:
    """亮度（BT.601 权重），用于“暗斑/亮线”判断。"""
    a = np.asarray(a, dtype=np.float32)
    return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)


def _geometry_from_mask(mask: np.ndarray, p: Dict[str, float]) -> Tuple[float, float, float]:
    """圆心 = 掩膜质心；半径 = 所有点到质心距离的 99.5% 分位。"""
    ys, xs = np.nonzero(mask)
    if xs.size < 100:                     # 兜底：图像几乎无彩色，退化为整图中心
        h, w = mask.shape[:2]
        return w / 2.0, h / 2.0, min(h, w) * 0.35
    cx = float(xs.mean(dtype=np.float64))
    cy = float(ys.mean(dtype=np.float64))
    # 半径用全部点的 99.5% 分位：分位数比最大值稳，也不怕个别噪点。
    # 注意这里不做隔点取样——R 差 0.2 像素就会让摊平采样点整体偏移，
    # 进而让“亮像素点数”变 2%，足以翻转划痕/污渍这两类的边界样本。
    # 距离的平方更便宜（省掉 n 次开方），而开方是单调的，
    # 所以“分位数再开方”与“先开方再取分位数”结果一致。
    dx = xs - cx
    dy = ys - cy
    r2 = dx * dx + dy * dy
    return cx, cy, float(np.sqrt(np.percentile(r2, p["radius_percentile"])))


def disc_geometry(a: np.ndarray, p: Dict[str, float]) -> Tuple[float, float, float]:
    """返回瓶盖圆心 (cx, cy) 与半径 R（像素）。"""
    return _geometry_from_mask(chroma(a) > p["chroma_thresh"], p)


def _rim_from_mask(mask: np.ndarray, cx: float, cy: float, R: float,
                   p: Dict[str, float]) -> np.ndarray:
    """外轮廓半径曲线（沿射线由外向内找第一个瓶盖像素）。

    缺口处这一圈的半径会突然变短 —— 曲线上去就是一个“坑”。
    返回长度为 rim_bins 的数组，值约在 0.85~1.0 之间。
    """
    nb = int(p["rim_bins"])
    h, w = mask.shape
    r_hi = 1.02 * R
    r_lo = 0.55 * R
    nsteps = int(max(48, min(300, math.ceil((r_hi - r_lo)) + 2)))
    rad = np.linspace(r_hi, r_lo, nsteps, dtype=np.float32)
    ang = np.float32(-math.pi) + (np.arange(nb, dtype=np.float32) + 0.5) / nb * np.float32(2 * math.pi)
    cos_t = np.cos(ang).astype(np.float32)
    sin_t = np.sin(ang).astype(np.float32)

    xs = np.clip(np.rint(cx + cos_t[:, None] * rad[None, :]), 0, w - 1).astype(np.int32)
    ys = np.clip(np.rint(cy + sin_t[:, None] * rad[None, :]), 0, h - 1).astype(np.int32)
    hit = mask[ys, xs]                                    # (nb, nsteps) 由外向内
    first = np.argmax(hit, axis=1)                        # 第一个“打中”的步号
    has = hit.any(axis=1)
    prof = np.where(has, rad[first.astype(np.int32)], 0.0).astype(np.float32)
    prof /= max(R, 1e-6)
    # 极少数扇区没有像素（被缺口完全切掉）时，用邻域插值补上，避免误判为“半径=0”
    empty = ~has
    if empty.any() and (~empty).any():
        idx = np.arange(nb)
        prof[empty] = np.interp(idx[empty], idx[~empty], prof[~empty])
    return prof


def rim_profile(a: np.ndarray, cx: float, cy: float, R: float,
                p: Dict[str, float]) -> np.ndarray:
    """外轮廓半径曲线（对外接口，内部复用同一份掩膜）。"""
    return _rim_from_mask(chroma(a) > p["chroma_thresh"], cx, cy, R, p)


# --- 摊平用的 cos/sin 表：与图像无关，按分辨率缓存，避免每张图重算 ----------
_UNWRAP_TABLES: Dict[Tuple[int, int, float, float], Dict[str, np.ndarray]] = {}


def _unwrap_tables(nt: int, nr: int, r_min: float, r_max: float) -> Dict[str, np.ndarray]:
    key = (nt, nr, round(r_min, 4), round(r_max, 4))
    t = _UNWRAP_TABLES.get(key)
    if t is None:
        theta = (np.arange(nt, dtype=np.float32) + 0.5) / nt * np.float32(2 * math.pi) \
            - np.float32(math.pi)
        rr = (r_min + (r_max - r_min) * (np.arange(nr, dtype=np.float32) + 0.5) / nr).astype(np.float32)
        t = {"cos": np.cos(theta).astype(np.float32),
             "sin": np.sin(theta).astype(np.float32),
             "rr": rr}
        _UNWRAP_TABLES[key] = t
    return t


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
    a = _as_u8(a)
    nt, nr = int(p["unwrap_angles"]), int(p["unwrap_radii"])
    t = _unwrap_tables(nt, nr, float(p["r_min"]), float(p["r_max"]))
    h, w = a.shape[:2]
    scale = t["rr"][:, None] * np.float32(R)                    # (nr, 1)
    xs = np.clip(cx + scale * t["cos"][None, :], 0, w - 1.001).astype(np.int32)
    ys = np.clip(cy + scale * t["sin"][None, :], 0, h - 1.001).astype(np.int32)

    # 展平索引取点：先分别取 R/G/B 三个平面，比 take 三维再切片快约 0.6 ms
    # （三维 fancy indexing 要按行复制 3 个通道，缓存不友好）
    idx = ys.ravel() * w + xs.ravel()
    r = a[..., 0].ravel()[idx]
    g = a[..., 1].ravel()[idx]
    b = a[..., 2].ravel()[idx]
    lum = (r * np.float32(0.299) + g * np.float32(0.587) + b * np.float32(0.114)).reshape(nr, nt)
    ch = (np.maximum(np.maximum(r, g), b).astype(np.int16)
          - np.minimum(np.minimum(r, g), b).astype(np.int16)).reshape(nr, nt)
    return lum, ch, xs, ys, t["rr"]


def residual_map(lum: np.ndarray) -> np.ndarray:
    """去掉公共环纹：先按图中位数归一化，再在每一半径行内减去该行中位数。

    环纹/刻度在每个角度上都在同一半径位置——所以它们对“行中位数”没有贡献；
    减去之后，圆盘变成一片 0 附近的灰，只有这一只瓶盖上独有的东西（缺陷）留下。
    """
    # 中位数用隔点取样算（11 万点取 5.5 万点，中位数几乎不变，省一半时间）
    stride = 2 if lum.size > 60000 else 1
    ln = lum / max(float(np.median(lum.ravel()[::stride])), 1e-6)
    row_med = np.median(ln[:, ::stride], axis=1)[:, None]
    return (ln - row_med).astype(np.float32)


def line_vote(xs: np.ndarray, ys: np.ndarray, p: Dict[str, float]) -> Tuple[float, float]:
    """直线带投票（Hough 的极简版）：把点往各个方向投影，看最多能有多少点落进同一条窄带。

    返回 (峰值点数, 该方向法向角)。

    实现要点：所有方向一次性算完，再用**一次** bincount 统计（把“方向编号 × 大常数
    + 分箱号”编码成一个整数键），避免 90 次 Python 循环。
    """
    n = int(xs.size)
    if n < 15:
        return 0.0, 0.0
    na = int(p["line_angles"])
    inv_step = np.float32(1.0 / float(p["line_step"]))
    ang = (math.pi * np.arange(na, dtype=np.float32) / na).astype(np.float32)
    cos_t = np.cos(ang).astype(np.float32)
    sin_t = np.sin(ang).astype(np.float32)

    X = xs.astype(np.float32)
    Y = ys.astype(np.float32)
    best, best_k = 0, 0
    # 逐方向投影 + bincount：实测比“先拼 90×N 大矩阵再一次 bincount”快一倍
    # （后者要为几千个点分配 90 份中间数组，内存往返比算术还贵）。
    for k in range(na):
        b = np.rint((X * cos_t[k] + Y * sin_t[k]) * inv_step).astype(np.int32)
        b -= b.min()
        v = int(np.bincount(b).max())
        if v > best:
            best, best_k = v, k
    return float(best), float(math.pi * best_k / na)


# ----------------------------------------------------------------------------
# 对外主函数
# ----------------------------------------------------------------------------
def _params(params: Optional[Dict[str, float]]) -> Dict[str, float]:
    p = dict(DEFAULTS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULTS})
    return p


def measure_image(path: str, params: Optional[Dict[str, float]] = None,
                  keep_debug: bool = True) -> FeatureVector:
    """测量一张图（从文件），返回 FeatureVector。"""
    return measure_array(load_rgb(path), params, keep_debug)


def measure_array(a: np.ndarray, params: Optional[Dict[str, float]] = None,
                  keep_debug: bool = True) -> FeatureVector:
    """测量一帧内存里的图像。以后接相机时直接传 ndarray 即可，不用改判断逻辑。"""
    p = _params(params)
    a = _as_u8(a)

    # --- 圆心/半径/外轮廓：可以在隔点取样的小图上做（这几个量都是“相对 R”的，
    #     半径减半、坐标减半不影响凹陷率；省掉 3/4 的逐像素运算）。
    g = int(p.get("geom_stride", 1) or 1)
    a_small = a[::g, ::g] if g > 1 else a
    mask_small = chroma(a_small) > p["chroma_thresh"]
    cx_s, cy_s, R_s = _geometry_from_mask(mask_small, p)
    prof = _rim_from_mask(mask_small, cx_s, cy_s, R_s, p)
    if g > 1:                       # 小图坐标换回原图坐标（像素中心对齐）
        cx, cy, R = cx_s * g + (g - 1) / 2.0, cy_s * g + (g - 1) / 2.0, R_s * g
    else:
        cx, cy, R = cx_s, cy_s, R_s

    # --- 特征 1：凹陷率（缺口）
    med = float(np.median(prof))
    notch = float(((med - prof) / max(med, 1e-6)).max()) if med > 0 else 0.0

    # --- 特征 2/3/4：暗斑面积、直线峰值、直线纯度
    lum, ch, xs, ys, rr = unwrap(a, cx, cy, R, p)
    resid = residual_map(lum)
    inner = (rr < np.float32(p["inner_r"]))[:, None]
    ch_med = float(np.median(ch))
    dark = (resid < np.float32(p["dark_split"])) & \
           (ch < np.float32(p["dark_chroma_ratio"] * ch_med)) & inner
    bright = (resid > np.float32(p["bright_split"])) & inner

    # 注意：dark 的比例按“整幅摊平图”为分母统计（历史口径，阈值与之一致）
    dark_frac = float(dark.sum() / dark.size)
    bxs, bys = xs[bright], ys[bright]
    peak, phi = line_vote(bxs, bys, p)
    purity = float(peak / max(math.sqrt(float(bright.sum())) * 1.6, 1.0))

    fv = FeatureVector(notch=notch, dark=dark_frac, purity=purity, peak=float(peak),
                       cx=cx, cy=cy, radius=R, image_shape=a.shape[:2])
    if keep_debug:
        # 只放“画图需要”的数据，避免把大数组带进结果对象
        fv.debug = {
            "rim": prof,
            "bright_pts": (bxs, bys),
            "bright_dir": phi,
            "dark_ratio": float(dark.mean()),
            "image": a,
            "unwrap_img": render_unwrap(resid, dark, bright),
            "resid": resid,
            "dark": dark,
            "bright": bright,
            "rr": rr,
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
