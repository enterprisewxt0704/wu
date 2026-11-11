# -*- coding: utf-8 -*-
"""Explanatory figures for the feature-based judging logic (PIL only)."""
import glob
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DS = r"C:\Users\wuxuanting\Downloads\创新实践\创新实践\dataset"
OUT = r"D:\智造创新实践\build"
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONTB = r"C:\Windows\Fonts\msyhbd.ttc"
NAVY = (31, 59, 99)
BLUE = (46, 92, 138)
RED = (200, 60, 50)
GREEN = (60, 130, 70)
GRAY = (110, 110, 110)
ORANGE = (214, 126, 30)

NB_TH, NB_R = 720, 160


def f(size, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, size)


def center(d, box, text, font, fill):
    x0, y0, x1, y1 = box
    bb = d.textbbox((0, 0), text, font=font)
    d.text((x0 + (x1 - x0 - (bb[2] - bb[0])) / 2 - bb[0],
            y0 + (y1 - y0 - (bb[3] - bb[1])) / 2 - bb[1]), text, font=font, fill=fill)


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def geom(a):
    chroma = a.max(axis=2) - a.min(axis=2)
    m = chroma > 45
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    R = float(np.percentile(np.hypot(xs - cx, ys - cy), 99.5))
    return cx, cy, R


def rim_profile(a, cx, cy, R, nb=NB_TH):
    chroma = a.max(axis=2) - a.min(axis=2)
    m = chroma > 45
    ys, xs = np.nonzero(m)
    th = (np.arctan2(ys - cy, xs - cx) + np.pi) / (2 * np.pi)
    r = np.hypot(xs - cx, ys - cy) / R
    b = np.minimum((th * nb).astype(np.int32), nb - 1)
    prof = np.zeros(nb, dtype=np.float32)
    np.maximum.at(prof, b, r.astype(np.float32))
    return prof


def polar(a, cx, cy, R):
    th = (np.arange(NB_TH) + 0.5) / NB_TH * 2 * np.pi - np.pi
    rr = 0.10 + 0.76 * (np.arange(NB_R) + 0.5) / NB_R
    TH, RR = np.meshgrid(th, rr)
    x = np.clip((cx + RR * R * np.cos(TH)).astype(np.int32), 0, a.shape[1] - 1)
    y = np.clip((cy + RR * R * np.sin(TH)).astype(np.int32), 0, a.shape[0] - 1)
    p = a[y, x]
    lum = 0.299 * p[..., 0] + 0.587 * p[..., 1] + 0.114 * p[..., 2]
    chroma = p.max(axis=2) - p.min(axis=2)
    return lum, chroma, rr


# ---------------------------------------------------------------- fig 1: unwrap
def fig_unwrap(img_path, name):
    a = load(img_path)
    cx, cy, R = geom(a)
    lum, chroma, rr = polar(a, cx, cy, R)
    ln = lum / np.median(lum)
    resid = ln - np.median(ln, axis=1)[:, None]

    W = 1500
    P = 430                      # panel size
    H = 700
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)

    # 1. original with circle
    im = Image.fromarray(a.astype(np.uint8)).resize((P, P), Image.LANCZOS)
    c.paste(im, (30, 90))
    dd = ImageDraw.Draw(c)
    s = P / a.shape[1]
    dd.ellipse([30 + (cx - R) * s, 90 + (cy - R) * s, 30 + (cx + R) * s, 90 + (cy + R) * s],
               outline=ORANGE, width=3)
    r0 = 7
    dd.line([30 + cx * s - r0, 90 + cy * s, 30 + cx * s + r0, 90 + cy * s], fill=ORANGE, width=3)
    dd.line([30 + cx * s, 90 + cy * s - r0, 30 + cx * s, 90 + cy * s + r0], fill=ORANGE, width=3)
    center(d, (30, 40, 30 + P, 86), "① 找到圆心，量出半径 R", f(30, True), NAVY)

    # 2. polar unwrap (grayscale)
    g = np.clip(ln / ln.max() * 255, 0, 255).astype(np.uint8)
    pol = Image.fromarray(g).resize((P, 300), Image.LANCZOS)
    c.paste(pol, (520, 90))

    # 3. threshold map: only clearly deviating pixels are coloured
    bright = resid > 0.08
    dark = resid < -0.10
    img3 = np.full((NB_R, NB_TH, 3), 236, dtype=np.uint8)
    img3[bright] = (214, 60, 50)
    img3[dark] = (40, 80, 170)
    res = Image.fromarray(img3).resize((P, 300), Image.LANCZOS)
    c.paste(res, (1000, 90))

    # annotations
    center(d, (520, 40, 520 + P, 86), "② 摊平成矩形（横=角度 纵=半径）", f(30, True), NAVY)
    center(d, (1000, 40, 1000 + P, 86), "③ 减掉标准环纹，只剩偏差", f(30, True), NAVY)
    y = 410
    d.line([30, y, 30 + P, y], fill=(225, 228, 233), width=2)
    d.line([520, y, 520 + P, y], fill=(225, 228, 233), width=2)
    d.line([1000, y, 1000 + P, y], fill=(225, 228, 233), width=2)

    # arrows between panels
    for x0 in (470, 950):
        d.line([x0, 240, x0 + 40, 240], fill=GRAY, width=4)
        d.polygon([(x0 + 40, 240), (x0 + 28, 234), (x0 + 28, 246)], fill=GRAY)

    notes = [
        "完好的瓶盖是一圈一圈的同心环，\n摊平后就是一条条水平直线。",
        "每只瓶盖都有环纹和刻度，\n它们不是缺陷；按“同一半径取中位数”\n把它们减掉，画面就干净了。\n剩下的亮暗斑才是这只瓶盖的异常。",
    ]
    d.text((30, 440), "\u3000", font=f(26), fill=GRAY)
    d.text((520, 440), notes[0], font=f(26), fill=(50, 50, 50), spacing=8)
    d.text((1000, 440), notes[1], font=f(26), fill=(50, 50, 50), spacing=8)
    d.text((30, 600), f"示例：{name}", font=f(24), fill=GRAY)

    c.save(os.path.join(OUT, "fig_explain_unwrap.png"))
    print("fig_explain_unwrap.png")


# ---------------------------------------------------------------- fig 2: rim curve
def fig_rim(normal_path, notch_path):
    W, H = 1500, 620
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)
    left, right, top, bot = 120, W - 60, 90, H - 130
    d.text((60, 30), "外轮廓半径-角度曲线：缺口就是曲线上的一个坑", font=f(34, True), fill=NAVY)

    def plot(img_path, color, label, dashed=False):
        a = load(img_path)
        cx, cy, R = geom(a)
        prof = rim_profile(a, cx, cy, R)
        prof = np.clip(prof, 0.6, 1.02)
        pts = []
        for i in range(0, NB_TH, 2):
            x = left + (right - left) * i / NB_TH
            y = bot - (bot - top) * (prof[i] - 0.6) / (1.02 - 0.6)
            pts.append((x, y))
        for i in range(len(pts) - 1):
            if dashed and i % 8 > 3:
                continue
            d.line([pts[i], pts[i + 1]], fill=color, width=4)
        return prof

    # axes
    d.line([left, bot, right, bot], fill=GRAY, width=2)
    d.line([left, top, left, bot], fill=GRAY, width=2)
    for k, lab in [(0.7, "0.7R"), (0.8, "0.8R"), (0.9, "0.9R"), (1.0, "1.0R")]:
        y = bot - (bot - top) * (k - 0.6) / 0.42
        d.line([left, y, right, y], fill=(232, 235, 240), width=1)
        d.text((left - 60, y - 14), lab, font=f(24), fill=GRAY)
    for ang in (0, 90, 180, 270, 360):
        x = left + (right - left) * ang / 360
        d.line([x, bot, x, bot + 8], fill=GRAY, width=2)
        d.text((x - 20, bot + 14), str(ang), font=f(24), fill=GRAY)
    d.text(((left + right) / 2 - 60, bot + 48), "角度 θ（度）", font=f(26), fill=GRAY)
    d.text((left - 90, top - 57), "半径/R", font=f(26), fill=GRAY)

    pn = plot(normal_path, BLUE, "正常")
    pp = plot(notch_path, RED, "缺口")

    # annotate dip
    i = int(np.argmin(pp))
    x = left + (right - left) * i / NB_TH
    y = bot - (bot - top) * (np.clip(pp[i], 0.6, 1.02) - 0.6) / 0.42
    d.line([x, y, x, y - 90], fill=RED, width=2)
    d.text((x + 10, y - 130), "这里半径突然变短\n= 边缘缺了一块", font=f(26, True), fill=RED, spacing=6)

    # legend
    d.line([left + 30, top + 20, left + 90, top + 20], fill=BLUE, width=5)
    d.text((left + 100, top + 6), "正常瓶盖：曲线几乎水平", font=f(26), fill=BLUE)
    d.line([left + 30, top + 66, left + 90, top + 66], fill=RED, width=5)
    d.text((left + 100, top + 52), "缺口瓶盖：某段角度上半径掉下去", font=f(26), fill=RED)

    d.text((60, H - 46), "正常最大凹陷 0.4%，缺口最大凹陷 14.5% —— 相差约 30 倍，非常容易分开。",
           font=f(26), fill=(50, 50, 50))
    c.save(os.path.join(OUT, "fig_explain_rim.png"))
    print("fig_explain_rim.png")


# ---------------------------------------------------------------- fig 3: hough
def fig_line(scratch_path, stain_path):
    W, H = 1500, 700
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)
    d.text((60, 28), "划痕与污渍的区别：亮像素是“排成一条线”还是“散成一团”", font=f(34, True), fill=NAVY)

    def scatter(img_path, x0, y0, S, title, color):
        a = load(img_path)
        cx, cy, R = geom(a)
        lum, chroma, rr = polar(a, cx, cy, R)
        ln = lum / np.median(lum)
        resid = ln - np.median(ln, axis=1)[:, None]
        th = (np.arange(NB_TH) + 0.5) / NB_TH * 2 * np.pi - np.pi
        TH, RR = np.meshgrid(th, rr)
        px = cx + RR * R * np.cos(TH)
        py = cy + RR * R * np.sin(TH)
        inner = RR < 0.72
        bright = (resid > 0.08) & inner
        d.rectangle([x0, y0, x0 + S, y0 + S], outline=(220, 224, 230), width=2)
        center(d, (x0, y0 - 52, x0 + S, y0 - 8), title, f(30, True), NAVY)
        xs = px[bright]
        ys = py[bright]
        if len(xs) == 0:
            return
        # best line via simple angle scan
        best, bphi = -1, 0.0
        for k in range(90):
            phi = np.pi * k / 90
            s = xs * np.cos(phi) + ys * np.sin(phi)
            b = np.rint(s / 1.5).astype(np.int32)
            b -= b.min()
            v = np.bincount(b).max()
            if v > best:
                best, bphi = v, phi
        k = S / a.shape[1]
        for X, Y in zip(xs, ys):
            d.point((x0 + X * k, y0 + Y * k), fill=color)
        # draw the winning band
        nx, ny = -np.sin(bphi), np.cos(bphi)
        cx0, cy0 = cx, cy
        s_all = xs * np.cos(bphi) + ys * np.sin(bphi)
        smid = np.median(s_all)
        p1 = (cx0 + nx * smid, cy0 + ny * smid)
        pts = [(x0 + (p1[0] - nx * R) * k, y0 + (p1[1] - ny * R) * k),
               (x0 + (p1[0] + nx * R) * k, y0 + (p1[1] + ny * R) * k)]
        d.line(pts, fill=(90, 90, 90), width=2)
        center(d, (x0, y0 + S + 12, x0 + S, y0 + S + 62),
               f"最宽的一条直线带里只有 {int(best)} 个点亮像素", f(24), GRAY)

    scatter(scratch_path, 90, 160, 520, "划痕：亮像素沿一条直线排开", RED)
    scatter(stain_path, 860, 160, 520, "污渍：亮像素散落在暗斑边缘", BLUE)

    d.text((90, H - 92), "把亮像素按不同方向投影，看“哪一个方向能堆出最长的一条带”。\n"
                         "划痕能堆出很长的一条；污渍的高光只是沿暗斑边缘弯着走，堆不长。",
           font=f(27), fill=(50, 50, 50), spacing=8)
    c.save(os.path.join(OUT, "fig_explain_line.png"))
    print("fig_explain_line.png")


# ---------------------------------------------------------------- fig 4: overlay gallery
def fig_gallery():
    picks = [("normal", "img_0_003.png"), ("notch", "img_1_004.png"),
             ("scratch", "img_2_002.png"), ("stain", "img_3_001.png")]
    S = 340
    W = 60 + 4 * (S + 40)
    H = 640
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)
    d.text((50, 24), "算法的“眼睛”：在原图上标出它认为异常的地方", font=f(32, True), fill=NAVY)
    for i, (cls, fn) in enumerate(picks):
        p = os.path.join(DS, "images", "train", cls, fn)
        a = load(p)
        cx, cy, R = geom(a)
        overlay = a.copy()
        lum, chroma, rr = polar(a, cx, cy, R)
        ln = lum / np.median(lum)
        resid = ln - np.median(ln, axis=1)[:, None]
        inner = (rr < 0.72)[:, None]
        bright = (resid > 0.08) & inner
        dark = (resid < -0.10) & (chroma < 0.8 * np.median(chroma)) & inner
        th = (np.arange(NB_TH) + 0.5) / NB_TH * 2 * np.pi - np.pi
        TH, RR = np.meshgrid(th, rr)
        px = (cx + RR * R * np.cos(TH))[dark]
        py = (cy + RR * R * np.sin(TH))[dark]
        overlay[py.astype(int), px.astype(int)] = (40, 80, 170)

        prof = rim_profile(a, cx, cy, R)
        Rmed = np.median(prof)
        if cls == "notch":
            yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            ang = ((np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi) * NB_TH).astype(int) % NB_TH
            rad = np.hypot(xx - cx, yy - cy) / R
            cut = (rad > prof[ang]) & (rad <= 1.02) & (prof[ang] < 0.95 * Rmed)
            overlay[cut] = (214, 60, 50)
        if cls == "scratch":
            xs = (cx + RR * R * np.cos(TH))[bright]
            ys = (cy + RR * R * np.sin(TH))[bright]
            if len(xs):
                best, bphi = -1, 0.0
                for k in range(90):
                    phi = np.pi * k / 90
                    sv = xs * np.cos(phi) + ys * np.sin(phi)
                    b = np.rint(sv / 1.5).astype(np.int32)
                    b -= b.min()
                    v = int(np.bincount(b).max())
                    if v > best:
                        best, bphi = v, phi
                sv = xs * np.cos(bphi) + ys * np.sin(bphi)
                band = np.abs(sv - np.median(sv)) < 3
                overlay[ys[band].astype(int), xs[band].astype(int)] = (214, 60, 50)

        im = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).resize((S, S), Image.LANCZOS)
        x = 50 + i * (S + 40)
        c.paste(im, (x, 140))
        d.rectangle([x, 140, x + S, 140 + S], outline=(210, 214, 220), width=2)
        name_cn = {"normal": "正常", "notch": "缺口", "scratch": "划痕", "stain": "污渍"}[cls]
        center(d, (x, 74, x + S, 134), f"{name_cn}（{cls}）", f(30, True), NAVY)

    d.text((50, 510), "红色 = 被判定为“比周围亮”（缺口处露出的背景、划痕）    蓝色 = 被判定为“又暗又不鲜艳”（污渍）",
           font=f(25), fill=(50, 50, 50))
    d.text((50, 552), "正常品：几乎什么都不标 —— 这就是“阈值以下是正常”的含义。",
           font=f(25), fill=(50, 50, 50))
    c.save(os.path.join(OUT, "fig_explain_gallery.png"))
    print("fig_explain_gallery.png")


if __name__ == "__main__":
    fig_unwrap(os.path.join(DS, "images", "train", "scratch", "img_2_002.png"), "scratch 划痕样本")
    fig_rim(os.path.join(DS, "images", "train", "normal", "img_0_002.png"),
            os.path.join(DS, "images", "train", "notch", "img_1_003.png"))
    fig_line(os.path.join(DS, "images", "train", "scratch", "img_2_002.png"),
             os.path.join(DS, "images", "train", "stain", "img_3_002.png"))
    fig_gallery()
