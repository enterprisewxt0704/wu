# -*- coding: utf-8 -*-
"""Teaching figures for the feature-based judgement logic (PIL only)."""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

DS = r"C:\Users\wuxuanting\Downloads\创新实践\创新实践\dataset"
OUT = r"D:\智造创新实践\build"
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONTB = r"C:\Windows\Fonts\msyhbd.ttc"
NAVY = (31, 59, 99)
BLUE = (46, 92, 138)
RED = (204, 66, 52)
GREEN = (56, 128, 74)
GRAY = (105, 108, 115)
ORANGE = (214, 126, 30)
NB_TH, NB_R = 720, 160


def F(size, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, size)


def txt(d, xy, s, size, fill, bold=False, spacing=10):
    d.multiline_text(xy, s, font=F(size, bold), fill=fill, spacing=spacing)


def ctext(d, box, s, size, fill, bold=False):
    x0, y0, x1, y1 = box
    fnt = F(size, bold)
    bb = d.multiline_textbbox((0, 0), s, font=fnt, spacing=8)
    d.multiline_text((x0 + (x1 - x0 - (bb[2] - bb[0])) / 2, y0 + (y1 - y0 - (bb[3] - bb[1])) / 2 - bb[1]),
                     s, font=fnt, fill=fill, spacing=8)


def arrow(d, p0, p1, color=GRAY, w=4):
    d.line([p0, p1], fill=color, width=w)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    L, Wd = 16, 8
    d.polygon([p1,
               (p1[0] - L * math.cos(ang) + Wd * math.sin(ang), p1[1] - L * math.sin(ang) - Wd * math.cos(ang)),
               (p1[0] - L * math.cos(ang) - Wd * math.sin(ang), p1[1] - L * math.sin(ang) + Wd * math.cos(ang))],
              fill=color)


# ------------------------------------------------------------------ features
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
    ys, xs = np.nonzero(chroma > 45)
    th = (np.arctan2(ys - cy, xs - cx) + np.pi) / (2 * np.pi)
    b = np.minimum((th * nb).astype(np.int32), nb - 1)
    r = np.hypot(xs - cx, ys - cy) / R
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
    return lum, chroma, x, y, rr


def line_peak(xs, ys, step=1.5, nang=90):
    best, bphi = 0.0, 0.0
    for k in range(nang):
        phi = np.pi * k / nang
        s = xs * np.cos(phi) + ys * np.sin(phi)
        b = np.rint(s / step).astype(np.int32)
        b -= b.min()
        v = float(np.bincount(b).max())
        if v > best:
            best, bphi = v, phi
    return best, bphi


def measure(p):
    a = load(p)
    cx, cy, R = geom(a)
    prof = rim_profile(a, cx, cy, R)
    med = float(np.median(prof))
    notch = float(((med - prof) / med).max())
    lum, chroma, xg, yg, RR = polar(a, cx, cy, R)
    ln = lum / np.median(lum)
    resid = ln - np.median(ln, axis=1)[:, None]
    inner = (RR < 0.72)[:, None]
    dark = (resid < -0.10) & (chroma < 0.8 * np.median(chroma)) & inner
    bright = (resid > 0.08) & inner
    peak = purity = 0.0
    bphi = 0.0
    if bright.sum() >= 15:
        peak, bphi = line_peak(xg[bright], yg[bright])
        purity = peak / max(np.sqrt(bright.sum()) * 1.6, 1.0)
    return dict(cx=cx, cy=cy, R=R, img=a, notch=notch, dark=float(dark.mean()),
                peak=peak, purity=purity, phi=bphi,
                pts=(xg[bright], yg[bright]))


# ------------------------------------------------------------------ fig 1
def fig_schema():
    W, H = 1560, 720
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)
    txt(d, (56, 22), "把圆盘摊平成矩形，是一切规则的前提", 34, NAVY, True)

    # ---------- panel A : disc ----------
    ax, ay, AR = 250, 258, 148
    d.ellipse([ax - AR, ay - AR, ax + AR, ay + AR], fill=(208, 213, 222), outline=(88, 94, 104), width=5)
    for k in (0.62, 0.86):
        d.ellipse([ax - AR * k, ay - AR * k, ax + AR * k, ay + AR * k], outline=(250, 250, 250), width=4)
    for i in range(24):
        th = i * math.pi / 12
        d.line([(ax + AR * 0.93 * math.cos(th), ay + AR * 0.93 * math.sin(th)),
                (ax + AR * 0.99 * math.cos(th), ay + AR * 0.99 * math.sin(th))], fill=(250, 250, 250), width=3)
    d.line([(ax - 52, ay + 46), (ax + 30, ay - 68)], fill=(255, 255, 255), width=4)   # scratch
    d.line([ax - 8, ay, ax + 8, ay], fill=ORANGE, width=4)
    d.line([ax, ay - 8, ax, ay + 8], fill=ORANGE, width=4)
    d.line([(ax + 5, ay - 5), (ax + AR * 0.71, ay - AR * 0.70)], fill=ORANGE, width=3)
    txt(d, (ax + AR * 0.62, ay - AR * 0.98), "半径 R", 23, ORANGE)
    ctext(d, (ax - 160, ay + AR + 16, ax + 160, ay + AR + 62), "① 找圆心、量半径", 27, NAVY, True)

    # ---------- panel B : unwrap ----------
    bx, by, bw, bh = 520, 152, 380, 250
    d.rectangle([bx, by, bx + bw, by + bh], fill=(208, 213, 222), outline=(88, 94, 104), width=3)
    for frac in (0.62, 0.86):
        y = by + bh * frac
        d.line([bx, y, bx + bw, y], fill=(250, 250, 250), width=4)
    for i in range(0, 41):
        x = bx + bw * i / 40
        d.line([x, by + bh * 0.94, x, by + bh * 0.99], fill=(250, 250, 250), width=3)
    d.line([(bx + bw * 0.36, by + bh * 0.18), (bx + bw * 0.52, by + bh * 0.74)], fill=(255, 255, 255), width=4)
    ctext(d, (bx - 60, by - 66, bx + bw + 60, by - 16), "② 摊平成矩形", 27, NAVY, True)
    ctext(d, (bx, by + bh + 12, bx + bw, by + bh + 48), "横轴 = 绕一圈的角度", 23, GRAY)
    txt(d, (bx + bw + 14, by + bh * 0.60), "纵轴" + chr(10) + "= 半径", 23, GRAY, spacing=4)

    # ---------- panel C : rings removed ----------
    cx2, cy2 = 1000, 152
    d.rectangle([cx2, cy2, cx2 + bw, cy2 + bh], fill=(208, 213, 222), outline=(88, 94, 104), width=3)
    for frac in (0.62, 0.86):
        y = cy2 + bh * frac
        for x in range(cx2, cx2 + bw, 17):
            d.line([x, y, min(x + 9, cx2 + bw), y], fill=(152, 157, 167), width=3)
    for i in range(0, 41):
        x = cx2 + bw * i / 40
        for y in range(int(cy2 + bh * 0.94), int(cy2 + bh * 0.99), 9):
            d.line([x, y, x, min(y + 4, cy2 + bh)], fill=(152, 157, 167), width=2)
    d.line([(cx2 + bw * 0.36, cy2 + bh * 0.18), (cx2 + bw * 0.52, cy2 + bh * 0.74)], fill=RED, width=5)
    ctext(d, (cx2 - 70, cy2 - 66, cx2 + bw + 70, cy2 - 16), "③ 减掉公共环纹，只剩偏差", 27, NAVY, True)
    ctext(d, (cx2 - 40, cy2 + bh + 12, cx2 + bw + 40, cy2 + bh + 48), "灰色虚线 = 被减掉的环纹与刻度", 23, GRAY)

    arrow(d, (ax + AR + 26, ay), (bx - 26, ay), GRAY, 4)
    arrow(d, (bx + bw + 22, ay), (cx2 - 30, ay), GRAY, 4)

    txt(d, (56, 512),
        "为什么要摊平：圆盘上“一圈一圈的环纹”在矩形里变成水平线，而“划过的划痕”变成一条斜线。" + chr(10) +
        "水平线是所有瓶盖共有的（可以减掉），斜线是这一只瓶盖独有的（要保留）。",
        25, (45, 48, 55), spacing=14)
    txt(d, (56, 610), "黄色十字 = 圆心；白色细线 = 划痕；这一步只是换了个“画布”，还没有做任何判断。", 23, GRAY)
    d.line([56, 470, W - 56, 470], fill=(236, 238, 242), width=2)
    c.save(os.path.join(OUT, "fig_explain_unwrap.png"))
    print("unwrap")


# ------------------------------------------------------------------ fig 2
def fig_rim():
    W, H = 1500, 700
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)
    txt(d, (56, 24), "判断缺口：量一圈“边缘到圆心有多远”", 34, NAVY, True)

    left, right, top, bot = 140, W - 70, 190, H - 150
    d.line([left, bot, right, bot], fill=GRAY, width=2)
    d.line([left, top, left, bot], fill=GRAY, width=2)
    for k, lab in [(0.7, "0.7R"), (0.8, "0.8R"), (0.9, "0.9R"), (1.0, "1.0R")]:
        y = bot - (bot - top) * (k - 0.6) / 0.42
        d.line([left, y, right, y], fill=(235, 238, 243), width=1)
        txt(d, (left - 74, y - 15), lab, 23, GRAY)
    for ang in (0, 90, 180, 270, 360):
        x = left + (right - left) * ang / 360
        d.line([x, bot, x, bot + 8], fill=GRAY, width=2)
        txt(d, (x - 18, bot + 14), str(ang), 23, GRAY)
    ctext(d, (left, bot + 40, right, bot + 76), "绕一圈的角度（度）", 25, GRAY)
    txt(d, (left - 76, top - 44), "半径 / R", 25, GRAY)

    def curve(path, color):
        a = load(path)
        cx, cy, R = geom(a)
        prof = np.clip(rim_profile(a, cx, cy, R), 0.6, 1.02)
        pts = [(left + (right - left) * i / NB_TH,
                bot - (bot - top) * (prof[i] - 0.6) / 0.42) for i in range(0, NB_TH, 2)]
        d.line(pts, fill=color, width=5)
        return prof

    curve(os.path.join(DS, "images", "train", "normal", "img_0_002.png"), BLUE)
    prof = curve(os.path.join(DS, "images", "train", "notch", "img_1_003.png"), RED)

    i = int(np.argmin(prof))
    x = left + (right - left) * i / NB_TH
    y = bot - (bot - top) * (np.clip(prof[i], 0.6, 1.02) - 0.6) / 0.42
    # leader to a free area at lower-left, label there
    d.line([(x - 8, y + 8), (left + 230, bot - 118)], fill=RED, width=2)
    txt(d, (left + 244, bot - 136), "这里半径突然变短 = 边缘缺了一块", 25, RED, True)

    # colour key in the empty lower-right area (colours match the curves)
    d.line([(right - 350, bot - 150), (right - 300, bot - 150)], fill=BLUE, width=6)
    txt(d, (right - 288, bot - 168), "正常瓶盖：一圈下来几乎水平", 24, BLUE)
    d.line([(right - 350, bot - 96), (right - 300, bot - 96)], fill=RED, width=6)
    txt(d, (right - 288, bot - 114), "缺口瓶盖：某一小段掉下去了", 24, RED)

    txt(d, (56, H - 54), "实测：正常最大凹陷 0.4%，缺口 14.5%——相差约 30 倍，所以阈值放在 4% 很安全。",
        25, (45, 48, 55))
    c.save(os.path.join(OUT, "fig_explain_rim.png"))
    print("rim")


# ------------------------------------------------------------------ fig 3
def fig_line():
    W, H = 1560, 860
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)
    txt(d, (56, 24), "判断划痕：亮像素是“能排成一条线”，还是“散成一团”", 34, NAVY, True)

    def panel(path, x0, y0, S, title, color, line_color=(70, 74, 82)):
        m = measure(path)
        cx, cy, R = m["cx"], m["cy"], m["R"]
        k = S * 0.74 / (2 * R)
        ccx, ccy = x0 + S / 2, y0 + S / 2
        d.ellipse([ccx - R * k, ccy - R * k, ccx + R * k, ccy + R * k],
                  outline=(212, 216, 224), width=2)
        d.line([ccx - 8, ccy, ccx + 8, ccy], fill=(200, 204, 212), width=1)
        d.line([ccx, ccy - 8, ccx, ccy + 8], fill=(200, 204, 212), width=1)
        xs, ys = m["pts"]
        for X, Y in zip(xs, ys):
            d.point((ccx + (X - cx) * k, ccy + (Y - cy) * k), fill=color)
        if len(xs) >= 15:
            phi = m["phi"]
            Nx, Ny = np.cos(phi), np.sin(phi)          # band normal
            Dx, Dy = -Ny, Nx                           # band direction
            s0 = float(np.median(xs * Nx + ys * Ny))   # band: p . N = s0
            off = s0 - (cx * Nx + cy * Ny)             # distance from disc centre to band
            bx0, by0 = Nx * off, Ny * off              # band point, relative to centre
            half = float(np.sqrt(max(R * R - off * off, 0.0)))
            p1 = (ccx + (bx0 - Dx * half) * k, ccy + (by0 - Dy * half) * k)
            p2 = (ccx + (bx0 + Dx * half) * k, ccy + (by0 + Dy * half) * k)
            d.line([p1, p2], fill=line_color, width=3)
        ctext(d, (x0, y0 - 60, x0 + S, y0 - 12), title, 28, NAVY, True)
        return m

    m1 = panel(os.path.join(DS, "images", "train", "scratch", "img_2_002.png"), 110, 170, 480,
               "划痕：亮像素排成一条线", RED)
    m2 = panel(os.path.join(DS, "images", "train", "stain", "img_3_001.png"), 860, 170, 480,
               "污渍：亮像素散在暗斑周围，连不成线", BLUE, (176, 180, 188))

    ctext(d, (110, 666, 590, 706),
          f"直线带 {int(m1['peak'])} 个点，纯度 {m1['purity']:.1f}（>5）→ 判为划痕", 23, RED)
    ctext(d, (860, 666, 1340, 706),
          f"直线带 {int(m2['peak'])} 个点，纯度 {m2['purity']:.1f}（<5）→ 不算划痕", 23, BLUE)

    d.line([60, 736, W - 60, 736], fill=(238, 240, 244), width=2)
    txt(d, (60, 758),
        "做法：把亮像素往不同方向“投影”，看哪个方向能把它们压进同一条窄带里。"
        "划痕是一条直线，能压出很长的一条；\n污渍的高光沿着暗斑边缘绕圈，压不长。",
        25, (45, 48, 55), spacing=12)
    c.save(os.path.join(OUT, "fig_explain_line.png"))
    print("line")


# ------------------------------------------------------------------ fig 4
def fig_numbers():
    picks = [("normal", "img_0_003.png"), ("notch", "img_1_004.png"),
             ("scratch", "img_2_002.png"), ("stain", "img_3_001.png")]
    W, H = 1560, 850
    c = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(c)
    txt(d, (50, 22), "同一套判据，量四种瓶盖得到什么数字", 34, NAVY, True)

    heads = [("样本", 60), ("凹陷率\n(阈值 4%)", 340), ("暗斑面积\n(阈值 0.7%)", 610),
             ("直线纯度\n(阈值 5)", 890), ("结论", 1150)]
    for name, x in heads:
        ctext(d, (x, 84, x + 230, 168), name, 24, NAVY, True)

    T1, T2, TP = 0.04, 0.007, 5.0
    for r, (cls, fn) in enumerate(picks):
        m = measure(os.path.join(DS, "images", "train", cls, fn))
        y = 196 + r * 152
        im = Image.fromarray(m["img"].astype(np.uint8)).resize((108, 108), Image.LANCZOS)
        c.paste(im, (60, y + 6))
        name_cn = {"normal": "正常", "notch": "缺口", "scratch": "划痕", "stain": "污渍"}[cls]
        txt(d, (182, y + 44), name_cn, 27, NAVY, True)

        def cell(x, value, thr, fmt):
            ctext(d, (x, y + 18, x + 200, y + 84), fmt.format(value), 30, (35, 38, 45), True)
            over = value / thr
            ctext(d, (x, y + 82, x + 200, y + 112),
                  f"超阈值 {over:.1f} 倍" if over >= 1 else f"仅为阈值 {over:.2f}", 21,
                  RED if over >= 1 else GREEN)

        cell(340, m["notch"], T1, "{:.3f}")
        cell(610, m["dark"], T2, "{:.4f}")
        cell(890, m["purity"], TP, "{:.2f}")

        if m["dark"] > 0.05:
            verdict = ("污渍\nREJECT", RED)
        elif m["purity"] > TP and m["peak"] > 200:
            verdict = ("划痕\nREJECT", RED)
        elif m["notch"] > T1:
            verdict = ("缺口\nREJECT", RED)
        elif m["dark"] > T2:
            verdict = ("污渍\nREJECT", RED)
        else:
            verdict = ("正常\nPASS", GREEN)
        ctext(d, (1150, y + 14, 1400, y + 108), verdict[0], 27, verdict[1], True)
        if r < 3:
            d.line([50, y + 146, W - 40, y + 146], fill=(238, 240, 244), width=1)

    txt(d, (50, H - 56),
        "注意：判据是有先后顺序的。这只污渍瓶盖的“凹陷率”也略微过了阈值（1.0 倍），但程序先看“暗斑面积”，"
        + chr(10) + "所以结论是污渍而不是缺口——顺序本身就是知识。",
        24, (45, 48, 55))
    c.save(os.path.join(OUT, "fig_explain_numbers.png"))
    print("numbers")


if __name__ == "__main__":
    fig_schema()
    fig_rim()
    fig_line()
    fig_numbers()
