# -*- coding: utf-8 -*-
"""Generate diagrams for the data & metrics handbook."""
import math
import os
from PIL import Image, ImageDraw, ImageFont

OUT = r"D:\智造创新实践\figs"
os.makedirs(OUT, exist_ok=True)

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BD = r"C:\Windows\Fonts\msyhbd.ttc"
NAVY = (31, 59, 99)
BLUE = (46, 92, 138)
LIGHT = (226, 235, 246)
MID = (68, 114, 196)
GRAY = (89, 89, 89)
ORANGE = (197, 90, 17)
GREEN = (55, 110, 46)
RED = (166, 28, 28)


def f(size, bold=False):
    return ImageFont.truetype(FONT_BD if bold else FONT, size)


def center_text(d, box, text, font, fill, spacing=10):
    x0, y0, x1, y1 = box
    bb = d.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.multiline_text(((x0 + x1 - w) / 2, (y0 + y1 - h) / 2 - bb[1]), text,
                     font=font, fill=fill, spacing=spacing, align="center")


def arrow(d, p0, p1, color, width=6, head=20):
    d.line([p0, p1], fill=color, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    pts = [p1,
           (p1[0] - head * math.cos(ang - 0.42), p1[1] - head * math.sin(ang - 0.42)),
           (p1[0] - head * math.cos(ang + 0.42), p1[1] - head * math.sin(ang + 0.42))]
    d.polygon(pts, fill=color)


def fig_data_flow():
    """Who produces / consumes which data, per module."""
    W, H = 2300, 880
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    cols = [
        ("感知", "Sense", ["触发信号、光照", "相机参数"], ["图像、时间戳", "到位信号"], BLUE),
        ("分析", "Analyze", ["图像、模型权重", "类别定义"], ["类别、分数", "bbox／掩膜"], MID),
        ("决策", "Decide", ["分数、标准", "节拍、设备状态"], ["处置动作", "执行时刻"], GREEN),
        ("执行", "Act", ["处置、目标位置", "延时补偿值"], ["动作完成信号", "分拣结果"], ORANGE),
        ("反馈", "Feedback", ["确认信号、日志", "统计结果"], ["报表、报警", "模型更新依据"], RED),
    ]
    bw, gap, y0, bh = 400, 37, 150, 600
    x0 = (W - (5 * bw + 4 * gap)) // 2
    for i, (cn, en, ins, outs, color) in enumerate(cols):
        x = x0 + i * (bw + gap)
        d.rounded_rectangle((x, y0, x + bw, y0 + bh), radius=20, fill="white", outline=color, width=5)
        d.rectangle((x, y0, x + bw, y0 + 96), fill=color)
        center_text(d, (x, y0 + 6, x + bw, y0 + 70), cn, f(46, True), "white")
        center_text(d, (x, y0 + 64, x + bw, y0 + 96), en, f(24), "white")
        center_text(d, (x, y0 + 112, x + bw, y0 + 152), "输入数据", f(32, True), GRAY)
        yy = y0 + 164
        for line in ins:
            center_text(d, (x + 10, yy, x + bw - 10, yy + 58), line, f(30), (60, 60, 60))
            yy += 60
        d.line([(x + 40, y0 + 330), (x + bw - 40, y0 + 330)], fill=(200, 208, 218), width=3)
        center_text(d, (x, y0 + 340, x + bw, y0 + 380), "输出数据", f(32, True), color)
        yy = y0 + 392
        for line in outs:
            center_text(d, (x + 10, yy, x + bw - 10, yy + 58), line, f(30), (30, 30, 30))
            yy += 60
        if i:
            arrow(d, (x - gap + 2, y0 + bh // 2), (x - 8, y0 + bh // 2), MID, 5, 16)
    center_text(d, (100, 800, W - 100, 856),
                "每个环节都有输入与输出：输出的名字、单位、取值范围不确定，下游就用不了", f(34), GRAY)
    img.save(os.path.join(OUT, "fig_data_flow.png"))
    return "fig_data_flow.png"


def fig_timeline():
    """Time chain of one bottle cap: events and the data each stamp leaves."""
    W, H = 2000, 620
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    events = [
        ("到位检测", "t0", "到位信号"),
        ("拍摄", "t1", "图像"),
        ("分析完成", "t2", "类别/分数"),
        ("决策", "t2 + ε", "处置"),
        ("延时／跟踪", "t2 + Δ", "目标位置"),
        ("剔除", "t3", "动作"),
        ("确认", "t4", "记录"),
    ]
    n = len(events)
    left, right = 110, W - 110
    step = (right - left) / (n - 1)
    base = 330
    d.line([(left - 50, base), (right + 50, base)], fill=(180, 190, 205), width=7)
    colors = [BLUE, MID, GREEN, ORANGE, RED, (120, 70, 150), NAVY]
    for i, (name, t, d1) in enumerate(events):
        x = int(left + i * step)
        color = colors[i]
        d.ellipse((x - 24, base - 24, x + 24, base + 24), fill=color, outline="white", width=5)
        # above: index, name, timestamp
        center_text(d, (x - 130, base - 206, x + 130, base - 162), "① ② ③ ④ ⑤ ⑥ ⑦"[i * 2], f(40, True), color)
        center_text(d, (x - 140, base - 158, x + 140, base - 92), name, f(46, True), NAVY)
        center_text(d, (x - 140, base - 90, x + 140, base - 44), t, f(40), color)
        # below: what data this moment leaves (short label, details live in the table)
        center_text(d, (x - 140, base + 46, x + 140, base + 100), d1, f(38), (50, 50, 50))
    center_text(d, (90, base + 168, W - 90, base + 226),
                "每经过一个时刻，都留下一条带时间戳的数据；没有时间戳，事后就无法判断谁先谁后", f(38), GRAY)
    img.save(os.path.join(OUT, "fig_timeline.png"))
    return "fig_timeline.png"


a = fig_data_flow()
b = fig_timeline()
for p in (a, b):
    im = Image.open(os.path.join(OUT, p))
    print(p, im.size)
