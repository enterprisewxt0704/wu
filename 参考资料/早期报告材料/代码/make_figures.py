# -*- coding: utf-8 -*-
"""Generate simple diagrams for the beginner course-notes document."""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = r"D:\智造创新实践\figs"
os.makedirs(OUT, exist_ok=True)

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BD = r"C:\Windows\Fonts\msyhbd.ttc"
BLUE = (31, 78, 121)
LIGHT = (222, 235, 247)
MID = (68, 114, 196)
GRAY = (89, 89, 89)
ORANGE = (197, 90, 17)
GREEN = (56, 118, 29)


def f(size, bold=False):
    return ImageFont.truetype(FONT_BD if bold else FONT, size)


def center_text(d, box, text, font, fill):
    x0, y0, x1, y1 = box
    bb = d.multiline_textbbox((0, 0), text, font=font, spacing=8, align="center")
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    d.multiline_text(((x0 + x1 - w) / 2, (y0 + y1 - h) / 2 - bb[1]), text,
                     font=font, fill=fill, spacing=8, align="center")


def arrow(d, p0, p1, color, width=6, head=18):
    d.line([p0, p1], fill=color, width=width)
    import math
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    pts = [
        p1,
        (p1[0] - head * math.cos(ang - 0.42), p1[1] - head * math.sin(ang - 0.42)),
        (p1[0] - head * math.cos(ang + 0.42), p1[1] - head * math.sin(ang + 0.42)),
    ]
    d.polygon(pts, fill=color)


def fig_loop():
    W, H = 2300, 780
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    names = [("感知", "Sense", "看清现场"), ("分析", "Analyze", "形成判断"),
             ("决策", "Decide", "选择处置"), ("执行", "Act", "改变现场"),
             ("反馈", "Feedback", "确认结果")]
    bw, bh, gap = 340, 230, 92
    x0, y = 110, 170
    boxes = []
    for i, (cn, en, desc) in enumerate(names):
        x = x0 + i * (bw + gap)
        boxes.append((x, y, x + bw, y + bh))
        d.rounded_rectangle((x, y, x + bw, y + bh), radius=22, fill=LIGHT, outline=BLUE, width=4)
        center_text(d, (x, y + 16, x + bw, y + 94), cn, f(54, True), BLUE)
        center_text(d, (x, y + 88, x + bw, y + 130), en, f(28), MID)
        center_text(d, (x, y + 152, x + bw, y + 200), desc, f(30), GRAY)
        center_text(d, (x + bw - 34, y + 6, x + bw - 6, y + 46), str(i + 1), f(26, True), ORANGE)
        if i:
            arrow(d, (x - gap + 8, y + bh // 2), (x - 8, y + bh // 2), MID, 6, 20)
    # feedback path: 反馈 bottom -> down -> back to 感知 bottom
    fb = boxes[-1]
    ss = boxes[0]
    mid_y = 620
    d.line([((fb[0] + fb[2]) // 2, fb[3]), ((fb[0] + fb[2]) // 2, mid_y)], fill=ORANGE, width=6)
    d.line([((fb[0] + fb[2]) // 2, mid_y), ((ss[0] + ss[2]) // 2, mid_y)], fill=ORANGE, width=6)
    arrow(d, ((ss[0] + ss[2]) // 2, mid_y), ((ss[0] + ss[2]) // 2, ss[3] + 4), ORANGE, 6, 20)
    center_text(d, (500, mid_y - 62, 1800, mid_y - 6),
                "反馈：动作结果返回系统，影响下一次感知与决策", f(34, True), ORANGE)
    img.save(os.path.join(OUT, "fig_loop.png"))


def fig_pipeline():
    W, H = 2300, 560
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    steps = [("采集", "彩色图像"), ("灰度化", "明暗信息"), ("滤波", "减弱噪点"),
             ("二值化", "目标/背景"), ("轮廓", "边界线"), ("特征", "直径/圆度"), ("规则", "PASS/R/C")]
    n = len(steps)
    margin, gap = 40, 46
    bw = (W - 2 * margin - gap * (n - 1)) // n
    bh, y = 150, 170
    for i, (cn, sub) in enumerate(steps):
        x = margin + i * (bw + gap)
        d.rounded_rectangle((x, y, x + bw, y + bh), radius=18, fill=LIGHT, outline=BLUE, width=4)
        center_text(d, (x, y + 8, x + bw, y + 90), cn, f(40, True), BLUE)
        center_text(d, (x, y + 88, x + bw, y + 132), sub, f(24), GRAY)
        if i:
            arrow(d, (x - gap + 4, y + bh // 2), (x - 6, y + bh // 2), MID, 5, 16)
    center_text(d, (margin, 380, W - margin, 440),
                "每一步都把上一步的结果，变成下一步可以使用的输入", f(34), GRAY)
    img.save(os.path.join(OUT, "fig_pipeline.png"))


def fig_tasks():
    W, H = 2300, 640
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    items = [
        ("分类", "Classification", "整张图是什么？", "一个类别\n（正常/缺口）", BLUE),
        ("目标检测", "Detection", "是什么＋在哪里？", "类别＋边界框\n（大致位置）", MID),
        ("分割", "Segmentation", "具体占哪些像素？", "像素掩膜\n（精确区域）", GREEN),
        ("异常检测", "Anomaly Detection", "是否偏离正常？", "异常分数\n（未知异常）", ORANGE),
    ]
    bw, gap, y, bh = 500, 60, 150, 330
    x0 = (W - (4 * bw + 3 * gap)) // 2
    for i, (cn, en, q, out, color) in enumerate(items):
        x = x0 + i * (bw + gap)
        d.rounded_rectangle((x, y, x + bw, y + bh), radius=20, fill="white", outline=color, width=5)
        d.rectangle((x, y, x + bw, y + 74), fill=color)
        center_text(d, (x, y + 6, x + bw, y + 70), cn, f(42, True), "white")
        center_text(d, (x + 10, y + 84, x + bw - 10, y + 128), en, f(24), GRAY)
        center_text(d, (x + 16, y + 136, x + bw - 16, y + 196), q, f(32, True), (0, 0, 0))
        center_text(d, (x + 16, y + 210, x + bw - 16, y + 310), out, f(30), color)
    center_text(d, (100, 520, W - 100, 580),
                "四类任务没有先进与落后之分，只有“输出是否满足制造需求”之分", f(32), GRAY)
    img.save(os.path.join(OUT, "fig_tasks.png"))


fig_loop()
fig_pipeline()
fig_tasks()
for p in ("fig_loop.png", "fig_pipeline.png", "fig_tasks.png"):
    im = Image.open(os.path.join(OUT, p))
    print(p, im.size)
