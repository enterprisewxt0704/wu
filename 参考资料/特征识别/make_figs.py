# -*- coding: utf-8 -*-
"""生成两张用于报告/答辩的图（不依赖 matplotlib，全部用 PIL 画）。

    out/fig_decision_flow.png   五阶段流程 + 两条阈值链的判定关系
    out/fig_threshold_sweep.png 单阈值扫描：漏检率 vs 误剔率（阈值就是风险旋钮）

运行：
    python make_figs.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FONT = r"C:\Windows\Fonts\msyh.ttc"
FONTB = r"C:\Windows\Fonts\msyhbd.ttc"

NAVY = (31, 59, 99)
BLUE = (46, 92, 138)
LIGHT = (234, 241, 248)
GRAY = (105, 112, 119)
GREEN = (56, 128, 74)
ORANGE = (214, 126, 30)
RED = (204, 66, 52)
LINE = (222, 227, 234)


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONTB if bold else FONT, size)


def center(d, box, text, font, fill, spacing=6):
    """在 box 内水平居中、垂直尽量居中地写字（支持多行）。"""
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    heights = []
    for ln in lines:
        b = d.textbbox((0, 0), ln, font=font)
        heights.append(b[3] - b[1])
    total = sum(heights) + spacing * (len(lines) - 1)
    y = y0 + (y1 - y0 - total) / 2
    for ln, h in zip(lines, heights):
        w = d.textbbox((0, 0), ln, font=font)[2]
        d.text((x0 + (x1 - x0 - w) / 2, y), ln, font=font, fill=fill)
        y += h + spacing


def left(d, x, y, text, font, fill):
    d.text((x, y), text, font=font, fill=fill)


def arrow(d, p1, p2, color=BLUE, width=4, head=14):
    d.line([p1, p2], fill=color, width=width)
    (x1, y1), (x2, y2) = p1, p2
    dx, dy = x2 - x1, y2 - y1
    L = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    a = (x2 - ux * head + px * head * 0.6, y2 - uy * head + py * head * 0.6)
    b = (x2 - ux * head - px * head * 0.6, y2 - uy * head - py * head * 0.6)
    d.polygon([(x2, y2), a, b], fill=color)

# --------------------------------------------------------------- 图 1：流程图
def fig_flow() -> str:
    W, H = 1520, 980
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    left(d, 56, 32, "串联判断：从一张图到一条处置记录", f(33, True), NAVY)
    left(d, 56, 82, "上半部分是“怎么算”，下半部分是“怎么判”。阈值只有两条，但分别管两件事。",
         f(22), GRAY)

    # ---------------- 第一行：五阶段
    stages = [("① 感知 拍照", "图像 + 时间戳"),
              ("② 测量四个特征", "凹陷率 / 暗斑面积\n直线纯度 / 带点数"),
              ("③ 异常分（门卫）", "离正常有多远，\n不看类别"),
              ("④ 分类器（分诊台）", "四类概率 + 置信度"),
              ("⑤ 决策出处置", "PASS / RECHECK / REJECT")]
    bw, bh, gap = 246, 136, 42
    x0, y = 56, 146
    for i, (title, sub) in enumerate(stages):
        x = x0 + i * (bw + gap)
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=16, fill=LIGHT, outline=BLUE, width=3)
        center(d, (x + 6, y + 10, x + bw - 6, y + 56), title, f(24, True), NAVY)
        center(d, (x + 6, y + 58, x + bw - 6, y + bh - 8), sub, f(18), GRAY, spacing=3)
        if i:
            arrow(d, (x - gap + 6, y + bh // 2), (x - 6, y + bh // 2), BLUE, 4, 12)

    # ---------------- 第二行：两条阈值链
    top2 = 330
    d.line([56, top2 - 30, W - 56, top2 - 30], fill=LINE, width=2)

    # 左：异常分阈值链
    bx, bw2 = 56, 690
    d.rounded_rectangle([bx, top2, bx + bw2, top2 + 250], radius=14, fill=(250, 251, 253),
                        outline=LINE, width=2)
    left(d, bx + 22, top2 + 12, "阈值链 A：异常分（门卫）", f(23, True), NAVY)

    with open(os.path.join(HERE, "config.json"), "r", encoding="utf-8") as fh:
        _d = json.load(fh)["decision"]
    low, high = float(_d["low"]), float(_d["high"])
    band_frac = _d.get("band", [0.30, 0.70])
    policy_cn = {"risk_averse": "风险厌恶", "balanced": "均衡"}.get(_d.get("policy", ""), "")
    # 空档带两端（由 optimize.py 记在 out/threshold_opt.json 里），拿不到就退回文字描述
    band = ("正常最高分 ~ 缺陷最低分")
    try:
        with open(os.path.join(OUT, "threshold_opt.json"), "r", encoding="utf-8") as fh:
            _a = json.load(fh)["analysis"]
        band = f"{_a['train_normal_max']:.2f} ── {_a['train_defect_min']:.2f}"
    except Exception:                                          # noqa: BLE001
        pass
    bar_x, bar_w, bar_h = bx + 40, bw2 - 84, 24
    bar_y = top2 + 116
    d.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=(237, 240, 244))
    d.rectangle([bar_x, bar_y, bar_x + bar_w * low, bar_y + bar_h], fill=(205, 231, 211))
    d.rectangle([bar_x + bar_w * low, bar_y, bar_x + bar_w * high, bar_y + bar_h],
                fill=(251, 227, 200))
    d.rectangle([bar_x + bar_w * high, bar_y, bar_x + bar_w, bar_y + bar_h],
                fill=(245, 207, 201))
    x_low, x_high = bar_x + bar_w * low, bar_x + bar_w * high
    d.line([x_low, bar_y - 10, x_low, bar_y + bar_h + 10], fill=ORANGE, width=3)
    d.line([x_high, bar_y - 10, x_high, bar_y + bar_h + 10], fill=RED, width=3)
    # 两个数字分开写：low 写在左侧、high 写在右侧，避免挤在一起
    d.text((x_low - 64, bar_y - 38), f"low {low:.2f}", font=f(19, True), fill=ORANGE)
    d.text((x_high - 86, bar_y - 64), f"high {high:.2f}", font=f(19, True), fill=RED)

    # 分区图例（放在条下方，横向排开）
    lx, ly = bar_x, bar_y + bar_h + 18
    for col, txt in (((205, 231, 211), f"PASS（分数 < {low:.2f}）"),
                     ((251, 227, 200), f"RECHECK（{low:.2f} – {high:.2f}）"),
                     ((245, 207, 201), f"REJECT（≥ {high:.2f}）")):
        d.rectangle([lx, ly + 3, lx + 16, ly + 17], fill=col, outline=(200, 206, 214))
        w = d.textbbox((0, 0), txt, font=f(18))[2]
        d.text((lx + 24, ly), txt, font=f(18), fill=(60, 64, 72))
        lx += 24 + w + 22
    left(d, bx + 30, top2 + 190, "异常分只回答“离正常多远”，用来拦住没见过的缺陷。", f(19), GRAY)
    left(d, bx + 30, top2 + 218,
         f"low / high：空档带 {band} 的 {band_frac[0]:.0%} / {band_frac[1]:.0%} 处（{policy_cn}）。",
         f(18), GRAY)

    # 右：分类置信度
    cx2, cw2 = 776, 700
    d.rounded_rectangle([cx2, top2, cx2 + cw2, top2 + 250], radius=14, fill=(250, 251, 253),
                        outline=LINE, width=2)
    left(d, cx2 + 22, top2 + 12, "阈值链 B：分类置信度（分诊台）", f(23, True), NAVY)
    rows = [("正常 normal", 0.00, (185, 198, 214)),
            ("缺口 notch", 0.00, (185, 198, 214)),
            ("划痕 scratch", 1.00, BLUE),
            ("污渍 stain", 0.00, (185, 198, 214))]
    for i, (name, pr, col) in enumerate(rows):
        yy = top2 + 56 + i * 32
        left(d, cx2 + 30, yy, name, f(19), (40, 44, 51))
        d.rectangle([cx2 + 160, yy + 2, cx2 + 460, yy + 24], fill=(237, 240, 244))
        d.rectangle([cx2 + 160, yy + 2, cx2 + 160 + 300 * pr, yy + 24], fill=col)
        left(d, cx2 + 474, yy, f"{pr:.2f}", f(19), NAVY if pr >= 0.99 else GRAY)
    left(d, cx2 + 30, top2 + 196,
         "置信度低于 conf_min 时，即便判了类别也降级为复检。" + chr(10) +
         "conf_min 取训练集上“判对样本”置信度的 5% 分位。", f(19), GRAY)

    # ---------------- 第三行：判定表
    top3 = 636
    d.line([56, top3 - 26, W - 56, top3 - 26], fill=LINE, width=2)
    left(d, 56, top3 - 10, "把两条链合起来，只有三种结局（判断顺序不能反）", f(25, True), NAVY)

    cases = [("分数 < low", "且 预测=正常 且 正常概率 ≥ 0.9", "PASS", GREEN),
             ("low ≤ 分数 < high", "不管分类器说什么，都交人工复检", "RECHECK", ORANGE),
             ("分数 ≥ high", "预测=正常 或 置信度 < conf_min → 记「未知缺陷」", "REJECT", RED)]
    ty = top3 + 40
    for i, (c1, c2, act, col) in enumerate(cases):
        yy = ty + i * 74
        d.rounded_rectangle([56, yy, W - 56, yy + 62], radius=10, fill="white",
                            outline=LINE, width=2)
        d.rectangle([56, yy, 64, yy + 62], fill=col)
        left(d, 86, yy + 18, c1, f(22, True), NAVY)
        left(d, 372, yy + 20, c2, f(20), GRAY)
        d.text((W - 210, yy + 12), act, font=f(26, True), fill=col)

    left(d, 56, H - 38,
         "记住一句话：异常分低 ≠ 一定放行（还要看分类器）；异常分高 ≠ 一定剔除（还要看是否超出已知类别）。",
         f(20), BLUE)

    path = os.path.join(OUT, "fig_decision_flow.png")
    img.save(path)
    return path


def _val_scores():
    """跑一遍验证集，拿到 (异常分, 真值) 列表。"""
    from pipeline import Inspector

    ins = Inspector.from_config()
    rows = []
    with open(os.path.join(ins.cfg["dataset_root"], ins.cfg["labels_csv"]),
              "r", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == "val":
                rows.append(r)
    scores = []
    for r in rows:
        p = os.path.join(ins.cfg["dataset_root"], r["relative_path"].replace("/", os.sep))
        res = ins.inspect(p)
        scores.append((res.score, r["class_name"]))
    return ins, scores


def fig_sweep() -> str:
    """图 2：上半是分数分布（看得出两类分得很开），下半是阈值扫描曲线。"""
    ins, scores = _val_scores()
    defects = [s for s, t in scores if t != "normal"]
    normals = [s for s, t in scores if t == "normal"]

    Ts = [i / 200 for i in range(10, 201)]
    miss = [sum(1 for s in defects if s < T) / max(len(defects), 1) for T in Ts]
    false_rej = [sum(1 for s in normals if s >= T) / max(len(normals), 1) for T in Ts]

    W, H = 1400, 900
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    left(d, 60, 28, "异常分把两类分得很开，所以阈值有一个很宽的安全区", f(31, True), NAVY)
    left(d, 60, 74, "上半：验证集 40 张的真实异常分；下半：假设只用异常分做“放行/剔除”时，两种错误怎么变。",
         f(21), GRAY)

    px0, px1 = 120, W - 90
    low = float(ins.cfg["decision"]["low"])
    high = float(ins.cfg["decision"]["high"])

    def X(T):
        return px0 + (T - 0.0) / 1.0 * (px1 - px0)

    # ---------------- 上半：分布带
    ty0, ty1 = 150, 420
    d.rectangle([px0, ty0, px1, ty1], fill=(252, 253, 255), outline=LINE, width=2)
    # 三个分区底色
    d.rectangle([X(0), ty0, X(low), ty1], fill=(240, 248, 241))
    d.rectangle([X(low), ty0, X(high), ty1], fill=(253, 246, 236))
    d.rectangle([X(high), ty0, px1, ty1], fill=(253, 240, 238))

    y_normal, y_defect = ty0 + 96, ty0 + 208
    for i, s in enumerate(sorted(normals)):
        x = X(s)
        y = y_normal - 26 + (i % 7) * 8
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=BLUE)
    for i, s in enumerate(sorted(defects)):
        x = X(s)
        y = y_defect - 26 + (i % 7) * 8
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=RED)

    left(d, px0 + 10, ty0 + 8, f"正常品 {len(normals)} 张", f(21, True), BLUE)
    left(d, px0 + 10, ty0 + 30, f"分数 {min(normals):.3f} – {max(normals):.3f}", f(19), GRAY)
    left(d, px0 + 10, y_defect - 56, f"缺陷 {len(defects)} 张", f(21, True), RED)
    left(d, px0 + 10, y_defect - 34, f"分数 {min(defects):.3f} – {max(defects):.3f}", f(19), GRAY)

    for val, col in ((low, ORANGE), (high, RED)):
        d.line([X(val), ty0, X(val), ty1], fill=col, width=3)
    left(d, X(low) - 104, ty0 + 6, f"low {low:.2f}", f(19, True), ORANGE)
    left(d, X(high) - 128, ty0 + 30, f"high {high:.2f}", f(19, True), RED)
    # 分区图例写在带子下方，避免在窄区里挤成一团
    lx, ly = px0 + 6, ty1 + 12
    for col, txt in (((240, 248, 241), f"PASS 区（< {low:.2f}）"),
                     ((253, 246, 236), f"RECHECK 区（{low:.2f} – {high:.2f}）"),
                     ((253, 240, 238), f"REJECT 区（≥ {high:.2f}）")):
        d.rectangle([lx, ly + 3, lx + 16, ly + 17], fill=col, outline=(206, 211, 218))
        w = d.textbbox((0, 0), txt, font=f(18))[2]
        d.text((lx + 24, ly), txt, font=f(18), fill=(60, 64, 72))
        lx += 24 + w + 26

    # ---------------- 下半：扫描曲线
    py0, py1 = 520, H - 130
    for i in range(6):
        v = i / 5
        yy = py1 - v * (py1 - py0)
        d.line([px0, yy, px1, yy], fill=(240, 243, 247), width=1)
        left(d, 74, yy - 12, f"{int(v * 100)}%", f(19), GRAY)
    for T in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        xx = X(T)
        d.line([xx, py0, xx, py1], fill=(240, 243, 247), width=1)
        center(d, (xx - 30, py1 + 8, xx + 30, py1 + 40), f"{T:.1f}", f(19), GRAY)
    left(d, 56, py0 - 46, "错误比例", f(20), GRAY)
    left(d, px0, py1 + 44, "阈值 T（异常分 ≥ T 判为剔除）", f(20), GRAY)

    def Y(v):
        return py1 - v * (py1 - py0)

    d.line([(X(T), Y(v)) for T, v in zip(Ts, miss)], fill=RED, width=5, joint="curve")
    d.line([(X(T), Y(v)) for T, v in zip(Ts, false_rej)], fill=BLUE, width=5, joint="curve")
    left(d, X(0.55), Y(0.30), "漏检率（缺陷被放行）", f(21, True), RED)
    left(d, X(0.05), Y(0.62), "误剔率（正常品被剔除）", f(21, True), BLUE)
    d.line([X(low), py0, X(low), py1], fill=ORANGE, width=2)
    center(d, (X(low) - 130, py0 - 34, X(low) + 130, py0 - 4), f"当前 low = {low:.2f}", f(19, True), ORANGE)

    left(d, 60, H - 44,
         "曲线中间那一大片几乎是平的：阈值在 0.12 – 0.94 之间随便取，漏检和误剔都是 0。"
         "这份数据集分得很开，真实产线不会这么友好。", f(20), GRAY)

    path = os.path.join(OUT, "fig_threshold_sweep.png")
    img.save(path)
    with open(os.path.join(OUT, "threshold_sweep.json"), "w", encoding="utf-8") as fh:
        json.dump({"T": Ts, "miss_rate": miss, "false_reject_rate": false_rej,
                   "defect_scores": sorted(defects), "normal_scores": sorted(normals)},
                  fh, ensure_ascii=False, indent=1)
    print("defect scores:", [round(x, 3) for x in sorted(defects)])
    print("normal scores:", [round(x, 3) for x in sorted(normals)])
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print(fig_flow())
    print(fig_sweep())
