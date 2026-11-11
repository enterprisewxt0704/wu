# -*- coding: utf-8 -*-
"""漂移压力测试：阈值该放在空档带的哪里？

背景
----
`optimize.py` 把 low / high 放在“正常品最高分 ~ 缺陷最低分”这段空档带的 30% / 70% 处，
也就是**最大裕度**：两边都留 0.20，任何一侧漂移 0.2 以内都不会立刻出错。
但“零复检”这个结果本身并不能证明缓冲带够厚——它只说明这批数据没有样本落进带子里。

真正该问的是：**如果分数整体漂移，哪种阈值先出事？**
于是这里做三组对照（同一份分数、同一批样本，只换阈值）：

    balanced     0.39 / 0.66   空档正中，两侧各留 0.20（当前默认）
    conservative 0.22 / 0.83   贴着空档两端（只留 0.03），复检带最宽
    old          0.82 / 0.99   tune.py 的原始值（high 落在缺陷堆里）

漂移模型（刻意做得“有攻击性”但可解释）
    · 缺陷侧漂移 d：缺陷分数整体下降 d  —— 模拟“更淡的划痕 / 新颜色 / 光照变差”
    · 正常侧漂移 d：正常分数整体上升 d  —— 模拟“新批次正常品长得更奇怪”
只平移分数、不改分类器，用来回答“阈值放在哪更抗漂移”。

损失函数（可改，写在 COST 里）
    漏检 100 : 误剔 10 : 复检 1
——漏检流出到客户手里最贵；误剔是废掉一只好瓶盖；复检只是多花人工。

运行：
    python stress.py             # 打印对照表
    python stress.py --fig       # 另外画 out/fig_drift_stress.png
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONTB = r"C:\Windows\Fonts\msyhbd.ttc"
NAVY, BLUE, GRAY = (31, 59, 99), (46, 92, 138), (105, 112, 119)
GREEN, ORANGE, RED = (56, 128, 74), (214, 126, 30), (204, 66, 52)
LINE = (222, 227, 234)

# 漏检 : 误剔 : 复检 的相对代价
COST = (100.0, 10.0, 1.0)

CANDIDATES: Dict[str, Dict[str, float]] = {
    "balanced": {"low": 0.3909, "high": 0.6616},
    "conservative": {"low": 0.2217, "high": 0.8309},
    "old": {"low": 0.8214, "high": 0.9868},
}


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONTB if bold else FONT, size)


def left(d, x, y, txt, font, fill=NAVY):
    d.text((x, y), txt, font=font, fill=fill)


def load_scores() -> List[dict]:
    """读 optimize.py 缓存的分数表（train + val 共 200 张）。"""
    p = os.path.join(OUT, "score_table.json")
    if not os.path.exists(p):
        raise SystemExit("缺少 out/score_table.json，请先运行 python optimize.py")
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate(table: List[dict], low: float, high: float,
             shift_defect: float = 0.0, shift_normal: float = 0.0) -> Dict[str, float]:
    """在一组平移后的分数上统计漏检 / 误剔 / 复检。"""
    leak = false_rej = recheck = n_def = n_norm = 0
    for t in table:
        s = t["score"]
        if t["cls"] == "normal":
            s = min(1.0, s + shift_normal)
            n_norm += 1
        else:
            s = max(0.0, s - shift_defect)
            n_def += 1
        if s < low:
            if t["cls"] != "normal":
                leak += 1                      # 缺陷被放行 —— 最严重
        elif s < high:
            recheck += 1
        else:
            if t["cls"] == "normal":
                false_rej += 1                 # 正常品被剔除 —— 费钱
    return {
        "leak": leak / max(n_def, 1),
        "false_reject": false_rej / max(n_norm, 1),
        "recheck": recheck / max(len(table), 1),
        "cost": (COST[0] * leak + COST[1] * false_rej + COST[2] * recheck) / max(len(table), 1),
    }


def catch_rate(low: float) -> float:
    """新缺陷（从未见过、分数在 0~1 均匀分布）被“没放行”的概率 = 1 - low。

    这是审稿意见真正的着力点：low 越高，越多的“没见过的轻度缺陷”会被直接放行。
    均匀分布是最悲观的假设（也最诚实：我们对新缺陷的分数一无所知）。
    """
    return 1.0 - low


def mild_defect_catch(low: float, mid: float = 0.5, sd: float = 0.15) -> float:
    """假设新缺陷分数服从 N(mid, sd)（轻度缺陷），算“没被放行”的比例。"""
    import math
    n = 0
    p = 0.0
    for i in range(-60, 61):
        x = mid + i * sd / 10.0
        w = math.exp(-0.5 * ((x - mid) / sd) ** 2)
        n += 1
        if x >= low:
            p += w
    tot = sum(math.exp(-0.5 * ((mid + i * sd / 10.0 - mid) / sd) ** 2)
              for i in range(-60, 61))
    return p / tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", action="store_true", help="另外生成 out/fig_drift_stress.png")
    args = ap.parse_args()

    table = load_scores()
    drifts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    print("=" * 96)
    print("缺陷侧漂移（缺陷分数整体下调 d）：看谁先开始漏检")
    print(f"{'方案':>13} | {'low':>6} {'high':>6} | " +
          " | ".join(f"d={d:.2f}" for d in drifts))
    print("-" * 96)
    leak_rows: Dict[str, List[float]] = {}
    for name, th in CANDIDATES.items():
        vals = [evaluate(table, th["low"], th["high"], shift_defect=d)["leak"] for d in drifts]
        leak_rows[name] = vals
        print(f"{name:>13} | {th['low']:>6.2f} {th['high']:>6.2f} | " +
              " | ".join(f"{v * 100:5.0f}%" for v in vals))
    print("（表内数字=缺陷被放行的比例；0% 是最理想的）\n")

    print("=" * 96)
    print("正常侧漂移（正常分数整体上调 d）：看谁先开始误剔")
    print(f"{'方案':>13} | {'low':>6} {'high':>6} | " +
          " | ".join(f"d={d:.2f}" for d in drifts))
    print("-" * 96)
    fr_rows: Dict[str, List[float]] = {}
    for name, th in CANDIDATES.items():
        vals = [evaluate(table, th["low"], th["high"], shift_normal=d)["false_reject"] for d in drifts]
        fr_rows[name] = vals
        print(f"{name:>13} | {th['low']:>6.2f} {th['high']:>6.2f} | " +
              " | ".join(f"{v * 100:5.0f}%" for v in vals))
    print("（表内数字=正常品被剔除的比例）\n")

    print("=" * 96)
    print(f"综合代价（漏检 {COST[0]:.0f} : 误剔 {COST[1]:.0f} : 复检 {COST[2]:.0f}），"
          "同时施加缺陷 -d 与正常 +d")
    print(f"{'方案':>13} | " + " | ".join(f"d={d:.2f}" for d in drifts))
    print("-" * 96)
    cost_rows: Dict[str, List[float]] = {}
    for name, th in CANDIDATES.items():
        vals = [evaluate(table, th["low"], th["high"], shift_defect=d, shift_normal=d)["cost"]
                for d in drifts]
        cost_rows[name] = vals
        print(f"{name:>13} | " + " | ".join(f"{v:>6.2f}" for v in vals))
    print()

    print("=" * 96)
    print("未知缺陷（从未见过、分数未知）：low 决定了“有多少会被直接放行”")
    print(f"{'方案':>13} | {'low':>6} | {'均匀分布 0~1 的拦截率':>22} | {'轻度缺陷 N(0.5,0.15) 的拦截率':>26}")
    print("-" * 96)
    catch_rows = {}
    for name, th in CANDIDATES.items():
        u = catch_rate(th["low"])
        m = mild_defect_catch(th["low"])
        catch_rows[name] = (u, m)
        print(f"{name:>13} | {th['low']:>6.2f} | {u:>21.0%} | {m:>25.0%}")
    print("（拦截率 = 没有直接 PASS 的比例；漏掉的那部分就是流到客户手里的风险）")

    print("结论：")
    print("  · d=0 时三套阈值的漏检/误剔都是 0：这批数据分数分得开，阈值放在空档带哪里都一样——")
    print("    所以“零复检”说明的是数据分得开，不是阈值更聪明。")
    print("  · 缺陷分数整体下降时，low 越高越早漏检：旧阈值 low=0.82 只要下降 0.05 就开始漏；")
    print("    balanced(low=0.39) 要下降 0.4 才漏；conservative(low=0.22) 最抗这一侧。")
    print("  · 正常分数整体上升时，high 越低越早误剔：但正常品基线在 0.05 附近，")
    print("    high 只要在 0.35 以上，上升 0.3 都不会误剔，这一侧不构成瓶颈。")
    print("  · 真正决定“未知缺陷漏不漏”的是 low：均匀分布下 balanced 只拦 61%，conservative 拦 78%。")
    print("  · 结论：漏检比复检贵两个数量级，所以 low 应该压在正常包络附近（0.22），")
    print("    high 可以抬到接近缺陷包络（0.83）把“疑似”交给人工——这就是 conservative。")

    if args.fig:
        make_fig(drifts, leak_rows, fr_rows, cost_rows)
        print("已生成 out/fig_drift_stress.png")


def make_fig(drifts, leak_rows, fr_rows, cost_rows) -> None:
    """图：上=漏检率随漂移的变化；下=未知缺陷拦截率（这才是审稿意见的着力点）。"""
    W, H = 1440, 985
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    left(d, 60, 24, "漂移压力测试：阈值该放在空档带的哪里？", f(31, True), NAVY)
    left(d, 60, 72, "同一份分数、同一批样本，只换阈值；上图画“分数整体平移”，下图画“从没见过的缺陷”。",
         f(20), GRAY)

    colors = {"balanced": ORANGE, "conservative": GREEN, "old": BLUE}
    labels = {"balanced": "均衡 0.39 / 0.66",
              "conservative": "保守 0.22 / 0.83（默认）",
              "old": "旧 0.82 / 0.99"}

    # ---------------- 上图：漏检率
    left(d, 60, 112, "① 缺陷分数整体下降：漏检率", f(22, True), NAVY)
    lx = 400
    for name in leak_rows:
        d.line([lx, 126, lx + 30, 126], fill=colors[name], width=4)
        w = d.textbbox((0, 0), labels[name], font=f(18))[2]
        d.text((lx + 38, 114), labels[name], font=f(18), fill=(60, 64, 72))
        lx += 38 + w + 26

    px0, px1 = 130, W - 90
    py0, py1 = 170, 430
    d.rectangle([px0, py0, px1, py1], outline=LINE, width=2, fill=(252, 253, 255))

    def X(x):
        return px0 + x / 0.40 * (px1 - px0)

    def Y(v):
        return py1 - min(max(v, 0.0), 1.0) * (py1 - py0)

    for gv in (0.0, 0.25, 0.5, 0.75, 1.0):                 # 固定 0~100% 的刻度
        yy = Y(gv)
        d.line([px0, yy, px1, yy], fill=(238, 241, 246), width=1)
        d.text((px0 - 62, yy - 11), f"{gv * 100:.0f}%", font=f(17), fill=GRAY)
    for xv in (0.0, 0.1, 0.2, 0.3, 0.4):
        d.text((X(xv) - 14, py1 + 8), f"{xv:.1f}", font=f(17), fill=GRAY)
    left(d, px1 - 120, py1 + 32, "横轴 = 漂移量 d", f(17), GRAY)

    for name, vals in leak_rows.items():
        pts = [(X(x), Y(v)) for x, v in zip(drifts, vals)]
        if len(set(round(v, 4) for v in vals)) <= 2 and max(vals) < 0.005:
            # 全程都是 0 的曲线，画在轴上会看不见 → 抬到 1% 处并标注
            pts = [(X(x), Y(0.01)) for x in drifts]
        d.line(pts, fill=colors[name], width=4, joint="curve")
        for x, y in pts[::2]:
            d.ellipse([x - 4, y - 4, x + 4, y + 4], fill="white", outline=colors[name])

    left(d, px0 + 14, py0 + 10, "三条线都从 0 出发；只有旧阈值的 low 太高，缺陷分一降就漏。",
         f(18), GRAY)

    # ---------------- 下图：未知缺陷拦截率（水平分组条）
    left(d, 60, 486, "② 从没见过的缺陷：low 决定了有多少会被直接放行", f(22, True), NAVY)
    left(d, 60, 522, "假设新缺陷的分数分布未知：均匀铺在 0~1（悲观）或集中在 0.5 附近（轻度缺陷）。",
         f(19), GRAY)

    bx0, bx1 = 320, W - 150
    bar_h, step = 20, 28
    y = 556
    for name, th in CANDIDATES.items():
        u = catch_rate(th["low"])
        m = mild_defect_catch(th["low"])
        left(d, 60, y, labels[name], f(19, True), colors[name])
        left(d, 60, y + 22, f"low={th['low']:.2f}", f(16), GRAY)
        for k, (val, col, tag) in enumerate(((u, colors[name], "均匀 0~1"),
                                            (m, (150, 158, 168), "轻度 N(0.5,0.15)"))):
            yy = y + 22 + (k + 1) * step
            left(d, 200, yy + 2, tag, f(16), GRAY)
            d.rectangle([bx0, yy, bx1, yy + bar_h], fill=(247, 249, 252), outline=LINE, width=1)
            end_x = bx0 + (bx1 - bx0) * val
            d.rectangle([bx0, yy, end_x, yy + bar_h], fill=col)
            txt = f"{val * 100:.0f}%"
            if val > 0.30:
                d.text((end_x - 60, yy + 1), txt, font=f(17, True), fill="white")
            else:
                d.text((end_x + 10, yy + 1), txt, font=f(17, True), fill=(60, 64, 72))
        y += 22 + 2 * step + 34

    left(d, 60, H - 62,
         "读法：拦截率越高＝越多的“没见过的东西”被拦住（REJECT 或 RECHECK），而不是当成正常品放行。"
         "漏检比复检贵两个数量级，所以默认取保守。", f(19), GRAY)
    img.save(os.path.join(OUT, "fig_drift_stress.png"))


if __name__ == "__main__":
    main()
