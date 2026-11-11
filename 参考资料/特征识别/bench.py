# -*- coding: utf-8 -*-
"""速度基准：把“单次判断”拆成几段分别计时，报告 40 张 val 图的中位数 / p90。

用法
----
    python bench.py            # 打表（默认 40 张 val）
    python bench.py --n 20     # 只看 20 张

口径说明
--------
* 「读图」= PNG 解码成 ndarray（真实产线里这一步是相机取流 + 解码，同样要做）
* 「特征」= measure_array(..., keep_debug=False)，只算 4 个特征与圆心半径
* 「判断」= 分类器 + 阈值决策（不含执行机构动作；执行是外部设备的时间）
* 「整段」= pipeline.inspect()，包含读图 + 特征 + 判断 + 写记录
* 「+调试图」= keep_debug=True，界面要显示摊平偏差图时必须算，产线批量判断不用
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Callable, List

import features as F
import pipeline as P


def timed(fn: Callable[[], object], n: int) -> List[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def pct(v: List[float], q: float) -> float:
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * q))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="取多少张 val 图")
    args = ap.parse_args()

    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    feats = cfg["features"]
    root = os.path.join(cfg["dataset_root"], "images", "val")
    paths = []
    for cls in sorted(os.listdir(root)):
        d = os.path.join(root, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            paths.append(os.path.join(d, f))
    paths = paths[: args.n]
    print(f"样本：val 前 {len(paths)} 张\n")

    ins_plain = P.Inspector.from_config("config.json", dispatcher=P.LogOnlyDispatcher(), keep_debug=False)
    ins_debug = P.Inspector.from_config("config.json", dispatcher=P.LogOnlyDispatcher(), keep_debug=True)
    per = int(feats.get("bench_repeat", 1)) or 1

    rows = []
    for p in paths:
        a = F.load_rgb(p)
        rows.append((
            timed(lambda: F.load_rgb(p), per),
            timed(lambda: F.measure_array(a, feats, False), per),
            timed(lambda: F.measure_array(a, feats, True), per),
            timed(lambda: ins_plain.inspect(p), per),
            timed(lambda: ins_debug.inspect(p), per),
        ))

    names = ["读图 PNG 解码", "特征（4 个量，不含调试图）", "特征 + 调试图", "整段 inspect（产线路径）", "整段 inspect（界面路径）"]
    print(f"{'阶段':28s}{'中位数':>10s}{'p90':>10s}")
    print("-" * 48)
    for i, name in enumerate(names):
        allv = [v for r in rows for v in r[i]]
        print(f"{name:26s}{statistics.median(allv):8.2f} ms{pct(allv, 0.9):8.2f} ms")
    print("-" * 48)
    print("提示：执行机构（气缸/电机）的动作时间不在上表内，那属于“执行”环节，")
    print("      真实节拍 = 上表整段 + 传送带延时 Δ + 动作时间。")


if __name__ == "__main__":
    main()
