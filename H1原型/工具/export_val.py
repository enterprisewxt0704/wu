# -*- coding: utf-8 -*-
"""导出 val 的逐张预测、混淆矩阵与分类别指标（报告/PPT 素材）。

用法：
    ../支持文件/Python环境/Scripts/python.exe 工具/export_val.py
输出：
    运行结果/val_predictions_m3.csv
    运行结果/val_confusion_m3.csv
    运行结果/val_confusion_m3.png
    运行结果/val_per_class_m3.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from core import config as C
from core.pipeline import Inspector
from core.types import CLASSES, CLASS_CN
from devices.base import NullDispatcher


def font(size: int):
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def main() -> int:
    cfg = C.load_config()
    paths = C.split_images(cfg, "val")
    ins = Inspector(cfg, ROOT, NullDispatcher(), None, None, keep_debug=False)
    ins.set_fusion_mode("M3")

    rows = []
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    for p in paths:
        res = ins.inspect(p, dispatch=False)
        rows.append({
            "frame_id": res.frame_id,
            "label": res.label,
            "pred": res.pred,
            "risk": f"{res.risk:.4f}",
            "anomaly_score": f"{res.anomaly_score:.4f}",
            "deep_defect": f"{res.deep.defect_score:.4f}" if res.deep else "",
            "action": res.action,
            "unknown": int(bool(res.fusion and res.fusion.is_unknown)),
            "elapsed_ms": f"{res.elapsed_ms:.1f}",
        })
        if res.label in CLASSES and res.pred in CLASSES:
            matrix[CLASSES.index(res.label)][CLASSES.index(res.pred)] += 1

    out = C.out_dir(cfg)
    with open(out / "val_predictions_m3.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(out / "val_confusion_m3.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["真值\\预测", *CLASSES])
        for i, c in enumerate(CLASSES):
            w.writerow([c, *matrix[i]])

    # 分类别指标
    lines = ["# M3 分类别指标（val 40 张）", ""]
    lines.append("| 类别 | 真值数 | 判对 | 召回 | 误判成 |")
    lines.append("|---|---|---|---|---|")
    for i, c in enumerate(CLASSES):
        total = sum(matrix[i])
        correct = matrix[i][i]
        wrong = []
        for j, c2 in enumerate(CLASSES):
            if j != i and matrix[i][j]:
                wrong.append(f"{CLASS_CN[c2]} {matrix[i][j]}")
        lines.append(f"| {CLASS_CN[c]} | {total} | {correct} | "
                     f"{correct / max(total,1):.2%} | {'、'.join(wrong) or '—'} |")
    lines.append("")
    lines.append("> 课程数据为教学合成图；仅用于流程验证。")
    (out / "val_per_class_m3.md").write_text("\n".join(lines), encoding="utf-8")

    # 画混淆矩阵
    cell = 96
    left, top = 132, 132
    width = left + cell * 4 + 40
    height = top + cell * 4 + 40
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    f_title = font(22)
    f_head = font(16)
    f_cell = font(22)
    d.text((24, 22), "M3 融合混淆矩阵（val 40 张）", fill=(20, 30, 45), font=f_title)
    d.text((24, 58), "行 = 真值，列 = 预测；对角线越深越好", fill=(90, 100, 115), font=font(13))

    for j, c in enumerate(CLASSES):
        d.text((left + j * cell + cell // 2, top - 30), CLASS_CN[c],
               fill=(40, 50, 65), font=f_head, anchor="mm")
    for i, c in enumerate(CLASSES):
        d.text((left - 16, top + i * cell + cell // 2), CLASS_CN[c],
               fill=(40, 50, 65), font=f_head, anchor="rm")

    vmax = max(max(row) for row in matrix) or 1
    for i in range(4):
        for j in range(4):
            v = matrix[i][j]
            t = v / vmax
            # 白 -> 蓝
            fill = (int(255 - 165 * t), int(255 - 125 * t), int(255 - 35 * t))
            x0, y0 = left + j * cell, top + i * cell
            d.rectangle([x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2],
                        fill=fill, outline=(220, 226, 234), width=1)
            color = (255, 255, 255) if t > 0.55 else (30, 40, 55)
            d.text((x0 + cell // 2, y0 + cell // 2), str(v),
                   fill=color, font=f_cell, anchor="mm")

    d.text((left + cell * 2, top + cell * 4 + 16), "预测类别",
           fill=(90, 100, 115), font=font(13), anchor="ma")
    d.text((22, top + cell * 2), "真值", fill=(90, 100, 115), font=font(13),
           anchor="mm")
    img.save(out / "val_confusion_m3.png")

    print("rows", len(rows))
    print("matrix", matrix)
    print("saved", out / "val_predictions_m3.csv")
    print("saved", out / "val_confusion_m3.png")
    print("saved", out / "val_per_class_m3.md")
    ins.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
