# -*- coding: utf-8 -*-
"""H1 演示脚本：连续跑 N 只瓶盖，输出记录、统计和一段可直接放进报告的文字。

用途
----
没有真实产线时，用 val 里的图片按类别均衡抽取，模拟“光电触发 -> 相机取图 ->
判断 -> 执行 -> 记录”的连续节拍；每只都会经过真实的 Inspector 和模拟执行器。

用法（在 H1原型/ 下）：
    ../支持文件/Python环境/Scripts/python.exe 工具/demo_run.py --samples 24 --preset demo_recheck
输出：
    运行结果/demo_records.csv        本次 N 只的完整记录
    运行结果/demo_run.json           统计
    运行结果/demo_run.md             报告可引用的表格
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import config as C
from core.pipeline import Inspector
from core.types import CLASS_CN, CLASSES
from devices.actuators import build_dispatcher
from devices.base import CsvRecorder, RECORD_FIELDS


def balanced_paths(cfg, split: str, per_class: int) -> list:
    paths = C.split_images(cfg, split)
    labels = C.load_labels(cfg)
    buckets = defaultdict(list)
    for p in paths:
        y = labels.get(os.path.basename(p))
        if y in CLASSES:
            buckets[y].append(p)
    picked = []
    for c in CLASSES:
        picked.extend(buckets[c][:per_class])
    # 轮转，避免同一类连续出现，演示更像真实产线
    out = []
    for i in range(per_class):
        for c in CLASSES:
            if i < len(buckets[c]):
                out.append(buckets[c][i])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=24, help="总样本数（会按类别轮转）")
    ap.add_argument("--preset", default="demo_recheck")
    ap.add_argument("--cycle-ms", type=float, default=300, help="仿真节拍，演示时可调快")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    cfg = C.load_config()
    per_class = max(1, args.samples // len(CLASSES))
    paths = balanced_paths(cfg, args.split, per_class)
    paths = paths[: args.samples]
    if not paths:
        print("NO_IMAGES")
        return 1

    preset = cfg.get("presets", {}).get(args.preset, {})
    low = float(preset.get("low", cfg["decision"]["low"]))
    high = float(preset.get("high", cfg["decision"]["high"]))
    conf = float(preset.get("conf_min", cfg["decision"]["conf_min"]))

    out_dir = C.out_dir(cfg)
    rec_path = out_dir / "demo_records.csv"
    if rec_path.exists():
        os.replace(rec_path, out_dir / f"demo_records.old_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    recorder = CsvRecorder(str(rec_path), RECORD_FIELDS)
    ins = Inspector(cfg, ROOT, build_dispatcher(cfg, "sim"), recorder,
                    feedback=None, keep_debug=False)
    ins.update_decision(low, high, conf)
    ok = ins.deep.load()
    if ok:
        ins.deep.warmup()
    print(f"deep={'OK' if ok else 'UNAVAILABLE'} preset={args.preset} "
          f"low={low:.4f} high={high:.4f} conf={conf:.2f} samples={len(paths)}")

    labels = C.load_labels(cfg)
    rows = []
    t_start = time.time()
    for i, p in enumerate(paths, 1):
        label = labels.get(os.path.basename(p), "")
        res = ins.inspect(p, frame_id=os.path.basename(p), label=label,
                          ts_trigger=time.time(), dispatch=True)
        rows.append(res)
        print(f"[{i:02d}/{len(paths)}] {res.frame_id} label={label or '-':7s} "
              f"pred={res.pred:7s} action={res.action:7s} risk={res.risk:.3f} "
              f"anom={res.anomaly_score:.3f} deep={res.deep.defect_score:.3f} "
              f"{res.elapsed_ms:.0f}ms act={'ok' if res.act and res.act.ok else 'fail'}")
        if args.cycle_ms > 0 and i < len(paths):
            time.sleep(args.cycle_ms / 1000.0)

    elapsed = time.time() - t_start
    acts = Counter(r.action for r in rows)
    unknown = sum(1 for r in rows if r.fusion and r.fusion.is_unknown)
    act_fail = sum(1 for r in rows if r.act and not r.act.ok)
    labeled = [r for r in rows if r.label]
    correct = sum(1 for r in labeled if r.label == r.pred)
    normal = [r for r in labeled if r.label == "normal"]
    defects = [r for r in labeled if r.label != "normal"]
    leak = sum(1 for r in defects if r.action == "PASS")
    false_reject = sum(1 for r in normal if r.action == "REJECT")

    summary = {
        "preset": args.preset,
        "samples": len(rows),
        "labels": {c: sum(1 for r in rows if r.label == c) for c in CLASSES},
        "actions": {a: acts.get(a, 0) for a in ("PASS", "RECHECK", "REJECT")},
        "unknown": unknown,
        "act_fail": act_fail,
        "accuracy": round(correct / max(len(labeled), 1), 4),
        "correct": correct,
        "labeled": len(labeled),
        "leak": leak,
        "false_reject": false_reject,
        "avg_ms": round(sum(r.elapsed_ms for r in rows) / len(rows), 1),
        "wall_seconds": round(elapsed, 2),
        "cycle_ms": args.cycle_ms,
        "deep_available": bool(rows and rows[0].deep and rows[0].deep.available),
        "deep_device": ins.deep.device_text(),
        "thresholds": {"low": low, "high": high, "conf_min": conf},
    }
    (out_dir / "demo_run.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        f"# H1 连续演示（{len(rows)} 只，preset={args.preset}）",
        "",
        f"- 设备/模型：{summary['deep_device']}（深度模型{'可用' if summary['deep_available'] else '不可用'}）",
        f"- 阈值：low={low:.4f}，high={high:.4f}，conf_min={conf:.2f}",
        f"- 处置：PASS {acts.get('PASS', 0)} / RECHECK {acts.get('RECHECK', 0)} / REJECT {acts.get('REJECT', 0)}",
        f"- 有标签准确率：{correct}/{len(labeled)} = {summary['accuracy']*100:.1f}%",
        f"- 漏检：{leak}；正常误剔：{false_reject}；未知缺陷标记：{unknown}；执行失败：{act_fail}",
        f"- 仿真节拍：{args.cycle_ms:.0f} ms/只；平均端到端耗时：{summary['avg_ms']:.1f} ms/只（不含节拍）；总墙钟：{summary['wall_seconds']:.2f} s",
        "",
        "| 序号 | 文件 | 真值 | 预测 | 风险 | 异常分 | 深度缺陷分 | 处置 | 执行 | 耗时ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, 1):
        md.append(
            f"| {i} | {r.frame_id} | {CLASS_CN.get(r.label, '-')} | {CLASS_CN.get(r.pred, r.pred)} | "
            f"{r.risk:.3f} | {r.anomaly_score:.3f} | "
            f"{r.deep.defect_score:.3f} | {r.action} | "
            f"{'OK' if r.act and r.act.ok else 'FAIL'} | {r.elapsed_ms:.0f} |")
    md += ["", "> 课程数据为教学合成图；以上只证明流程闭环与统计方法，不代表真实工业性能。", ""]
    (out_dir / "demo_run.md").write_text("\n".join(md), encoding="utf-8")

    recorder.close()
    ins.close()
    print("DONE", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
