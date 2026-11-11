# -*- coding: utf-8 -*-
"""离线评估：在同一份 val 上比较 M0/M1/M2/M3，按 H1 的优先级选默认融合方式。

指标顺序（用户定的优先级）
    1. 漏检 leak        ：缺陷被判 PASS 的数量（必须为 0）
    2. 四类准确率       ：预测类别 == 真值
    3. 正常误剔         ：正常品被判 REJECT
    4. 平均耗时         ：单张端到端（不含执行等待；离线模式）
    5. 复检率           ：RECHECK 占比

用法（在 H1原型/ 下）：
    ../支持文件/Python环境/Scripts/python.exe 工具/eval_fusion.py
    ../支持文件/Python环境/Scripts/python.exe 工具/eval_fusion.py --split val --modes M0 M1 M2 M3
输出：
    运行结果/fusion_eval.json   机器可读
    运行结果/fusion_eval.md     可直接粘进报告
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import config as C                      # noqa: E402
from core.pipeline import Inspector               # noqa: E402
from devices.actuators import build_dispatcher    # noqa: E402


def evaluate(ins: Inspector, paths, labels, mode: str) -> dict:
    ins.set_fusion_mode(mode)
    rows = []
    t0 = time.time()
    for p in paths:
        r = ins.inspect(p, dispatch=False)
        rows.append(r)
    dt = time.time() - t0

    total = len(rows)
    correct = leak = false_reject = 0
    defects = normals = 0
    acts = Counter()
    unknown = 0
    for r in rows:
        true = labels.get(os.path.basename(r.path) or r.frame_id, "")
        acts[r.action] += 1
        if r.fusion and r.fusion.is_unknown:
            unknown += 1
        if not true:
            continue
        if r.pred == true:
            correct += 1
        if true == "normal":
            normals += 1
            if r.action == "REJECT":
                false_reject += 1
        else:
            defects += 1
            if r.action == "PASS":
                leak += 1

    return {
        "mode": mode,
        "split": "val",
        "total": total,
        "labeled": normals + defects,
        "defects": defects,
        "normals": normals,
        "accuracy": round(correct / max(normals + defects, 1), 4),
        "correct": correct,
        "leak": leak,
        "leak_rate": round(leak / max(defects, 1), 4),
        "defect_disposition_rate": round((defects - leak) / max(defects, 1), 4),
        "false_reject": false_reject,
        "false_reject_rate": round(false_reject / max(normals, 1), 4),
        "recheck": acts.get("RECHECK", 0),
        "recheck_rate": round(acts.get("RECHECK", 0) / max(total, 1), 4),
        "pass": acts.get("PASS", 0),
        "reject": acts.get("REJECT", 0),
        "unknown": unknown,
        "avg_ms": round(dt * 1000.0 / max(total, 1), 2),
        "deep_available": bool(rows and rows[0].deep and rows[0].deep.available),
        "deep_device": ins.deep.device_text(),
    }


def rank_key(m: dict) -> tuple:
    """按用户优先级排序：漏检 -> 准确率 -> 误剔 -> 耗时 -> 复检。"""
    return (m["leak"], -m["accuracy"], m["false_reject"], m["avg_ms"], m["recheck_rate"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train", "test_unlabeled"])
    ap.add_argument("--modes", nargs="*", default=["M0", "M1", "M2", "M3"])
    ap.add_argument("--out", default=os.path.join(ROOT, "运行结果"))
    args = ap.parse_args()

    cfg = C.load_config()
    paths = C.split_images(cfg, args.split)
    labels = C.load_labels(cfg)
    if not paths:
        print("没有图片，检查 dataset_root")
        return 1

    print(f"split={args.split} images={len(paths)} deep={cfg['deep'].get('enabled')}")
    ins = Inspector(cfg, C.PROJECT_ROOT, build_dispatcher(cfg, "offline"),
                    recorder=None, feedback=None, keep_debug=False)
    if cfg.get("deep", {}).get("enabled", True):
        print("加载深度模型…")
        ok = ins.deep.load()
        print("  深度模型:", "OK" if ok else f"不可用：{ins.deep.error}")
        if ok:
            ins.deep.warmup()

    results = []
    for mode in args.modes:
        m = evaluate(ins, paths, labels, mode)
        results.append(m)
        print(f"{mode}: acc={m['accuracy']:.4f} leak={m['leak']} "
              f"false_reject={m['false_reject']} recheck={m['recheck_rate']:.3f} "
              f"avg={m['avg_ms']:.1f}ms acts=({m['pass']}/{m['recheck']}/{m['reject']})")

    best = min(results, key=rank_key)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "fusion_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "best": best["mode"], "results": results},
                  f, ensure_ascii=False, indent=2)

    lines = [
        f"# 融合方式对比（{args.split}，{len(paths)} 张）", "",
        f"- 深度模型：{results[0]['deep_device']}（{'可用' if results[0]['deep_available'] else '不可用'}）",
        f"- 选择规则：漏检 > 准确率 > 正常误剔 > 耗时 > 复检率",
        f"- 推荐：**{best['mode']}**", "",
        "| 模式 | 准确率 | 漏检 | 缺陷处置率 | 正常误剔 | 复检率 | 平均耗时 | PASS/RECHECK/REJECT | 未知 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in results:
        lines.append(
            f"| {m['mode']} | {m['accuracy']*100:.2f}% ({m['correct']}/{m['labeled']}) | "
            f"{m['leak']} | {m['defect_disposition_rate']*100:.1f}% | {m['false_reject']} | "
            f"{m['recheck_rate']*100:.1f}% | {m['avg_ms']:.1f} ms | "
            f"{m['pass']}/{m['recheck']}/{m['reject']} | {m['unknown']} |"
        )
    lines += ["", "> 说明：课程数据为教学合成图；这里的数字只证明流程与对比方法，不代表真实工业性能。", ""]
    with open(os.path.join(args.out, "fusion_eval.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"推荐 {best['mode']}，结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
