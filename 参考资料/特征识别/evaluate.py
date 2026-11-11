# -*- coding: utf-8 -*-
"""评估：在 val（以及可选的 test_unlabeled）上跑一遍，出报告。

用法
----
    python evaluate.py                 # 评 val
    python evaluate.py --split val     # 同上
    python evaluate.py --split test_unlabeled   # 只出预测分布（没有标签）

报告口径（重要，答辩时按这个说）
------------------------------
1. **四类准确率**（第一指标）：预测类别 == 真实类别
2. **缺陷处置率**：真缺陷中「没有拿到 PASS」的比例（目标 100%）
3. **漏检数**：真缺陷被判 PASS 的个数（这是最贵的错误）
4. **误剔数**：正常品被判 REJECT 的个数（浪费 + 复检压力）
5. **复检率**：进 RECHECK 的比例（人工负荷）
6. 按 severity 的召回（mild 最难）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import CLASSES                       # noqa: E402
from pipeline import Inspector                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_val_rows(cfg: dict, split: str):
    csv_path = os.path.join(cfg["dataset_root"], cfg["labels_csv"])
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["split"] == split]


def evaluate(split: str = "val", write_report: bool = True) -> dict:
    ins = Inspector.from_config(recorder=None, keep_debug=False)   # 评估不写 records.csv
    cfg = ins.cfg
    rows = load_val_rows(cfg, split)
    if not rows:
        raise SystemExit(f"split={split} 没有样本")

    print(f"开始评估 split={split}，共 {len(rows)} 张 ...")
    t0 = time.time()
    results = []
    for i, r in enumerate(rows, 1):
        path = os.path.join(cfg["dataset_root"], r["relative_path"].replace("/", os.sep))
        res = ins.inspect(path)
        res.frame_id = r["file_name"]
        results.append((r, res))
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  ({time.time() - t0:.1f}s)")

    # ---------------- 统计 ----------------
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)     # 行=真值，列=预测
    sev_stat = {}
    for r, res in results:
        t_idx = CLASSES.index(r["class_name"])
        p_idx = CLASSES.index(res.pred) if res.pred in CLASSES else 0
        cm[t_idx, p_idx] += 1
        k = r["severity"]
        d = sev_stat.setdefault(k, {"n": 0, "correct": 0, "not_pass": 0})
        d["n"] += 1
        d["correct"] += int(res.pred == r["class_name"])
        d["not_pass"] += int(res.action != "PASS")

    n = len(results)
    n_correct = int(np.trace(cm))
    acc = n_correct / n

    # 缺陷 / 正常 视角
    defect = [(r, res) for r, res in results if r["class_name"] != "normal"]
    normal = [(r, res) for r, res in results if r["class_name"] == "normal"]
    defect_pass = [r["file_name"] for r, res in defect if res.action == "PASS"]
    normal_reject = [r["file_name"] for r, res in normal if res.action == "REJECT"]
    defect_recheck = [r["file_name"] for r, res in defect if res.action == "RECHECK"]
    normal_recheck = [r["file_name"] for r, res in normal if res.action == "RECHECK"]

    summary = {
        "split": split,
        "n": n,
        "n_classes": len(CLASSES),
        "class_accuracy": round(acc, 4),
        "n_correct": n_correct,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_note": "行 = 真实类别，列 = 预测类别；顺序 " + ", ".join(CLASSES),
        "defect_total": len(defect),
        "defect_disposition_rate": round((len(defect) - len(defect_pass)) / max(len(defect), 1), 4),
        "defect_pass_filenames": defect_pass,
        "defect_recheck": len(defect_recheck),
        "normal_total": len(normal),
        "normal_false_reject_filenames": normal_reject,
        "normal_recheck": len(normal_recheck),
        "recheck_rate": round((len(defect_recheck) + len(normal_recheck)) / n, 4),
        "false_alarm_rate": round(len(normal_reject) / max(len(normal), 1), 4),
        "by_severity": {k: {**v, "accuracy": round(v["correct"] / v["n"], 4),
                            "disposition": round(v["not_pass"] / v["n"], 4)}
                        for k, v in sev_stat.items()},
        "avg_elapsed_ms": round(float(np.mean([res.elapsed_ms for _, res in results])), 2),
        "thresholds": {"feature": ins.cfg["thresholds"],
                       "decision": {k: v for k, v in ins.cfg["decision"].items()
                                    if not k.startswith("_")},
                       "classifier": ins.cfg.get("model", {})},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 错例清单
    errors = []
    for r, res in results:
        if res.pred != r["class_name"]:
            f = res.features
            errors.append({
                "file": r["file_name"], "true": r["class_name"], "pred": res.pred,
                "action": res.action, "score": round(res.score, 4),
                "conf": round(res.conf, 4),
                "notch": round(f.notch, 4), "dark": round(f.dark, 4),
                "purity": round(f.purity, 2), "peak": round(f.peak, 0),
            })

    # ---------------- 打印 ----------------
    print("\n" + "=" * 68)
    print(f"四类准确率：{acc:.4f}（{n_correct}/{n}）")
    print("混淆矩阵（行=真值，列=预测，顺序 " + "/".join(CLASSES) + "）：")
    head = " " * 10 + "".join(f"{c:>10s}" for c in CLASSES)
    print(head)
    for i, c in enumerate(CLASSES):
        print(f"{c:>10s}" + "".join(f"{v:>10d}" for v in cm[i]))
    print(f"缺陷处置率（未拿到 PASS 的比例）：{summary['defect_disposition_rate']:.4f}"
          f"（{len(defect) - len(defect_pass)}/{len(defect)}）")
    print(f"  漏检（缺陷被判 PASS）：{len(defect_pass)} 张 {defect_pass}")
    print(f"  缺陷进复检：{len(defect_recheck)} 张")
    print(f"正常品误剔（被判 REJECT）：{len(normal_reject)} 张 {normal_reject}")
    print(f"正常品进复检：{len(normal_recheck)} 张；总复检率 {summary['recheck_rate']:.2%}")
    print(f"平均单张耗时：{summary['avg_elapsed_ms']:.1f} ms")
    print("按严重程度：")
    for k, v in summary["by_severity"].items():
        print(f"  {k:7s} n={v['n']:3d} 四类准确 {v['accuracy']:.3f} 处置率 {v['disposition']:.3f}")
    if errors:
        print(f"\n错例 {len(errors)} 张（类别错但处置可能仍正确）：")
        for e in errors:
            print(f"  {e['file']}: 真 {e['true']} -> 判 {e['pred']} ({e['action']}, "
                  f"score {e['score']}, conf {e['conf']})")
    print("=" * 68)

    # ---------------- 写盘 ----------------
    if write_report:
        out_dir = os.path.join(HERE, ins.cfg["paths"]["out_dir"])
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"eval_report_{split}.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, f"errors_{split}.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(errors[0].keys()) if errors else ["file"])
            w.writeheader()
            w.writerows(errors)
        with open(os.path.join(out_dir, f"eval_report_{split}.md"), "w", encoding="utf-8") as f:
            f.write(render_markdown(summary, errors))
        print(f"报告已写入 {out_dir}")

    return summary


def render_markdown(s: dict, errors: list) -> str:
    """把结果写成一段可以直接贴进报告的 markdown。"""
    lines = [f"# 评估报告（split = {s['split']}）", "",
             f"- 样本数：{s['n']}",
             f"- **四类准确率：{s['class_accuracy']:.2%}**（{s['n_correct']}/{s['n']}）",
             f"- **缺陷处置率：{s['defect_disposition_rate']:.2%}**（{s['defect_total']} 张缺陷中"
             f" {len(s['defect_pass_filenames'])} 张被判 PASS）",
             f"- 正常品误剔：{len(s['normal_false_reject_filenames'])} 张；复检率：{s['recheck_rate']:.2%}",
             f"- 平均单张耗时：{s['avg_elapsed_ms']} ms",
             "", "## 四类混淆矩阵", "",
             "| 真值＼预测 | " + " | ".join(CLASSES) + " |",
             "|---|" + "---|" * len(CLASSES)]
    for i, c in enumerate(CLASSES):
        lines.append(f"| **{c}** | " + " | ".join(str(v) for v in s["confusion_matrix"][i]) + " |")
    lines += ["", "## 按严重程度", "", "| severity | n | 四类准确率 | 缺陷处置率 |", "|---|---|---|---|"]
    for k, v in s["by_severity"].items():
        lines.append(f"| {k} | {v['n']} | {v['accuracy']:.2%} | {v['disposition']:.2%} |")
    lines += ["", "## 阈值（全部由 train 定）", "",
              "| 参数 | 值 |", "|---|---|"]
    for k, v in s["thresholds"]["feature"].items():
        lines.append(f"| {k} | {v} |")
    for k, v in s["thresholds"]["decision"].items():
        lines.append(f"| {k} | {v} |")
    lines.append(f"| 选中分类器 | {s['thresholds']['classifier'].get('selected_classifier', '-')} "
                 f"(CV {s['thresholds']['classifier'].get('cv_accuracy', '-')}) |")
    if errors:
        lines += ["", "## 错例（类别错、处置可能仍对）", "",
                  "| 文件 | 真值 | 预测 | 处置 | 异常分 | 置信度 |", "|---|---|---|---|---|---|"]
        for e in errors:
            lines.append(f"| {e['file']} | {e['true']} | {e['pred']} | {e['action']} | "
                         f"{e['score']} | {e['conf']} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="在 val/test_unlabeled 上评估")
    ap.add_argument("--split", default="val", choices=["val", "train", "test_unlabeled"])
    args = ap.parse_args()
    evaluate(args.split)
