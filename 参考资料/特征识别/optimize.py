# -*- coding: utf-8 -*-
"""阈值优化：把 low / high / conf_min 调到这份数据上的最优位置。

优化的目标（按优先级）
---------------------
1. 缺陷处置率必须 100%、漏检必须 0        —— 安全底线
2. 正常品误剔必须 0                        —— 成本底线
3. 四类准确率尽量高（注：准确率只取决于分类器，与阈值无关，见下文）
4. 复检率尽量低                            —— 人力成本
5. 在上述都满足的前提下，让阈值离两群样本尽量远（最大裕度），
   这样未来数据轻微漂移时不会立刻误判。

**关键结论（答辩要点）**
四类准确率 = argmax(概率) 与真值的比较，和 low/high/conf_min 没有任何关系。
阈值只影响“处置”（PASS / RECHECK / REJECT）和“未知缺陷”标记：
    · 旧的 high = 0.9868 比很多缺陷的分数还高 → 这些缺陷只能进复检；
    · 而 train 上缺陷最低分是 0.8647、正常最高分是 0.1879，中间 0.19–0.86 是空的。
所以正确做法是把复检带（RECHECK）放进这条“空档带”里，而不是压在缺陷分布上。

运行：
    python optimize.py          # 打印分析 + 写入 config.json
    python optimize.py --dry    # 只看分析，不改文件
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import CLASSES                      # noqa: E402
from features import measure_image                  # noqa: E402
from pipeline import Inspector                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------- 数据
def load_scored(ins: Inspector) -> List[Dict]:
    """对 train+val 的每张图算一遍 (特征, 异常分, 概率)，结果缓存到 out/score_table.json。"""
    root = ins.cfg["dataset_root"]
    cache = os.path.join(HERE, "out", "score_table.json")
    rows = []
    with open(os.path.join(root, ins.cfg["labels_csv"]), "r", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    table = []
    for r in rows:
        p = os.path.join(root, r["relative_path"].replace("/", os.sep))
        fv = measure_image(p, ins.cfg["features"], keep_debug=False)
        feat = fv.as_features()
        score, z = ins.baseline.score(feat)
        proba = ins.model.predict_proba([feat])[0]
        table.append({
            "split": r["split"], "cls": r["class_name"], "path": p,
            "score": float(score),
            "pred": CLASSES[int(np.argmax(proba))],
            "conf": float(max(proba)),
            "p_normal": float(proba[CLASSES.index("normal")]),
        })
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(table, fh, ensure_ascii=False, indent=1)
    return table


# ------------------------------------------------------------------ 评估函数
def apply_decision(t: Dict, low: float, high: float, conf_min: float,
                   pass_normal_prob: float = 0.9) -> str:
    """复刻 decision.decide 的处置逻辑（只关心 action）。"""
    if t["score"] < low:
        if t["pred"] == "normal" and t["p_normal"] >= pass_normal_prob:
            return "PASS"
        return "RECHECK"
    if t["score"] < high:
        return "RECHECK"
    return "REJECT"


def evaluate(table: List[Dict], split: str, low: float, high: float,
             conf_min: float) -> Dict:
    sub = [t for t in table if t["split"] == split]
    actions = [(t, apply_decision(t, low, high, conf_min)) for t in sub]
    n = len(sub)
    correct = sum(1 for t, _ in actions if t["pred"] == t["cls"])
    defects = [(t, a) for t, a in actions if t["cls"] != "normal"]
    normals = [(t, a) for t, a in actions if t["cls"] == "normal"]
    leak = [t["path"] for t, a in defects if a == "PASS"]
    false_rej = [t["path"] for t, a in normals if a == "REJECT"]
    recheck = [t["path"] for t, a in actions if a == "RECHECK"]
    # “未知缺陷”：异常分判为缺陷，但分类器说正常 / 或很没把握
    unknown = [t["path"] for t, a in actions
               if a == "REJECT" and (t["pred"] == "normal" or t["conf"] < conf_min)]
    return {
        "n": n,
        "accuracy": correct / n,
        "defect_disposition": 1 - len(leak) / max(len(defects), 1),
        "n_leak": len(leak),
        "n_false_reject": len(false_rej),
        "n_recheck": len(recheck),
        "recheck_rate": len(recheck) / n,
        "n_reject": sum(1 for _, a in actions if a == "REJECT"),
        "n_pass": sum(1 for _, a in actions if a == "PASS"),
        "n_unknown": len(unknown),
        "unknown_paths": [os.path.basename(x) for x in unknown],
    }


# ------------------------------------------------------------------ 优化器
POLICIES = {
    # 名字 -> (low 在空档带里的位置, high 在空档带里的位置)
    "risk_averse": (0.05, 0.95),   # 复检带最宽：low 贴着正常包络，high 贴着缺陷包络
    "balanced": (0.30, 0.70),      # 两侧裕度对称（旧默认）
}


def optimize(table: List[Dict], policy: str = "risk_averse") -> Tuple[float, float, Dict]:
    """只用 train 统计量定阈值，返回 (low, high, 分析信息)。

    空档带 = [train 正常品最高分, train 缺陷最低分]（这份数据是 0.1879 ~ 0.8647）。
    `policy` 决定把 low / high 放在带子里的哪个位置：

    · risk_averse（默认）：low 压到接近正常包络（5%）、high 抬到接近缺陷包络（95%）。
      复检带最宽 —— “拿不准就交人工”，但判明确的缺陷也不会被硬判成剔除。
    · balanced：两侧各留 30% / 70%，裕度对称，机器自己拍板的比例更高。
    """
    tr = [t for t in table if t["split"] == "train"]
    normal_scores = np.array([t["score"] for t in tr if t["cls"] == "normal"])
    defect_scores = np.array([t["score"] for t in tr if t["cls"] != "normal"])

    a = float(normal_scores.max())          # 正常品最高分（保守取最大值，不用分位数）
    b = float(defect_scores.min())          # 缺陷最低分
    gap = b - a
    if gap <= 0:                            # 理论上不该发生；退化为分位数
        a = float(np.percentile(normal_scores, 99.5))
        b = float(np.percentile(defect_scores, 0.5))
        gap = max(b - a, 0.02)

    frac_lo, frac_hi = POLICIES[policy]
    low = round(a + frac_lo * gap, 4)
    high = round(a + frac_hi * gap, 4)

    # conf_min：train 上“判对的样本”置信度的 5% 分位。
    # 为什么用 5% 分位而不是 10%：这份数据的置信度分布很极端（多数为 1.0），
    # 10% 分位会抬到 0.99 以上，于是大量判对的样本也被标成“未知缺陷”，
    # 5% 分位（≈0.98）刚好落在“判对的边界”上，判错的样本（0.957 以下）会被标出来。
    confs = [t["conf"] for t in tr if t["pred"] == t["cls"]]
    conf_min = round(float(max(0.50, np.percentile(confs, 5))), 3)

    info = {
        "train_normal_max": round(a, 4),
        "train_defect_min": round(b, 4),
        "gap": round(gap, 4),
        "margin_low": round(low - a, 4),      # low 距离正常品最高分多远
        "margin_high": round(b - high, 4),    # high 距离缺陷最低分多远
        "train_normal_p50": round(float(np.median(normal_scores)), 4),
        "train_normal_p95": round(float(np.percentile(normal_scores, 95)), 4),
        "train_defect_p50": round(float(np.median(defect_scores)), 4),
        "conf_min_p10": conf_min,
        "n_train": len(tr),
        "policy": policy,
        "band_frac": [frac_lo, frac_hi],
    }
    return low, high, info


def main() -> None:
    ap = argparse.ArgumentParser(description="阈值优化（只用 train 的位置，val 只作验证）")
    ap.add_argument("--dry", action="store_true", help="只打印，不写 config.json")
    ap.add_argument("--policy", default="risk_averse", choices=list(POLICIES),
                    help="阈值在空档带里的位置：risk_averse（默认，复检带最宽）/ balanced（对称裕度）")
    args = ap.parse_args()

    ins = Inspector.from_config(recorder=None)
    table = load_scored(ins)

    low_new, high_new, info = optimize(table, args.policy)
    conf_new = info["conf_min_p10"]
    pnp = ins.cfg["decision"].get("pass_normal_prob", 0.9)

    # “旧”阈值从 model/thresholds.json 读（那是 tune.py 的产物，config.json 会被本脚本覆盖）
    old = {"low": 0.8214, "high": 0.9868, "conf_min": 0.95}   # tune.py 原始结果的兜底值
    tp = os.path.join(HERE, "model", "thresholds.json")
    if os.path.exists(tp):
        try:
            with open(tp, "r", encoding="utf-8") as fh:
                old = json.load(fh)["decision"]
        except Exception:                                     # noqa: BLE001
            pass
    low_old, high_old, conf_old = old["low"], old["high"], old["conf_min"]

    print("=" * 72)
    print("train 上的分数分布（阈值只能看这一半的数据）")
    print(f"  正常品：中位 {info['train_normal_p50']:.4f}  95 分位 {info['train_normal_p95']:.4f}  "
          f"最高 {info['train_normal_max']:.4f}")
    print(f"  缺陷  ：中位 {info['train_defect_p50']:.4f}  最低 {info['train_defect_min']:.4f}")
    print(f"  → 两组之间有一条 {info['gap']:.4f} 宽的空档（{info['train_normal_max']:.3f} ~ "
          f"{info['train_defect_min']:.3f}）")
    print()
    print(f"新阈值（策略 = {args.policy}，复检带占空档带的 "
          f"{round((info['band_frac'][1] - info['band_frac'][0]) * 100)}%）")
    print(f"  low  = {low_new:.4f}   （离正常品最高分 {info['margin_low']:.4f}）")
    print(f"  high = {high_new:.4f}   （离缺陷最低分 {info['margin_high']:.4f}）")
    print(f"  conf_min = {conf_new:.3f}（train 判对样本置信度的 5% 分位）")
    print()

    head = (f"{'阈值方案':>14} | {'split':<5} | {'准确率':>7} | {'处置率':>7} | {'漏检':>4} | "
            f"{'误剔':>4} | {'复检':>4} | {'剔除':>4} | {'放行':>4} | {'未知':>4}")
    print(head)
    print("-" * 104)
    best = {}
    new_label = {"risk_averse": "新（风险厌恶）", "balanced": "新（均衡）"}[args.policy]
    for name, (lo, hi, cf) in (("旧（tune.py）", (low_old, high_old, conf_old)),
                               (new_label, (low_new, high_new, conf_new))):
        for split in ("train", "val"):
            m = evaluate(table, split, lo, hi, cf)
            best[(name, split)] = m
            print(f"{name:>14} | {split:<5} | {m['accuracy']:>7.2%} | "
                  f"{m['defect_disposition']:>7.2%} | {m['n_leak']:>4} | "
                  f"{m['n_false_reject']:>4} | {m['n_recheck']:>4} | "
                  f"{m['n_reject']:>4} | {m['n_pass']:>4} | {m['n_unknown']:>4}")
    print()

    old_val, new_val = best[("旧（tune.py）", "val")], best[(new_label, "val")]
    print("val 上的变化：")
    print(f"  四类准确率   {old_val['accuracy']:.2%} → {new_val['accuracy']:.2%}（不变，阈值不参与分类）")
    print(f"  缺陷处置率   {old_val['defect_disposition']:.2%} → {new_val['defect_disposition']:.2%}")
    print(f"  复检         {old_val['n_recheck']} 张 → {new_val['n_recheck']} 张"
          f"（{old_val['recheck_rate']:.1%} → {new_val['recheck_rate']:.1%}）")
    if new_val["unknown_paths"]:
        print(f"  标“未知缺陷”：{new_val['n_unknown']} 张 {new_val['unknown_paths']}")
    print()

    if args.dry:
        print("（--dry：没有修改任何文件）")
        return

    # 写回配置：默认用新阈值；旧的留作“演示预设”，方便答辩时展示 RECHECK 分支
    cfg_path = os.path.join(HERE, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["decision"].update({"low": low_new, "high": high_new, "conf_min": conf_new,
                            "policy": args.policy, "band": list(info["band_frac"])})
    # 两套阈值都留成预设，方便现场对比“保守 vs 均衡”
    a_b, gap_p = info["train_normal_max"], info["gap"]
    cfg["presets"] = {
        args.policy: {"low": low_new, "high": high_new, "conf_min": conf_new},
        "balanced": {"low": round(a_b + 0.30 * gap_p, 4),
                     "high": round(a_b + 0.70 * gap_p, 4), "conf_min": conf_new},
        "demo_recheck": {"low": round(low_old, 4), "high": round(high_old, 4),
                         "conf_min": round(conf_old, 3)},
        "course": cfg["presets"].get("course", {"low": 0.5, "high": 0.9}),
        "starter_kit": cfg["presets"].get("starter_kit", {"low": 0.2, "high": 0.8}),
    }
    cfg["decision"]["_说明"] = ("串联决策阈值。low/high 由 optimize.py 按“最大裕度”定在 "
                                "train 的空档带里；conf_min 只决定是否标“未知缺陷”。")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)

    # 存一份分析报告
    report = {"chosen": {"low": low_new, "high": high_new, "conf_min": conf_new},
              "previous": {"low": low_old, "high": high_old, "conf_min": conf_old},
              "analysis": info,
              "metrics": {f"{k[0]}|{k[1]}": v for k, v in best.items()}}
    with open(os.path.join(HERE, "out", "threshold_opt.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(f"已写入 {cfg_path}")
    print(f"已保存 out/threshold_opt.json")


if __name__ == "__main__":
    main()
