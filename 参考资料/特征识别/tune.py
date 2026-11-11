# -*- coding: utf-8 -*-
"""定阈值 + 选分类器：**只读 train**，val 一张都不看。

产出
----
model/baseline.json     正常品基线（异常分用）
model/classifier.json   选中的分类器
model/thresholds.json   4 个特征阈值 + 异常分阈值 + 置信度阈值
                          并附带“这些数是从 train 的哪些统计量算出来的”，可追溯
out/features_cache.json 全部样本的特征（避免每次重算，约 1 秒/张）

用法
----
    python tune.py              # 计算特征（有缓存则复用）并重新定参
    python tune.py --recalc     # 强制重算特征
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anomaly import Baseline                       # noqa: E402
from classifier import (CLASSES, Model, Standardizer, cross_validate,
                        fit_candidate)             # noqa: E402
from features import measure_image                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(HERE, "config.json")


# ----------------------------------------------------------------------------
# 配置与样本清单
# ----------------------------------------------------------------------------
def load_config(path: str = CFG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict, path: str = CFG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_labels(cfg: dict) -> List[Dict[str, str]]:
    csv_path = os.path.join(cfg["dataset_root"], cfg["labels_csv"])
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def image_path(cfg: dict, row: Dict[str, str]) -> str:
    return os.path.join(cfg["dataset_root"], row["relative_path"].replace("/", os.sep))


# ----------------------------------------------------------------------------
# 特征计算（带缓存）
# ----------------------------------------------------------------------------
def compute_features(cfg: dict, rows: List[Dict[str, str]], cache_path: str,
                     recalc: bool = False) -> Dict[str, Dict[str, float]]:
    """返回 {relative_path: {notch,dark,purity,peak}}。"""
    cache: Dict[str, Dict[str, float]] = {}
    if os.path.exists(cache_path) and not recalc:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    todo = [r for r in rows if r["relative_path"] not in cache]
    if todo:
        t0 = time.time()
        for i, r in enumerate(todo, 1):
            fv = measure_image(image_path(cfg, r), cfg["features"], keep_debug=False)
            cache[r["relative_path"]] = fv.as_features()
            if i % 20 == 0 or i == len(todo):
                print(f"  特征计算 {i}/{len(todo)}  ({time.time() - t0:.1f}s)")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
    return cache


# ----------------------------------------------------------------------------
# 阈值规则
# ----------------------------------------------------------------------------
def _midpoint_threshold(neg: np.ndarray, pos: np.ndarray, prefer: str = "geom") -> Tuple[float, str]:
    """在两群之间取阈值。

    neg：正常品的该特征值；pos：该类缺陷的值（都要“越大越异常”）。
    两群分开时取中间值（几何平均更稳，因为特征跨数量级）；
    若两群有重叠，则改成“在 train 上扫描所有候选切点，取判别最好的那个”——
    这仍然只用 train，不会偷看 val。
    """
    hi_neg = float(np.percentile(neg, 99.5))
    lo_pos = float(np.percentile(pos, 10))
    if lo_pos > hi_neg and hi_neg > 0:
        if prefer == "geom":
            return float(np.sqrt(max(hi_neg, 1e-9) * lo_pos)), "两群分开：几何中点"
        return float((hi_neg + lo_pos) / 2), "两群分开：算术中点"
    # 重叠：在 train 上扫描
    vals = np.unique(np.concatenate([neg, pos]))
    best_t, best_score = float(np.median(pos)), -1.0
    for t in (vals[:-1] + vals[1:]) / 2:
        tp = float((pos > t).mean())
        tn = float((neg <= t).mean())
        score = 0.5 * (tp + tn)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, f"两群重叠：train 扫描最优（平衡准确率 {best_score:.3f}）"


def tune_thresholds(feats: Dict[str, Dict[str, float]], rows: List[Dict[str, str]],
                    cfg: dict) -> Tuple[Dict[str, float], Dict[str, str]]:
    """只用 train 行算阈值。returns (thresholds, 说明)"""
    tr = [r for r in rows if r["split"] == "train"]

    def vals(cls: str, key: str) -> np.ndarray:
        return np.asarray([feats[r["relative_path"]][key] for r in tr if r["class_name"] == cls],
                          dtype=np.float64)

    normal_notch, notch_notch = vals("normal", "notch"), vals("notch", "notch")
    normal_dark = vals("normal", "dark")
    stain_dark = np.concatenate([vals("stain", "dark"), vals("scratch", "dark") * 0.0])
    normal_purity = vals("normal", "purity")
    scratch_purity = vals("scratch", "purity")
    normal_peak = vals("normal", "peak")
    scratch_peak = vals("scratch", "peak")

    t_notch, why_notch = _midpoint_threshold(normal_notch, notch_notch)
    t_dark, why_dark = _midpoint_threshold(normal_dark, vals("stain", "dark"))
    t_purity, why_pur = _midpoint_threshold(normal_purity, scratch_purity)
    # 峰值是“绝对点数”，和纯度配合使用；用划痕的 10% 分位与正常的 99.5% 分位之间取中点
    t_peak, why_peak = _midpoint_threshold(normal_peak, scratch_peak, prefer="arith")
    # 大暗斑快速通道：取污渍 train 的 30% 分位（保证真污渍多半直接命中），但不低于 2.5 倍阈值
    t_dark_big = float(max(2.5 * t_dark, np.percentile(vals("stain", "dark"), 30)))

    thr = {"notch": t_notch, "dark": t_dark, "purity": t_purity,
           "peak": t_peak, "dark_big": t_dark_big}
    why = {"notch": why_notch, "dark": why_dark, "purity": why_pur,
           "peak": why_peak, "dark_big": "污渍 train 的 30% 分位与 2.5×dark 阈值取大"}
    # 记录用到的 train 统计量，报告里可追溯
    why["_stats"] = (
        f"train 各群中位数：凹陷率 normal {np.median(normal_notch):.4f}/notch {np.median(notch_notch):.4f}；"
        f"暗斑 normal {np.median(normal_dark):.4f}/stain {np.median(vals('stain', 'dark')):.4f}；"
        f"纯度 normal {np.median(normal_purity):.2f}/scratch {np.median(scratch_purity):.2f}；"
        f"峰值 normal {np.median(normal_peak):.0f}/scratch {np.median(scratch_peak):.0f}")
    return thr, why


def tune_anomaly_thresholds(baseline: Baseline, feats: Dict[str, Dict[str, float]],
                            rows: List[Dict[str, str]], z_cap: float
                            ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """用 train 定 low / high。

    low  —— 尽量低，但要求「train 里所有缺陷的异常分都 ≥ low」（零漏检优先），
             同时保证正常品的误剔（进 RECHECK）比例可接受。
    high —— 只影响“进 REJECT 还是 RECHECK”，取 train 缺陷分数的 25% 分位，
             再用 (low + 0.05) 兜底，保证 high > low。
    """
    tr = [r for r in rows if r["split"] == "train"]
    s_norm = np.asarray([baseline.score(feats[r["relative_path"]])[0]
                         for r in tr if r["class_name"] == "normal"])
    s_def = np.asarray([baseline.score(feats[r["relative_path"]])[0]
                        for r in tr if r["class_name"] != "normal"])

    # 零漏检：取缺陷分数的最小值再留 5% 余量
    low = float(max(0.05, min(0.95, float(s_def.min()) * 0.95)))
    high = float(max(low + 0.05, float(np.percentile(s_def, 25))))
    high = float(min(high, 0.99))
    info = {
        "train_normal_p50": float(np.median(s_norm)),
        "train_normal_p95": float(np.percentile(s_norm, 95)),
        "train_normal_recheck_rate": float((s_norm >= low).mean()),
        "train_defect_min": float(s_def.min()),
        "train_defect_p50": float(np.median(s_def)),
        "train_defect_reject_rate": float((s_def >= high).mean()),
    }
    return {"low": low, "high": high}, info


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main(recalc: bool = False) -> None:
    cfg = load_config()
    rows = load_labels(cfg)
    out_dir = os.path.join(HERE, cfg["paths"]["out_dir"])
    model_dir = os.path.join(HERE, cfg["paths"]["model_dir"])
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 68)
    print("第 1 步：计算全部样本的特征（只算一次，之后走缓存）")
    cache = compute_features(cfg, rows, os.path.join(out_dir, "features_cache.json"), recalc)

    print("=" * 68)
    print("第 2 步：只用 train 定 4 个特征阈值")
    thr, why = tune_thresholds(cache, rows, cfg)
    for k in ("notch", "dark", "purity", "peak", "dark_big"):
        print(f"  {k:9s} = {thr[k]:.5f}   [{why[k]}]")
    print("  " + why["_stats"])

    print("=" * 68)
    print("第 3 步：只用 train 的正常品估计异常分基线")
    tr_rows = [r for r in rows if r["split"] == "train"]
    tr_normal = [cache[r["relative_path"]] for r in tr_rows if r["class_name"] == "normal"]
    baseline = Baseline.fit(tr_normal, thresholds=thr, z_cap=cfg["anomaly"]["z_cap"])
    print(f"  正常品 {baseline.n_samples} 张；中位数：", {k: round(v, 4) for k, v in baseline.medians.items()})
    print("  打分尺度 scale：", {k: round(v, 4) for k, v in baseline.scales.items()})

    print("=" * 68)
    print("第 4 步：只用 train 定异常分阈值 low / high（零漏检优先）")
    an_thr, an_info = tune_anomaly_thresholds(baseline, cache, rows, cfg["anomaly"]["z_cap"])
    for k, v in an_thr.items():
        print(f"  {k:5s} = {v:.4f}")
    print("  train 诊断：", {k: round(v, 4) for k, v in an_info.items()})

    print("=" * 68)
    print("第 5 步：train 内部 5 折交叉验证选分类器（不看 val）")
    tr_feats = [cache[r["relative_path"]] for r in tr_rows]
    y = np.asarray([CLASSES.index(r["class_name"]) for r in tr_rows], dtype=np.int64)
    cv = {}
    for kind in ("rule", "centroid", "logreg", "tree", "hybrid"):
        mean_acc, folds = cross_validate(kind, tr_feats, y, thr, k=5, seed=0)
        cv[kind] = mean_acc
        print(f"  {kind:9s} CV 准确率 {mean_acc:.4f}  ({', '.join(f'{a:.3f}' for a in folds)})")
    best_kind = max(cv, key=cv.get)
    print(f"  -> 选中：{best_kind}（train CV {cv[best_kind]:.4f}）")

    std = Standardizer.fit(tr_feats)
    model = fit_candidate(best_kind, tr_feats, y, std, thr)

    print("=" * 68)
    print("第 6 步：用 train 上“预测正确”的样本定分类置信度阈值")
    pred, proba = model.predict(tr_feats)
    conf = proba.max(axis=1)
    correct = np.asarray([p == CLASSES[t] for p, t in zip(pred, y)])
    conf_min = float(np.percentile(conf[correct], 10)) if correct.any() else 0.6
    conf_min = float(min(max(conf_min, 0.30), 0.95))
    print(f"  正确样本置信度 p10 = {conf_min:.3f}")

    # ---------------- 写盘 ----------------
    baseline.save(os.path.join(model_dir, "baseline.json"))
    model.save(os.path.join(model_dir, "classifier.json"))
    thresholds = {
        "features": thr,
        "anomaly": {"z_cap": cfg["anomaly"]["z_cap"], **an_thr},
        "decision": {"low": an_thr["low"], "high": an_thr["high"], "conf_min": conf_min,
                     "pass_normal_prob": 0.9},
        "provenance": {"fit_split": "train", "n_train": len(tr_rows),
                       "why": why, "anomaly_info": an_info, "cv": cv,
                       "selected": best_kind,
                       "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    with open(os.path.join(model_dir, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    # 同步写回 config.json（去掉 _说明 字段的旧值，保持可读）
    cfg["thresholds"] = {k: round(v, 6) for k, v in thr.items()}
    cfg["decision"] = {"_说明": "串联决策：low/high 是异常分阈值，conf_min 是分类置信度阈值。",
                       "low": round(an_thr["low"], 4), "high": round(an_thr["high"], 4),
                       "conf_min": round(conf_min, 4), "pass_normal_prob": 0.9}
    cfg["model"] = {"selected_classifier": best_kind, "cv_accuracy": round(cv[best_kind], 4)}
    save_config(cfg)

    print("=" * 68)
    print(f"完成：{os.path.join(model_dir, 'thresholds.json')}")
    print(f"      {os.path.join(model_dir, 'classifier.json')}  ({best_kind})")
    print("下一步：python evaluate.py --split val")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="只用 train 定阈值并选分类器")
    ap.add_argument("--recalc", action="store_true", help="强制重算特征缓存")
    args = ap.parse_args()
    main(recalc=args.recalc)
