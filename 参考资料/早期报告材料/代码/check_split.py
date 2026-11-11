# -*- coding: utf-8 -*-
"""Honest split: pick thresholds from TRAIN only, then score VAL only."""
import csv
import json
import os

import numpy as np

ROOT = r"C:\Users\wuxuanting\Downloads\创新实践\创新实践\dataset"
d = json.load(open(r"D:\智造创新实践\build\feat_rules.json", encoding="utf-8"))
tr = [r for r in d if r["split"] == "train"]
va = [r for r in d if r["split"] == "val"]


def q(rows, key, p):
    return float(np.percentile([r[key] for r in rows], p))


# thresholds only from train
normal = [r for r in tr if r["cls"] == "normal"]
notch = [r for r in tr if r["cls"] == "notch"]
scratch = [r for r in tr if r["cls"] == "scratch"]
stain = [r for r in tr if r["cls"] == "stain"]

T1 = 0.5 * q(notch, "notch_depth", 5)                    # half of the weakest notch
T2 = 0.5 * q([r for r in stain if r["dark_area"] > 0.004], "dark_area", 5)
T4 = 0.5 * (q(stain, "line_purity", 95) + q(scratch, "line_purity", 5))
T3 = 0.5 * q(scratch, "line_peak", 5)
BIG = 0.5 * q(stain, "dark_area", 50)
print(f"train-derived thresholds: T_notch={T1:.4f}  T_stain={T2:.4f}  T_purity={T4:.2f}  T_peak={T3:.0f}  BIG={BIG:.4f}")


def rule(r):
    if r["dark_area"] > BIG:
        return "stain"
    if r["line_purity"] > T4 and r["line_peak"] > T3:
        return "scratch"
    if r["notch_depth"] > T1:
        return "notch"
    if r["dark_area"] > T2:
        return "stain"
    return "normal"


classes = ["normal", "notch", "scratch", "stain"]
for name, rows in (("TRAIN", tr), ("VAL", va)):
    cm = {a: {b: 0 for b in classes} for a in classes}
    for r in rows:
        cm[r["cls"]][rule(r)] += 1
    ok = sum(cm[c][c] for c in classes)
    print(f"\n{name} (n={len(rows)})  accuracy={ok}/{len(rows)}={ok/len(rows):.3f}")
    print(f"{'':9s}" + "".join(f"{c:>9s}" for c in classes))
    for a in classes:
        print(f"{a:9s}" + "".join(f"{cm[a][b]:9d}" for b in classes))
    if name == "VAL":
        for c in classes:
            tp = cm[c][c]
            fp = sum(cm[a][c] for a in classes if a != c)
            fn = sum(cm[c][b] for b in classes if b != c)
            prec = tp / (tp + fp) if tp + fp else 0
            rec = tp / (tp + fn) if tp + fn else 0
            print(f"  {c:8s} precision={prec:.3f} recall={rec:.3f}")
        # 3-way disposition mapping for a defect-vs-ok product view
        print("\n  as binary defect/ok on VAL:")
        tp = sum(cm[c][c] for c in ["notch", "scratch", "stain"])
        fn = sum(cm[c]["normal"] for c in ["notch", "scratch", "stain"])
        fp = sum(cm["normal"][c] for c in ["notch", "scratch", "stain"])
        tn = cm["normal"]["normal"]
        print(f"    TP={tp} FN={fn} FP={fp} TN={tn}")
        print(f"    recall={tp/(tp+fn):.3f} precision={tp/(tp+fp) if tp+fp else 0:.3f} accuracy={(tp+tn)/len(rows):.3f}")
