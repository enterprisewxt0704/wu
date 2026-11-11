# -*- coding: utf-8 -*-
"""用自己的 16 张测试图批量跑特征识别流水线，输出逐张结果。"""
import sys, glob, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pipeline import Inspector

files = sorted(glob.glob(r"C:\Users\wuxuanting\WorkBuddy\2026-11-07-21-15-45\cap_test\imgs\*.png"))
insp = Inspector.from_config(recorder=None)
print(f"{'file':28s} {'pred':8s} {'conf':6s} {'score':6s} {'worst':8s} {'action':8s} {'unknown':7s} {'ms':6s}")
for p in files:
    r = insp.inspect(p)
    if r.error:
        print(f"{r.frame_id:28s} ERROR: {r.error}")
        continue
    proba = " ".join(f"{c}={v:.3f}" for c, v in sorted(r.proba.items(), key=lambda kv: -kv[1]))
    print(f"{r.frame_id:28s} {r.pred:8s} {r.conf:.3f}  {r.score:.4f} {r.worst:8s} {r.action:8s} "
          f"{r.decision.is_unknown!s:7s} {r.elapsed_ms:6.0f}  [{proba}]  {r.decision.reason}")
