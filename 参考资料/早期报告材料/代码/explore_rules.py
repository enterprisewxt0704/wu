# -*- coding: utf-8 -*-
"""Final check: hierarchical, human-explainable rules on the cap dataset.

Features (all on the polar unwrap of the cap disc):
  notch_depth   - relative dip of the rim radius profile
  dark_area     - fraction of pixels clearly darker AND less saturated than the cap body
  line_peak     - longest straight bright band (Hough-style) inside the disc
Rules:
  1) notch_depth  > T1   -> notch
  2) dark_area    > T2   -> stain
  3) line_peak    > T3   -> scratch
  4) otherwise           -> normal
"""
import csv
import os

import numpy as np
from PIL import Image

ROOT = r"C:\Users\wuxuanting\Downloads\创新实践\创新实践\dataset"
NB_TH, NB_R = 720, 160


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def geom(a):
    chroma = a.max(axis=2) - a.min(axis=2)
    m = chroma > 45
    ys, xs = np.nonzero(m)
    cx, cy = xs.mean(), ys.mean()
    R = float(np.percentile(np.hypot(xs - cx, ys - cy), 99.5))
    return cx, cy, R, m


def rim_profile(m, cx, cy, R):
    ys, xs = np.nonzero(m)
    th = (np.arctan2(ys - cy, xs - cx) + np.pi) / (2 * np.pi)
    r = np.hypot(xs - cx, ys - cy) / R
    b = np.minimum((th * NB_TH).astype(np.int32), NB_TH - 1)
    prof = np.zeros(NB_TH, dtype=np.float32)
    np.maximum.at(prof, b, r.astype(np.float32))
    return prof


def polar(a, cx, cy, R):
    th = (np.arange(NB_TH) + 0.5) / NB_TH * 2 * np.pi - np.pi
    rr = 0.10 + 0.76 * (np.arange(NB_R) + 0.5) / NB_R
    TH, RR = np.meshgrid(th, rr)
    x = np.clip((cx + RR * R * np.cos(TH)).astype(np.int32), 0, a.shape[1] - 1)
    y = np.clip((cy + RR * R * np.sin(TH)).astype(np.int32), 0, a.shape[0] - 1)
    p = a[y, x]
    lum = 0.299 * p[..., 0] + 0.587 * p[..., 1] + 0.114 * p[..., 2]
    chroma = p.max(axis=2) - p.min(axis=2)
    return lum, chroma, x, y, RR


def line_peak(xs, ys, step=1.5, nang=90):
    best = 0.0
    for k in range(nang):
        phi = np.pi * k / nang
        s = xs * np.cos(phi) + ys * np.sin(phi)
        b = np.rint(s / step).astype(np.int32)
        b -= b.min()
        best = max(best, float(np.bincount(b).max()))
    return best


def feats(p):
    a = load(p)
    cx, cy, R, m = geom(a)
    prof = rim_profile(m, cx, cy, R)
    Rmed = float(np.median(prof))
    notch_depth = float(np.clip(((Rmed - prof) / max(Rmed, 1e-6)).max(), 0, 1))

    lum, chroma, xg, yg, RR = polar(a, cx, cy, R)
    ln = lum / max(float(np.median(lum)), 1e-6)
    resid = ln - np.median(ln, axis=1)[:, None]
    inner = RR < 0.72
    dark = (resid < -0.10) & (chroma < 0.80 * float(np.median(chroma))) & inner
    bright = (resid > 0.08) & inner
    peak = 0.0
    purity = 0.0
    nb = int(bright.sum())
    if nb >= 15:
        peak = line_peak(xg[bright].astype(np.float32), yg[bright].astype(np.float32))
        purity = peak / max(np.sqrt(nb) * 1.6, 1.0)
    return dict(notch_depth=notch_depth, dark_area=float(dark.mean()), line_peak=peak,
                line_purity=purity, n_bright=nb)


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "labels_train_val.csv"), encoding="utf-8-sig")))
    data = []
    for r in rows:
        p = os.path.join(ROOT, r["relative_path"].replace("/", os.sep))
        f = feats(p)
        f.update(cls=r["class_name"], split=r["split"], sev=r["severity"])
        data.append(f)

    keys = ["notch_depth", "dark_area", "line_peak", "line_purity"]
    print(f"{'class':8s}" + "".join(f"{k:>22s}" for k in keys))
    for c in ["normal", "notch", "scratch", "stain"]:
        sel = [r for r in data if r["cls"] == c]
        line = f"{c:8s}"
        for k in keys:
            v = np.array([r[k] for r in sel])
            line += f"{v.mean():9.3f}±{v.std():8.3f}"
        print(line)

    T1, T2, T3, T4 = 0.05, 0.005, 60.0, 4.0

    def rule(r):
        if r["dark_area"] > 0.05:
            return "stain"
        if r["line_purity"] > T4 and r["line_peak"] > T3:
            return "scratch"
        if r["notch_depth"] > T1:
            return "notch"
        if r["dark_area"] > T2:
            return "stain"
        return "normal"

    classes = ["normal", "notch", "scratch", "stain"]
    cm = {a: {b: 0 for b in classes} for a in classes}
    for r in data:
        cm[r["cls"]][rule(r)] += 1
    print("\nconfusion (rows=true, cols=pred) thresholds", (T1, T2, T3))
    print(f"{'':9s}" + "".join(f"{c:>10s}" for c in classes))
    for a in classes:
        print(f"{a:9s}" + "".join(f"{cm[a][b]:10d}" for b in classes))
    ok = sum(cm[c][c] for c in classes)
    print(f"\naccuracy on train+val (n={len(data)}): {ok}/{len(data)} = {ok/len(data):.3f}")

    print("\nper-severity recall")
    for c in ["notch", "scratch", "stain"]:
        for sev in ["mild", "medium", "severe"]:
            sel = [r for r in data if r["cls"] == c and r["sev"] == sev]
            if not sel:
                continue
            hit = sum(1 for r in sel if rule(r) == c)
            print(f"  {c:8s}{sev:8s} n={len(sel):3d} recall={hit}/{len(sel)}")

    # save features for later plotting
    out = os.path.join(r"D:\智造创新实践\build", "feat_rules.json")
    import json
    json.dump(data, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
