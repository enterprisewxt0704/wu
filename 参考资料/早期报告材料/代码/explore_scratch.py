# -*- coding: utf-8 -*-
"""Second exploration: Hough-style straight-line feature for scratches."""
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


def polar(a, cx, cy, R, rmin=0.10, rmax=0.86):
    th = (np.arange(NB_TH) + 0.5) / NB_TH * 2 * np.pi - np.pi
    rr = rmin + (rmax - rmin) * (np.arange(NB_R) + 0.5) / NB_R
    TH, RR = np.meshgrid(th, rr)
    px = cx + RR * R * np.cos(TH)
    py = cy + RR * R * np.sin(TH)
    x = np.clip(px.astype(np.int32), 0, a.shape[1] - 1)
    y = np.clip(py.astype(np.int32), 0, a.shape[0] - 1)
    lum = 0.299 * a[y, x, 0] + 0.587 * a[y, x, 1] + 0.114 * a[y, x, 2]
    return lum, x, y, RR


def hough_line_peak(xs, ys, rmin_pt, R, cx, cy, step=1.5, nang=180):
    """max number of points falling into one narrow straight band"""
    best = 0.0
    best_phi = -1
    for k in range(nang):
        phi = np.pi * k / nang
        s = xs * np.cos(phi) + ys * np.sin(phi)
        b = np.rint(s / step).astype(np.int32)
        b -= b.min()
        c = np.bincount(b)
        v = float(c.max())
        if v > best:
            best, best_phi = v, phi
    return best, best_phi


def feats(p):
    a = load(p)
    cx, cy, R, m = geom(a)
    lum, xg, yg, RR = polar(a, cx, cy, R)
    ln = lum / max(float(np.median(lum)), 1e-6)
    med_r = np.median(ln, axis=1)
    resid = ln - med_r[:, None]

    inner = RR < 0.72
    bright = (resid > 0.08) & inner
    dark = (resid < -0.10) & inner
    # a stain has a bright specular edge hugging its dark body: drop such pixels
    dd = dark.astype(np.uint8)
    for sh in ((0, 3), (3, 0), (0, -3), (-3, 0), (2, 2), (-2, 2), (2, -2), (-2, -2)):
        dd = dd | np.roll(np.roll(dd, sh[0], axis=0), sh[1], axis=1)
    bright = bright & (dd == 0)
    n = int(bright.sum())
    peak = 0.0
    purity = 0.0
    if n >= 15:
        xs = xg[bright].astype(np.float32)
        ys = yg[bright].astype(np.float32)
        peak, phi = hough_line_peak(xs, ys, 0.0, R, cx, cy)
        purity = peak / max(np.sqrt(n) * 1.6, 1.0)
    dark = (resid < -0.10) & inner
    return dict(
        n_bright=n,
        bright_area=float(bright.mean()),
        line_peak=peak,
        line_purity=purity,
        dark_area=float(dark.mean()),
        dark_mean=float(-resid[dark].mean()) if dark.any() else 0.0,
        R=R,
    )


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "labels_train_val.csv"), encoding="utf-8-sig")))
    out = []
    for r in rows:
        p = os.path.join(ROOT, r["relative_path"].replace("/", os.sep))
        f = feats(p)
        f.update(cls=r["class_name"], split=r["split"], sev=r["severity"], file=r["file_name"])
        out.append(f)
    keys = ["bright_area", "line_peak", "line_purity", "dark_area", "dark_mean"]
    print(f"{'class':8s}" + "".join(f"{k:>16s}" for k in keys))
    for c in ["normal", "notch", "scratch", "stain"]:
        sel = [r for r in out if r["cls"] == c]
        line = f"{c:8s}"
        for k in keys:
            v = np.array([r[k] for r in sel])
            line += f"{v.mean():7.3f}±{v.std():7.3f}"
        print(line)
    print()
    # per-severity for stain and scratch
    for c in ["scratch", "stain"]:
        print(c)
        for sev in ["mild", "medium", "severe"]:
            sel = [r for r in out if r["cls"] == c and r["sev"] == sev]
            if not sel:
                continue
            v = np.array([r["line_peak"] for r in sel])
            print(f"  {sev:7s} n={len(sel):3d} line_peak={v.mean():7.1f}  bright_area={np.mean([r['bright_area'] for r in sel]):.3f}  dark_area={np.mean([r['dark_area'] for r in sel]):.4f}")


if __name__ == "__main__":
    main()
