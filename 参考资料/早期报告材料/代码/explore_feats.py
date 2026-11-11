# -*- coding: utf-8 -*-
"""Explore simple, explainable features on the cap dataset (numpy + PIL only).

Idea: work in polar coordinates of the cap disc.
  - rim radial profile r(theta)      -> notch
  - per-radius median subtraction    -> removes concentric rings
  - residual dark + desaturated blob -> stain
  - residual bright thin cluster     -> scratch
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = r"C:\Users\wuxuanting\Downloads\创新实践\创新实践\dataset"
NB_TH = 720          # angular bins
NB_R = 160           # radial bins
RMIN, RMAX = 0.10, 0.86   # normalized radius window for surface features


def load(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.float32)


def disc_geometry(a):
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    chroma = mx - mn
    mask = chroma > 45
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return None
    cx, cy = xs.mean(), ys.mean()
    d = np.hypot(xs - cx, ys - cy)
    R = float(np.percentile(d, 99.5))
    return cx, cy, R, mask


def radial_profile(mask, cx, cy, R):
    ys, xs = np.nonzero(mask)
    th = (np.arctan2(ys - cy, xs - cx) + np.pi) / (2 * np.pi)  # 0..1
    r = np.hypot(xs - cx, ys - cy) / R
    b = np.minimum((th * NB_TH).astype(np.int32), NB_TH - 1)
    prof = np.zeros(NB_TH, dtype=np.float32)
    np.maximum.at(prof, b, r.astype(np.float32))
    return prof


def polar_sample(a, cx, cy, R):
    th = (np.arange(NB_TH) + 0.5) / NB_TH * 2 * np.pi - np.pi
    rr = RMIN + (RMAX - RMIN) * (np.arange(NB_R) + 0.5) / NB_R
    TH, RR = np.meshgrid(th, rr)
    x = np.clip((cx + RR * R * np.cos(TH)).astype(np.int32), 0, a.shape[1] - 1)
    y = np.clip((cy + RR * R * np.sin(TH)).astype(np.int32), 0, a.shape[0] - 1)
    patch = a[y, x]                      # (NB_R, NB_TH, 3)
    lum = 0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2]
    chroma = patch.max(axis=2) - patch.min(axis=2)
    return lum, chroma, x, y


def features(path):
    a = load(path)
    g = disc_geometry(a)
    if g is None:
        return None
    cx, cy, R, mask = g
    prof = radial_profile(mask, cx, cy, R)

    # --- notch: dip of the rim radius profile ------------------------------
    Rmed = float(np.median(prof))
    dip = (Rmed - prof) / max(Rmed, 1e-6)
    notch_depth = float(np.clip(dip.max(), 0, 1))
    notch_frac = float((dip > 0.06).mean())      # angular width of the dip

    # --- surface features via polar unwrap --------------------------------
    lum, chroma, xg, yg = polar_sample(a, cx, cy, R)
    med_all = float(np.median(lum))              # per-image brightness normalisation
    ln = lum / max(med_all, 1e-6)
    med_r = np.median(ln, axis=1)                # clean ring profile per radius
    resid = ln - med_r[:, None]
    medC = float(np.median(chroma))

    dark = (resid < -0.10) & (chroma < 0.80 * medC)
    stain_area = float(dark.mean())
    stain_contrast = float(-resid[dark].mean()) if dark.any() else 0.0

    bright = resid > 0.09
    # ignore the outermost 12% (ticks) and drop blobs that are not elongated
    rr = (np.arange(NB_R) + 0.5) / NB_R * (RMAX - RMIN) + RMIN
    valid = rr < 0.74
    bright[~valid, :] = False
    scratch_area = float(bright.mean())
    scratch_elong = 0.0
    scratch_maxlen = 0.0
    if bright.sum() >= 20:
        ys_i, xs_i = np.nonzero(bright)
        px = xg[bright]
        py = yg[bright]
        cov = np.cov(np.vstack([px, py]))
        ev = np.linalg.eigvalsh(cov)
        ev = np.sort(np.maximum(ev, 1e-6))
        scratch_elong = float(np.sqrt(ev[1] / ev[0]))     # sqrt of eigenvalue ratio
        scratch_maxlen = float(np.sqrt(ev[1]) * 3.5)
    return dict(
        notch_depth=notch_depth,
        notch_frac=notch_frac,
        stain_area=stain_area,
        stain_contrast=stain_contrast,
        scratch_area=scratch_area,
        scratch_elong=scratch_elong,
        scratch_len=scratch_maxlen,
        R=float(R),
    )


def main():
    rows = []
    with open(os.path.join(ROOT, "labels_train_val.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    out = []
    for row in rows:
        p = os.path.join(ROOT, row["relative_path"].replace("/", os.sep))
        f = features(p)
        if f is None:
            continue
        f.update(cls=row["class_name"], split=row["split"], sev=row["severity"], file=row["file_name"])
        out.append(f)
    json.dump(out, open(r"D:\智造创新实践\build\feat_explore.json", "w", encoding="utf-8"), ensure_ascii=False)

    keys = ["notch_depth", "notch_frac", "stain_area", "stain_contrast", "scratch_area", "scratch_elong", "scratch_len"]
    print(f"{'class':8s}" + "".join(f"{k:>17s}" for k in keys))
    for c in ["normal", "notch", "scratch", "stain"]:
        sel = [r for r in out if r["cls"] == c and r["split"] == "train"]
        line = f"{c:8s}"
        for k in keys:
            vals = np.array([r[k] for r in sel])
            line += f"{vals.mean():8.3f}±{vals.std():6.3f}"
        print(line)
    print()
    # quick separability check on train+val with nearest centroid
    tr = [r for r in out if r["split"] in ("train", "val")]
    classes = ["normal", "notch", "scratch", "stain"]
    cent = {}
    for c in classes:
        sel = np.array([[r[k] for k in keys] for r in tr if r["cls"] == c])
        cent[c] = sel.mean(axis=0)
    sd = np.array([[r[k] for k in keys] for r in tr]).std(axis=0) + 1e-9
    correct = 0
    for r in tr:
        v = np.array([r[k] for k in keys]) / sd
        best = min(classes, key=lambda c: np.linalg.norm(v - cent[c] / sd))
        correct += best == r["cls"]
    print(f"nearest-centroid accuracy on train+val (n={len(tr)}): {correct}/{len(tr)} = {correct/len(tr):.3f}")


if __name__ == "__main__":
    main()
