# -*- coding: utf-8 -*-
"""Extract text (titles, body, tables, notes, group shapes) from course PPTX files."""
import sys, os, glob
from pptx import Presentation
from pptx.util import Emu

SRC = r"C:\Users\wuxuanting\Downloads\创新实践\创新实践"
OUT = r"D:\智造创新实践\extracted"

os.makedirs(OUT, exist_ok=True)


def iter_shapes(shapes):
    for sh in shapes:
        if sh.shape_type == 6:  # group
            yield from iter_shapes(sh.shapes)
        else:
            yield sh


def shape_text(sh):
    parts = []
    if sh.has_text_frame:
        for p in sh.text_frame.paragraphs:
            t = "".join(r.text for r in p.runs).strip()
            if t:
                parts.append(("  " * (p.level or 0)) + t)
    if sh.has_table:
        tbl = sh.table
        for row in tbl.rows:
            cells = []
            for c in row.cells:
                cells.append(" ".join(c.text.split()))
            parts.append("| " + " | ".join(cells) + " |")
    return parts


for path in sorted(glob.glob(os.path.join(SRC, "*.pptx"))):
    name = os.path.splitext(os.path.basename(path))[0]
    prs = Presentation(path)
    lines = []
    lines.append("=" * 80)
    lines.append(f"FILE: {name}  slides={len(prs.slides)}  size={prs.slide_width}x{prs.slide_height}")
    lines.append("=" * 80)
    for idx, slide in enumerate(prs.slides, 1):
        lines.append(f"\n----- slide {idx} -----")
        for sh in iter_shapes(slide.shapes):
            if sh.has_text_frame or sh.has_table:
                txt = shape_text(sh)
                if txt:
                    lines.append("[" + (sh.name or "shape") + "]")
                    lines.extend(txt)
        if slide.has_notes_slide:
            nt = slide.notes_slide.notes_text_frame.text.strip()
            if nt:
                lines.append("[NOTES] " + " ".join(nt.split()))
    outpath = os.path.join(OUT, name + ".txt")
    with open(outpath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    total_chars = sum(len(l) for l in lines)
    print(f"{name}: {len(prs.slides)} slides, {total_chars} chars -> {outpath}")
