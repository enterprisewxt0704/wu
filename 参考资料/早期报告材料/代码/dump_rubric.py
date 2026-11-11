# -*- coding: utf-8 -*-
"""Dump the grading rubric workbook: sheets, merged ranges, values."""
import io

import openpyxl

PATH = (r"C:\Users\wuxuanting\Downloads\创新实践\创新实践"
        r"\智能制造系统设计与实战_课程考核评分表.xlsx")
OUT = r"D:\智造创新实践\build\rubric.txt"

wb = openpyxl.load_workbook(PATH, data_only=False)
out = []
for ws in wb.worksheets:
    out.append(f"=== SHEET {ws.title!r}  dims={ws.dimensions}  "
               f"max_row={ws.max_row} max_col={ws.max_column}")
    out.append("merged: " + ", ".join(str(r) for r in ws.merged_cells.ranges))
    for row in ws.iter_rows():
        cells = []
        for c in row:
            if c.value is None:
                continue
            cells.append(f"{c.coordinate}={c.value!r}")
        if cells:
            out.append("  " + " | ".join(cells))
    out.append("")

io.open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("sheets", [ws.title for ws in wb.worksheets])
print("chars", sum(len(x) for x in out))
