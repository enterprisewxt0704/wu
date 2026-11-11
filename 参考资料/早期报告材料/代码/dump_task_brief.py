# -*- coding: utf-8 -*-
"""Dump the DOCX assignment brief with paragraph styles and table contents."""
import io

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

PATH = r"C:\Users\wuxuanting\Downloads\创新实践\创新实践\智能制造系统设计与实战_课程考核任务书.docx"
OUT = r"D:\智造创新实践\build\task_brief.txt"


def iter_block_items(parent):
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)


doc = Document(PATH)
out = []
for item in iter_block_items(doc):
    if isinstance(item, Paragraph):
        t = item.text.strip()
        if t:
            style = item.style.name if item.style is not None else ""
            out.append(f"[{style}] {t}")
    else:
        out.append("<TABLE>")
        for row in item.rows:
            out.append("  | " + " | ".join(c.text.replace("\n", " / ").strip() for c in row.cells))
        out.append("</TABLE>")

io.open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("lines", len(out))
print("inline shapes", len(doc.inline_shapes))
