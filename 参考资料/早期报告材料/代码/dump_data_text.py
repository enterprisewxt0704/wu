# -*- coding: utf-8 -*-
"""Dump the data handbook in reading order for proofreading."""
import io

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

PATH = r"D:\智造创新实践\智能制造课件数据与指标定义手册.docx"
OUT = r"D:\智造创新实践\build\doc_data_text.txt"


def iter_block_items(parent):
    from docx.oxml.ns import qn
    body = parent.element.body
    for child in body.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)


doc = Document(PATH)
lines = []
for item in iter_block_items(doc):
    if isinstance(item, Paragraph):
        t = item.text.strip()
        if t:
            lines.append(t)
    else:
        for row in item.rows:
            cells = []
            for c in row.cells:
                cells.append(c.text.replace("\n", " ").strip())
            lines.append(" ｜ ".join(cells))
        lines.append("")

io.open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("lines", len(lines), "chars", sum(len(x) for x in lines))
