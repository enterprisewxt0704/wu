# -*- coding: utf-8 -*-
"""Dump the document in reading order (body blocks interleaved with tables)."""
import sys
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

sys.stdout.reconfigure(encoding="utf-8")
PATH = r"D:\智造创新实践\从一只瓶盖看懂智能制造_四讲知识点整理.docx"
doc = Document(PATH)
body = doc.element.body
out = []
for child in body.iterchildren():
    if child.tag == qn("w:p"):
        p = Paragraph(child, doc)
        t = p.text.strip()
        if p.style.name.startswith("Heading"):
            out.append("\n### " + p.text)
        elif t:
            out.append(t)
    elif child.tag == qn("w:tbl"):
        tbl = Table(child, doc)
        out.append("[TABLE %dx%d]" % (len(tbl.rows), len(tbl.columns)))
        for row in tbl.rows:
            cells = [c.text.replace("\n", " / ").strip() for c in row.cells]
            out.append(" | ".join(cells))
print("\n".join(out))
