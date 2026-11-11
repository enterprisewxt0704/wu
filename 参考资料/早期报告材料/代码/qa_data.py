# -*- coding: utf-8 -*-
"""Structural QA for the generated DOCX (Word COM is unavailable in this session)."""
import zipfile, re, sys, os
from lxml import etree
from docx import Document
from docx.shared import Cm
from docx.oxml.ns import qn

PATH = r"D:\智造创新实践\智能制造课件数据与指标定义手册.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

z = zipfile.ZipFile(PATH)
bad = []
for n in z.namelist():
    if n.endswith(".xml") or n.endswith(".rels"):
        try:
            etree.fromstring(z.read(n))
        except Exception as e:
            bad.append((n, str(e)))
print("zip entries:", len(z.namelist()))
print("xml parse errors:", bad or "none")

doc = Document(PATH)
print("paragraphs:", len(doc.paragraphs))
print("tables:", len(doc.tables))
print("sections:", len(doc.sections))
print("inline shapes:", len(doc.inline_shapes))
print("page size cm:", round(doc.sections[0].page_width.cm, 2), "x", round(doc.sections[0].page_height.cm, 2))
sec = doc.sections[0]
text_w = sec.page_width.cm - sec.left_margin.cm - sec.right_margin.cm
text_h = sec.page_height.cm - sec.top_margin.cm - sec.bottom_margin.cm
print("text area cm:", round(text_w, 2), "x", round(text_h, 2))

# headers / footers
print("header text:", sec.header.paragraphs[0].text)
print("footer text:", sec.footer.paragraphs[0].text)
print("diff first page:", sec.different_first_page_header_footer)

# duplicate child check
dups = []
for tag in ("pPr", "tcPr", "tblPr", "rPr"):
    for el in doc.element.body.iter(W + tag):
        tags = [c.tag for c in el]
        if len(tags) != len(set(tags)):
            dups.append((tag, tags))
print("duplicate children:", len(dups), dups[:3])

# table width sanity
over = []
for i, t in enumerate(doc.tables):
    widths = []
    for row in t.rows[:1]:
        widths = [c.width.cm if c.width else 0 for c in row.cells]
    if abs(sum(widths) - text_w) > 0.6:
        over.append((i, [round(w, 2) for w in widths], round(sum(widths), 2)))
print("tables with width mismatch:", over or "none")

# empty paragraphs count & page breaks
pb = 0
for p in doc.paragraphs:
    br = p._p.findall(".//" + W + "br")
    for b in br:
        if b.get(W + "type") == "page":
            pb += 1
print("explicit page breaks:", pb)

# styles used
from collections import Counter
c = Counter(p.style.name for p in doc.paragraphs)
print("styles:", dict(c))

# word count-ish
txt = "\n".join(p.text for p in doc.paragraphs)
print("chars:", len(txt))
print("headings:")
for p in doc.paragraphs:
    if p.style.name.startswith("Heading"):
        print("  ", p.style.name, "|", p.text)

# check every table cell has content / header colors etc.
empty_cells = 0
for t in doc.tables:
    for row in t.rows:
        for cell in row.cells:
            if not cell.text.strip() and len(cell.paragraphs) == 1 and not cell.paragraphs[0].runs:
                empty_cells += 1
print("empty cells:", empty_cells)

# images exist?
print("media:", [n for n in z.namelist() if n.startswith("word/media/")])
