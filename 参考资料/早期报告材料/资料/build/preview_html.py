# -*- coding: utf-8 -*-
"""Render an approximate HTML preview of the DOCX for visual QA (no Word available)."""
import sys, os, html
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

sys.stdout.reconfigure(encoding="utf-8")
PATH = r"D:\智造创新实践\从一只瓶盖看懂智能制造_四讲知识点整理.docx"
OUT = r"D:\智造创新实践\build\preview.html"
MEDIA = r"D:\智造创新实践"

doc = Document(PATH)


def px(v):  # twips -> px-ish (cm) ; use pt
    return v


def run_html(r):
    t = html.escape(r.text).replace("\n", "<br>")
    style = []
    if r.font.size:
        style.append("font-size:%.2fpt" % (r.font.size.pt))
    if r.font.bold:
        style.append("font-weight:700")
    if r.font.italic:
        style.append("font-style:italic")
    if r.font.color and r.font.color.rgb is not None:
        style.append("color:#%s" % str(r.font.color.rgb))
    name = r.font.name or "宋体"
    style.append("font-family:%s,serif" % name)
    if not t:
        return ""
    return "<span style='%s'>%s</span>" % (";".join(style), t)


def para_html(p):
    pPr = p._p.pPr
    css = ["margin:0"]
    if p.style.name.startswith("Heading"):
        lvl = p.style.name[-1]
        sizes = {"1": 18, "2": 13.5, "3": 11.5}
        css.append("font-family:微软雅黑,sans-serif")
        css.append("font-size:%.2fpt" % sizes.get(lvl, 11))
        css.append("font-weight:700")
        css.append("color:#%s" % ({"1": "1F3B63", "2": "2E5C8A", "3": "1F3B63"}[lvl]))
        css.append("margin:%dpt 0 %dpt 0" % ({"1": 18, "2": 14, "3": 10}[lvl],
                                            {"1": 10, "2": 5, "3": 4}[lvl]))
        if lvl == "1":
            css.append("page-break-before:always")
    else:
        css.append("font-size:10.5pt")
        css.append("color:#000")
        css.append("font-family:宋体,serif")
        css.append("text-align:justify")
        css.append("line-height:1.4")
        css.append("margin:0 0 5pt 0")
        if pPr is not None:
            ind = pPr.find(qn("w:ind"))
            if ind is not None and ind.get(qn("w:firstLine")):
                tw = int(ind.get(qn("w:firstLine")))
                css.append("text-indent:%.2fpt" % (tw / 20))
            if ind is not None and ind.get(qn("w:left")):
                tw = int(ind.get(qn("w:left")))
                css.append("margin-left:%.2fpt" % (tw / 20))
    style = ";".join(css)
    inner = "".join(run_html(r) for r in p.runs)
    if not inner:
        return "<p style='%s'>&nbsp;</p>" % style
    return "<p style='%s'>%s</p>" % (style, inner)


def cell_css(cell):
    css = ["padding:1.5pt 4pt", "vertical-align:middle", "font-size:9.2pt",
           "font-family:宋体,serif", "border:0.5pt solid #C9D3DF"]
    tcPr = cell._tc.tcPr
    if tcPr is not None:
        shd = tcPr.find(qn("w:shd"))
        if shd is not None:
            css.append("background:#%s" % (shd.get(qn("w:fill")) or "FFFFFF"))
        b = tcPr.find(qn("w:tcBorders"))
        if b is not None:
            left = b.find(qn("w:left"))
            if left is not None and left.get(qn("w:sz")):
                sz = int(left.get(qn("w:sz")))
                if sz >= 12:
                    css.append("border-left:%dpt solid #%s" % (sz / 8.0, left.get(qn("w:color"))))
    return css


def table_html(tbl):
    rows = []
    widths = []
    if tbl.rows:
        widths = [c.width.cm if c.width else 3 for c in tbl.rows[0].cells]
    total = sum(widths) or 1
    cols = ["<col style='width:%.2f%%'>" % (w * 100.0 / total) for w in widths]
    rows.append("<table style='width:100%%;border-collapse:collapse;table-layout:fixed;margin:2pt 0'><colgroup>%s</colgroup>"
                % "".join(cols))
    for row in tbl.rows:
        cells = []
        for cell in row.cells:
            inner = "".join(para_html(p) for p in cell.paragraphs)
            cells.append("<td style='%s'>%s</td>" % (";".join(cell_css(cell)), inner))
        rows.append("<tr>%s</tr>" % "".join(cells))
    rows.append("</table>")
    return "".join(rows)


def img_html(p):
    # find drawing/ rel id
    blips = p._p.findall(".//" + qn("a:blip"))
    if not blips:
        return None
    rid = blips[0].get(qn("r:embed"))
    part = doc.part.related_parts[rid]
    data = part.blob
    name = os.path.basename(part.partname)
    path = os.path.join(MEDIA, "build", name)
    with open(path, "wb") as f:
        f.write(data)
    return "<div style='text-align:center;margin:8pt 0 2pt 0'><img src='%s' style='width:16cm'></div>" % html.escape(path)


parts = []
body = doc.element.body
for child in body.iterchildren():
    if child.tag == qn("w:p"):
        p = Paragraph(child, doc)
        img = img_html(p)
        if img:
            parts.append(img)
        else:
            parts.append(para_html(p))
    elif child.tag == qn("w:tbl"):
        parts.append(table_html(Table(child, doc)))

CSS = """
@page { size: 21cm 29.7cm; margin: 2.2cm 2.3cm 2.0cm 2.3cm; }
body { width: 16.4cm; margin: 2.2cm 2.3cm; font-family:宋体,serif; font-size:10.5pt; }
"""
html_doc = "<html><head><meta charset='utf-8'><style>%s</style></head><body>%s</body></html>" % (CSS, "".join(parts))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html_doc)
print("wrote", OUT, len(html_doc))
