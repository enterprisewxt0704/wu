# -*- coding: utf-8 -*-
"""Build the beginner-friendly DOCX study document from content_blocks.BLOCKS."""
import os
import sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from content_blocks import BLOCKS

OUT = r"D:\智造创新实践\从一只瓶盖看懂智能制造_四讲知识点整理.docx"

NAVY = RGBColor(0x1F, 0x3B, 0x63)
BLUE = RGBColor(0x2E, 0x5C, 0x8A)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHTGRAY = RGBColor(0x80, 0x80, 0x80)
ORANGE = RGBColor(0xC5, 0x5A, 0x11)
GREEN = RGBColor(0x37, 0x6E, 0x2E)
RED = RGBColor(0xA6, 0x1C, 0x1C)

BODY_FONT = "宋体"
HEAD_FONT = "微软雅黑"
TEXT_WIDTH_CM = 16.4

TABLE_NO = 0
FIG_NO = 0


# --------------------------------------------------------------------------- xml helpers
def _el(tag, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn("w:" + k), str(v))
    return e


def set_run(run, size=10.5, bold=False, color=None, font=BODY_FONT, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = _el("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), font)
    rfonts.set(qn("w:hAnsi"), font)
    rfonts.set(qn("w:eastAsia"), font)
    if color is not None:
        run.font.color.rgb = color


def add_rich(par, text, size=10.5, color=None, font=BODY_FONT, bold=False):
    """Add text supporting **bold** markers."""
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = par.add_run(part)
        set_run(r, size=size, bold=(bold or i % 2 == 1), color=color, font=font)
    return par


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = _el("w:shd", val="clear", color="auto", fill=hex_color)
    tcPr.append(shd)


def cell_border(cell, edge, color="FFFFFF", sz=4, val="single"):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = _el("w:tcBorders")
        tcPr.append(borders)
    old = borders.find(qn("w:" + edge))
    if old is not None:
        borders.remove(old)
    borders.append(_el("w:" + edge, val=val, sz=sz, space=0, color=color))


def table_borders(table, color="C9D3DF", sz=4, inside=True):
    tblPr = table._tbl.tblPr
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    borders = _el("w:tblBorders")
    edges = ["top", "left", "bottom", "right"]
    if inside:
        edges += ["insideH", "insideV"]
    for edge in edges:
        borders.append(_el("w:" + edge, val="single", sz=sz, space=0, color=color))
    tblPr.append(borders)


def table_cell_margins(table, top=40, bottom=40, left=90, right=90):
    tblPr = table._tbl.tblPr
    mar = _el("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        mar.append(_el("w:" + k, w=v, type="dxa"))
    tblPr.append(mar)


def row_cant_split(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(_el("w:cantSplit"))


def header_repeat(row):
    trPr = row._tr.get_or_add_trPr()
    trPr.append(_el("w:tblHeader"))


def add_page_number(par):
    run = par.add_run()
    fld_begin = _el("w:fldChar", fldCharType="begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = _el("w:fldChar", fldCharType="end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run(run, size=9, color=LIGHTGRAY)


def para_bottom_border(par, color="1F3B63", sz=12, space=4):
    pPr = par._p.get_or_add_pPr()
    pbdr = _el("w:pBdr")
    pbdr.append(_el("w:bottom", val="single", sz=sz, space=space, color=color))
    pPr.append(pbdr)


# --------------------------------------------------------------------------- schema order fix
_ORDERS = {
    "w:pPr": ["w:pStyle", "w:keepNext", "w:keepLines", "w:pageBreakBefore", "w:framePr",
              "w:widowControl", "w:numPr", "w:suppressLineNumbers", "w:pBdr", "w:shd",
              "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap", "w:overflowPunct",
              "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi", "w:adjustRightInd",
              "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents",
              "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
              "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr",
              "w:sectPr", "w:pPrChange"],
    "w:tcPr": ["w:cnfStyle", "w:tcW", "w:gridSpan", "w:hMerge", "w:vMerge", "w:tcBorders",
               "w:shd", "w:noWrap", "w:tcMar", "w:textDirection", "w:tcFitText", "w:vAlign",
               "w:hideMark", "w:headers", "w:cellIns", "w:cellDel", "w:cellMerge", "w:tcPrChange"],
    "w:tblPr": ["w:tblStyle", "w:tblpPr", "w:tblOverlap", "w:bidiVisual",
                "w:tblStyleRowBandSize", "w:tblStyleColBandSize", "w:tblW", "w:jc",
                "w:tblCellSpacing", "w:tblInd", "w:tblBorders", "w:shd", "w:tblLayout",
                "w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription", "w:tblPrChange"],
    "w:rPr": ["w:rStyle", "w:rFonts", "w:b", "w:bCs", "w:i", "w:iCs", "w:caps", "w:smallCaps",
              "w:strike", "w:dstrike", "w:outline", "w:shadow", "w:emboss", "w:imprint",
              "w:noProof", "w:snapToGrid", "w:vanish", "w:webHidden", "w:color", "w:spacing",
              "w:w", "w:kern", "w:position", "w:sz", "w:szCs", "w:highlight", "w:u",
              "w:effect", "w:bdr", "w:shd", "w:fitText", "w:vertAlign", "w:rtl", "w:cs",
              "w:em", "w:lang", "w:eastAsianLayout", "w:specVanish", "w:oMath"],
}


def normalize_schema_order(doc):
    """Reorder children of pPr/tcPr/tblPr/rPr to satisfy the OOXML schema order Word expects."""
    roots = [doc.element.body]
    for sec in doc.sections:
        for part in (sec.header, sec.footer, sec.first_page_header, sec.first_page_footer):
            try:
                roots.append(part._element)
            except Exception:
                pass
    for root in roots:
        for tag, order in _ORDERS.items():
            for el in root.iter(qn(tag)):
                children = list(el)
                # drop duplicate sibling elements, keeping the last one written
                seen = {}
                for c in children:
                    if c.tag in seen:
                        el.remove(seen[c.tag])
                    seen[c.tag] = c
                children = list(el)
                idx = {name: i for i, name in enumerate(order)}
                children.sort(key=lambda c: idx.get(c.tag.replace(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}", "w:"), 999))
                for c in children:
                    el.remove(c)
                for c in children:
                    el.append(c)


# --------------------------------------------------------------------------- document setup
def setup_styles(doc):
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(10.5)
    rpr = st.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = _el("w:rFonts")
        rpr.insert(0, rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), BODY_FONT)
    pf = st.paragraph_format
    pf.line_spacing = 1.4
    pf.space_before = Pt(0)
    pf.space_after = Pt(5)
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    specs = {
        "Heading 1": (18, NAVY, 18, 10, True),
        "Heading 2": (13.5, BLUE, 14, 5, False),
        "Heading 3": (11.5, NAVY, 10, 4, False),
    }
    for name, (size, color, before, after, pagebreak) in specs.items():
        s = doc.styles[name]
        s.font.name = HEAD_FONT
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = color
        rpr = s.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = _el("w:rFonts")
            rpr.insert(0, rfonts)
        for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rfonts.set(qn(attr), HEAD_FONT)
        pf = s.paragraph_format
        pf.space_before = Pt(before)
        pf.space_after = Pt(after)
        pf.keep_with_next = True
        pf.line_spacing = 1.25
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if pagebreak:
            pf.page_break_before = True
            pf.space_before = Pt(0)


def setup_page(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.3)
    sec.right_margin = Cm(2.3)
    sec.header_distance = Cm(1.1)
    sec.footer_distance = Cm(1.0)
    sec.different_first_page_header_footer = True

    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = hp.add_run("从一只瓶盖看懂智能制造  ·  《智能制造系统设计与实战》1–4讲学习笔记")
    set_run(r, size=8, color=LIGHTGRAY, font=HEAD_FONT)
    para_bottom_border(hp, color="D6DCE4", sz=4, space=2)

    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run("— ")
    set_run(r, size=9, color=LIGHTGRAY)
    add_page_number(fp)
    r = fp.add_run(" —")
    set_run(r, size=9, color=LIGHTGRAY)


# --------------------------------------------------------------------------- building blocks
def body_par(doc, text, indent=True, size=10.5, color=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             before=0, after=5, spacing=1.4):
    p = doc.add_paragraph()
    p.paragraph_format.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = spacing
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    add_rich(p, text, size=size, color=color)
    return p


def add_list(doc, items, numbered=False):
    for i, item in enumerate(items, 1):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Cm(0.75)
        pf.first_line_indent = Cm(-0.75)
        pf.space_before = Pt(0)
        pf.space_after = Pt(3)
        pf.line_spacing = 1.35
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        marker = f"{i}. " if numbered else "• "
        r = p.add_run(marker)
        set_run(r, size=10.5, bold=numbered, color=BLUE if numbered else GRAY)
        add_rich(p, item, size=10.5)
    # small gap after list
    doc.paragraphs[-1].paragraph_format.space_after = Pt(7)


def add_table(doc, spec):
    global TABLE_NO
    TABLE_NO += 1
    headers = spec["headers"]
    rows = spec["rows"]
    widths = spec.get("widths", [TEXT_WIDTH_CM / len(headers)] * len(headers))
    total = sum(widths)
    if total > TEXT_WIDTH_CM - 0.05:
        widths = [w * (TEXT_WIDTH_CM - 0.05) / total for w in widths]
    small = spec.get("small", False)
    fs = 8.5 if small else 9.2

    if spec.get("caption"):
        cp = doc.add_paragraph()
        cp.paragraph_format.space_before = Pt(8)
        cp.paragraph_format.space_after = Pt(3)
        cp.paragraph_format.keep_with_next = True
        cp.paragraph_format.line_spacing = 1.15
        r = cp.add_run(f"表 {TABLE_NO}　")
        set_run(r, size=9, bold=True, color=BLUE, font=HEAD_FONT)
        r = cp.add_run(spec["caption"])
        set_run(r, size=9, bold=True, color=GRAY, font=HEAD_FONT)

    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table_borders(table)
    table_cell_margins(table)
    # fixed layout
    tblPr = table._tbl.tblPr
    tblW = tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = _el("w:tblW", w=0, type="auto")
        tblPr.append(tblW)
    tblW.set(qn("w:type"), "dxa")
    tblW.set(qn("w:w"), str(int(sum(widths) * 567)))
    # 同步 tblGrid 列宽，避免 Word/WPS 用等分网格重算列宽
    for g, w in zip(table._tbl.tblGrid.gridCol_lst, widths):
        g.set(qn("w:w"), str(int(w * 567)))

    hdr = table.rows[0]
    header_repeat(hdr)
    row_cant_split(hdr)
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.width = Cm(widths[i])
        shade(cell, "1F3B63")
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.line_spacing = 1.15
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if h:
            add_rich(p, h, size=fs, color=RGBColor(0xFF, 0xFF, 0xFF), font=HEAD_FONT, bold=True)

    for ridx, rowdata in enumerate(rows):
        row = table.add_row()
        row_cant_split(row)
        for i, val in enumerate(rowdata):
            cell = row.cells[i]
            cell.width = Cm(widths[i])
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if ridx % 2 == 1:
                shade(cell, "F2F6FA")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.2
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            add_rich(p, str(val), size=fs)

    gap = doc.add_paragraph()
    gap.paragraph_format.space_before = Pt(0)
    gap.paragraph_format.space_after = Pt(2)
    gap.paragraph_format.line_spacing = 1.0
    r = gap.add_run("")
    set_run(r, size=4)


def add_callout(doc, spec, number=None):
    kind = spec.get("kind", "note")
    palette = {
        "tip": ("EAF3E7", "4F7A3A", GREEN),
        "warn": ("FBEDEA", "B03A2E", RED),
        "note": ("EAF1F8", "2E5C8A", BLUE),
        "key": ("E8EEF6", "1F3B63", NAVY),
    }
    fill, border, title_color = palette.get(kind, palette["note"])
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.rows[0].cells[0]
    cell.width = Cm(TEXT_WIDTH_CM)
    shade(cell, fill)
    # borders: colored left accent, invisible others
    for edge in ("top", "bottom", "right"):
        cell_border(cell, edge, color=fill, sz=4)
    cell_border(cell, "left", color=border, sz=18)
    table_cell_margins(table, top=90, bottom=90, left=150, right=140)

    first = True
    if spec.get("title"):
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.2
        add_rich(p, spec["title"], size=9.8, color=title_color, font=HEAD_FONT, bold=True)
        first = False
    lines = list(spec.get("lines", []))
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    for line in lines:
        if first:
            p = cell.paragraphs[0]
            first = False
        else:
            p = cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.3
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if line:
            add_rich(p, line, size=9.5)
        else:
            r = p.add_run("")
            set_run(r, size=4)

    gap = doc.add_paragraph()
    gap.paragraph_format.space_before = Pt(0)
    gap.paragraph_format.space_after = Pt(4)
    gap.paragraph_format.line_spacing = 1.0
    r = gap.add_run("")
    set_run(r, size=4)


def add_key(doc, text):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.rows[0].cells[0]
    cell.width = Cm(TEXT_WIDTH_CM)
    shade(cell, "F2F5F9")
    for edge in ("top", "bottom", "left", "right"):
        cell_border(cell, edge, color="D6DEE8", sz=4)
    table_cell_margins(table, top=80, bottom=80, left=100, right=100)
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.3
    add_rich(p, text, size=11.5, color=NAVY, font=HEAD_FONT, bold=True)
    gap = doc.add_paragraph()
    gap.paragraph_format.space_before = Pt(0)
    gap.paragraph_format.space_after = Pt(4)
    gap.paragraph_format.line_spacing = 1.0
    r = gap.add_run("")
    set_run(r, size=4)


def add_figure(doc, spec):
    global FIG_NO
    FIG_NO += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(spec["path"], width=Cm(spec.get("width", 16.0)))
    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_before = Pt(0)
    cp.paragraph_format.space_after = Pt(8)
    cp.paragraph_format.line_spacing = 1.1
    r = cp.add_run(f"图 {FIG_NO}　{spec.get('caption','')}")
    set_run(r, size=9, bold=True, color=GRAY, font=HEAD_FONT)


def build_cover(doc):
    # top spacer
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(60)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(30)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("智能制造系统设计与实战　·　四讲知识点整理")
    set_run(r, size=11, color=BLUE, font=HEAD_FONT, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(26)
    para_bottom_border(p, color="1F3B63", sz=14, space=2)
    r = p.add_run("")
    set_run(r, size=2)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.25
    add_rich(p, "从一只瓶盖\n看懂智能制造", size=34, color=NAVY, font=HEAD_FONT, bold=True)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(24)
    r = p.add_run("《智能制造系统设计与实战》第 1–4 讲　初学者精讲笔记")
    set_run(r, size=15, color=BLUE, font=HEAD_FONT)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(30)
    add_rich(p, "把系统、视觉、AI 与落地串成一条主线：\n"
                "感知 → 分析 → 决策 → 执行 → 反馈。", size=12, color=GRAY, font=HEAD_FONT)

    # stat band
    stats = [("4", "讲课件提炼"), ("3", "张系统示意图"), ("15", "个入门易错点"), ("15", "道自测题")]
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, (num, label) in enumerate(stats):
        cell = table.rows[0].cells[i]
        cell.width = Cm(TEXT_WIDTH_CM / 4)
        shade(cell, "F2F6FA")
        cell_border(cell, "left", color="FFFFFF", sz=12)
        cell_border(cell, "right", color="FFFFFF", sz=12)
        cell_border(cell, "top", color="FFFFFF", sz=12)
        cell_border(cell, "bottom", color="FFFFFF", sz=12)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(num)
        set_run(r, size=24, bold=True, color=NAVY, font=HEAD_FONT)
        p2 = cell.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2.paragraph_format.space_before = Pt(2)
        p2.paragraph_format.space_after = Pt(10)
        p2.paragraph_format.line_spacing = 1.0
        r = p2.add_run(label)
        set_run(r, size=9, color=GRAY, font=HEAD_FONT)

    for _ in range(7):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("适用对象：零基础学习者、课程复习者、项目设计者")
    set_run(r, size=9.5, color=GRAY, font=HEAD_FONT)
    p = doc.add_paragraph()
    r = p.add_run("整理日期：2026 年 11 月")
    set_run(r, size=9.5, color=GRAY, font=HEAD_FONT)


def build_front_matter(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("阅读说明")
    set_run(r, size=18, bold=True, color=NAVY, font=HEAD_FONT)
    para_bottom_border(p, color="D6DCE4", sz=6, space=3)

    body_par(doc, "这是一份为初学者准备的课程笔记。它不复制幻灯片，而是把四讲内容重新组织成"
                  "**一条可以讲通的主线**：一只瓶盖怎样被看见、被理解、被决策、被执行，"
                  "最后结果又怎样回到系统。", indent=True)
    body_par(doc, "阅读时不必一次记住所有术语。建议先读第 0 章建立全局印象，再按章节顺序推进；"
                  "每章末尾有“一分钟自测”，第 5 章有完整的速查表、易错点和 15 道自测题。", indent=True)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("目录")
    set_run(r, size=16, bold=True, color=NAVY, font=HEAD_FONT)
    para_bottom_border(p, color="D6DCE4", sz=6, space=3)

    toc = [
        ("第0章", "先看全局：一只瓶盖的智能制造之旅", "四讲知识地图与五句话总结"),
        ("第1章", "认识智能制造：不是聪明的设备，而是会闭环的系统", "六要素、三种流、五模块、规则与 AI"),
        ("第2章", "感知：机器怎样“看见”瓶盖", "成像、像素、传统视觉七步链、人工特征"),
        ("第3章", "分析：机器怎样“看懂”瓶盖", "四类 AI 任务、混淆矩阵、精确率与召回率"),
        ("第4章", "决策、执行与反馈：让模型真正改变生产", "阈值、位置跟踪、节拍、异常降级、数据记录"),
        ("第5章", "一页纸复习、易错点与自测", "速查表、15 个易错点、15 道题、4 个练习"),
        ("附　录", "术语中英对照表", "按出现顺序整理，便于查证"),
    ]
    table = doc.add_table(rows=0, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for i, (no, title, desc) in enumerate(toc):
        row = table.add_row()
        cells = row.cells
        for c, w in zip(cells, (2.2, 7.4, 6.8)):
            c.width = Cm(w)
            c_borders = [("top", "FFFFFF"), ("left", "FFFFFF"), ("bottom", "FFFFFF"), ("right", "FFFFFF")]
            for edge, color in c_borders:
                cell_border(c, edge, color=color, sz=8)
        p0 = cells[0].paragraphs[0]
        p0.paragraph_format.space_before = Pt(6)
        p0.paragraph_format.space_after = Pt(6)
        r = p0.add_run(no)
        set_run(r, size=10, bold=True, color=BLUE, font=HEAD_FONT)
        p1 = cells[1].paragraphs[0]
        p1.paragraph_format.space_before = Pt(6)
        p1.paragraph_format.space_after = Pt(6)
        p1.paragraph_format.line_spacing = 1.2
        r = p1.add_run(title)
        set_run(r, size=10, bold=True, color=NAVY, font=HEAD_FONT)
        p2 = cells[2].paragraphs[0]
        p2.paragraph_format.space_before = Pt(6)
        p2.paragraph_format.space_after = Pt(6)
        p2.paragraph_format.line_spacing = 1.2
        r = p2.add_run(desc)
        set_run(r, size=9, color=GRAY)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("说明：本文档依据第 1–4 讲课件及其讲稿备注整理；文中案例数字均为课件中的教学示例，"
                  "不代表通用工业标准。")
    set_run(r, size=9, color=GRAY)

    doc.add_page_break()


def build(blocks):
    doc = Document()
    setup_styles(doc)
    setup_page(doc)
    doc.core_properties.title = "从一只瓶盖看懂智能制造：《智能制造系统设计与实战》1–4讲初学者精讲笔记"
    doc.core_properties.subject = "智能制造系统设计与实战 课程知识点整理"
    doc.core_properties.author = "课程学习笔记"
    doc.core_properties.keywords = "智能制造;机器视觉;缺陷检测;混淆矩阵;闭环"
    doc.core_properties.comments = "依据第1-4讲课件整理，面向初学者。"

    build_cover(doc)
    doc.add_page_break()
    build_front_matter(doc)

    for block in blocks:
        kind = block[0]
        if kind == "h1":
            doc.add_heading(block[1], level=1)
        elif kind == "h2":
            doc.add_heading(block[1], level=2)
        elif kind == "h3":
            doc.add_heading(block[1], level=3)
        elif kind == "p":
            body_par(doc, block[1])
        elif kind == "lead":
            p = body_par(doc, block[1], indent=False, size=11)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.line_spacing = 1.45
        elif kind == "ul":
            add_list(doc, block[1], numbered=False)
        elif kind == "ol":
            add_list(doc, block[1], numbered=True)
        elif kind == "table":
            add_table(doc, block[1])
        elif kind == "fig":
            add_figure(doc, block[1])
        elif kind == "callout":
            add_callout(doc, block[1], )
        elif kind == "key":
            add_key(doc, block[1])
        elif kind == "pb":
            doc.add_page_break()
        else:
            raise ValueError("unknown block: " + kind)

    normalize_schema_order(doc)
    doc.save(OUT)
    print("saved:", OUT)
    print("tables:", TABLE_NO, "figures:", FIG_NO)


if __name__ == "__main__":
    build(BLOCKS)
