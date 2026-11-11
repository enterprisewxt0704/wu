import io, sys
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn

path = r"C:\Users\wuxuanting\Downloads\创新实践\创新实践\智能制造系统设计与实战_课程考核任务书.docx"
doc = Document(path)

def iter_block_items(parent):
    parent_elm = parent.element.body
    for child in parent_elm.iterchildren():
        if child.tag == qn('w:p'):
            yield Paragraph(child, parent)
        elif child.tag == qn('w:tbl'):
            yield Table(child, parent)

out = []
for block in iter_block_items(doc):
    if isinstance(block, Paragraph):
        t = block.text.strip()
        if t:
            out.append(f"[{block.style.name}] {t}")
    else:
        out.append("=== TABLE ===")
        for row in block.rows:
            cells = [" ".join(c.text.split()) for c in row.cells]
            out.append(" | ".join(cells))
        out.append("=== END TABLE ===")

with io.open("task_docx.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("OK lines:", len(out))
