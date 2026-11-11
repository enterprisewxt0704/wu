import io
from openpyxl import load_workbook

path = r"C:\Users\wuxuanting\Downloads\创新实践\创新实践\智能制造系统设计与实战_课程考核评分表.xlsx"
wb = load_workbook(path, data_only=False)
out = []
for ws in wb.worksheets:
    out.append(f"===== SHEET: {ws.title}  dims={ws.dimensions} =====")
    out.append("merged: " + ", ".join(str(r) for r in ws.merged_cells.ranges))
    for row in ws.iter_rows():
        vals = [f"{c.coordinate}={c.value!r}" for c in row if c.value is not None]
        if vals:
            out.append(" ; ".join(vals))
    out.append("")

with io.open("task_xlsx.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("OK lines:", len(out))
