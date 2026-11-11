# -*- coding: utf-8 -*-
"""Render the data handbook docx to HTML for visual checking."""
import io
import sys

sys.path.insert(0, r"D:\智造创新实践\build")

src = io.open(r"D:\智造创新实践\build\preview_html.py", encoding="utf-8").read()
src = src.replace(r'PATH = r"D:\智造创新实践\从一只瓶盖看懂智能制造_四讲知识点整理.docx"',
                  r'PATH = r"D:\智造创新实践\智能制造课件数据与指标定义手册.docx"')
src = src.replace(r'OUT = r"D:\智造创新实践\build\preview.html"',
                  r'OUT = r"D:\智造创新实践\build\preview_data.html"')
exec(compile(src, "preview_html_data", "exec"))
