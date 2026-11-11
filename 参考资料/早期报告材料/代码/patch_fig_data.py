# -*- coding: utf-8 -*-
"""Bump font sizes in the two data figures so they stay legible at 16 cm width."""
import io
import re

p = 'make_figures_data.py'
s = io.open(p, encoding='utf-8').read()

# ---- fig_data_flow: bigger text
s = s.replace('center_text(d, (x, y0 + 6, x + bw, y0 + 70), cn, f(44, True), "white")',
              'center_text(d, (x, y0 + 6, x + bw, y0 + 70), cn, f(46, True), "white")')
s = s.replace('center_text(d, (x, y0 + 62, x + bw, y0 + 94), en, f(22), "white")',
              'center_text(d, (x, y0 + 64, x + bw, y0 + 96), en, f(24), "white")')
s = s.replace('center_text(d, (x, y0 + 118, x + bw, y0 + 156), "输入数据", f(26, True), GRAY)',
              'center_text(d, (x, y0 + 112, x + bw, y0 + 152), "输入数据", f(32, True), GRAY)')
s = s.replace('center_text(d, (x + 18, yy, x + bw - 18, yy + 56), line, f(24), (60, 60, 60))',
              'center_text(d, (x + 10, yy, x + bw - 10, yy + 58), line, f(30), (60, 60, 60))')
s = s.replace('center_text(d, (x, y0 + 432, x + bw, y0 + 470), "输出数据", f(26, True), color)',
              'center_text(d, (x, y0 + 428, x + bw, y0 + 468), "输出数据", f(32, True), color)')
s = s.replace('center_text(d, (x + 18, yy, x + bw - 18, yy + 56), line, f(24), (30, 30, 30))',
              'center_text(d, (x + 10, yy, x + bw - 10, yy + 58), line, f(30), (30, 30, 30))')
s = s.replace('"每个环节都有输入与输出：输出的名字、单位、取值范围不确定，下游就用不了", f(30), GRAY)',
              '"每个环节都有输入与输出：输出的名字、单位、取值范围不确定，下游就用不了", f(34), GRAY)')

# ---- fig_timeline: bigger text
s = s.replace('center_text(d, (x - 150, base - 210, x + 150, base - 174), "\u2460 \u2461 \u2462 \u2463 \u2464 \u2465 \u2466"[i * 2], f(30, True), color)',
              'center_text(d, (x - 150, base - 214, x + 150, base - 174), "\u2460 \u2461 \u2462 \u2463 \u2464 \u2465 \u2466"[i * 2], f(34, True), color)')
s = s.replace('center_text(d, (x - 160, base - 168, x + 160, base - 104), name, f(34, True), NAVY)',
              'center_text(d, (x - 165, base - 170, x + 165, base - 100), name, f(38, True), NAVY)')
s = s.replace('center_text(d, (x - 160, base - 96, x + 160, base - 46), t, f(28), color)',
              'center_text(d, (x - 165, base - 98, x + 165, base - 44), t, f(32), color)')
s = s.replace('center_text(d, (x - 165, base + 44, x + 165, base + 100), d1, f(25), (50, 50, 50))',
              'center_text(d, (x - 170, base + 42, x + 170, base + 102), d1, f(30), (50, 50, 50))')
s = s.replace('center_text(d, (x - 165, base + 96, x + 165, base + 150), d2, f(23), GRAY)',
              'center_text(d, (x - 170, base + 98, x + 170, base + 156), d2, f(28), GRAY)')
s = s.replace('"每经过一个时刻，都留下一条带时间戳的数据；没有时间戳，事后就无法判断谁先谁后", f(28), GRAY)',
              '"每经过一个时刻，都留下一条带时间戳的数据；没有时间戳，事后就无法判断谁先谁后", f(32), GRAY)')

io.open(p, 'w', encoding='utf-8').write(s)
print('fonts bumped')
