# -*- coding: utf-8 -*-
"""Fix: backticks are not supported by add_rich; widen a header."""
import io

p = 'content_data.py'
s = io.open(p, encoding='utf-8').read()
n = s.count('`')
s = s.replace('`', '')
s = s.replace('"headers": ["异常类型", "发生环节", "触发数据", "处置动作", "要记的字段"]',
              '"headers": ["异常类型", "发生环节", "触发数据", "处置动作", "字段"]')
io.open(p, 'w', encoding='utf-8').write(s)
print('removed backticks:', n)
