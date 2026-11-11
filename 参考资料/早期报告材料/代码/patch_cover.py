# -*- coding: utf-8 -*-
import io

p = 'build_docx_data.py'
s = io.open(p, encoding='utf-8').read()

reps = [
    ("智能制造系统设计与实战　·　四讲知识点整理", "智能制造系统设计与实战　·　数据与指标手册"),
    ('add_rich(p, "从一只瓶盖\\n看懂智能制造", size=34, color=NAVY, font=HEAD_FONT, bold=True)',
     'add_rich(p, "一张表看懂\\n检测数据", size=34, color=NAVY, font=HEAD_FONT, bold=True)'),
    ('r = p.add_run("《智能制造系统设计与实战》第 1–4 讲　初学者精讲笔记")',
     'r = p.add_run("《智能制造系统设计与实战》第 1–4 讲　数据与指标定义手册")'),
    ('add_rich(p, "把系统、视觉、AI 与落地串成一条主线：\\n"\n'
     '                "感知 → 分析 → 决策 → 执行 → 反馈。", size=12, color=GRAY, font=HEAD_FONT)',
     'add_rich(p, "课件里出现的每一个数，都有名字、单位和来历：\\n"\n'
     '                "采集 → 计算 → 判断 → 处置 → 记录。", size=12, color=GRAY, font=HEAD_FONT)'),
    ('stats = [("4", "讲课件提炼"), ("3", "张系统示意图"), ("15", "个入门易错点"), ("15", "道自测题")]',
     'stats = [("120+", "条数据定义"), ("40+", "张对照表"), ("15", "个数据陷阱"), ("15", "道自测题")]'),
    ('r = p.add_run("适用对象：零基础学习者、课程复习者、项目设计者")',
     'r = p.add_run("适用对象：零基础学习者、课程复习者、做课程项目的同学")'),
]
for a, b in reps:
    assert a in s, a[:50]
    s = s.replace(a, b)

io.open(p, 'w', encoding='utf-8').write(s)
print('cover patched')
