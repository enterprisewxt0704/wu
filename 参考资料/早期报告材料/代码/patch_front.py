# -*- coding: utf-8 -*-
import io

p = 'build_docx_data.py'
s = io.open(p, encoding='utf-8').read()

# --- 阅读说明正文
old_body = '''    body_par(doc, "这是一份为初学者准备的课程笔记。它不复制幻灯片，而是把四讲内容重新组织成"
                  "**一条可以讲通的主线**：一只瓶盖怎样被看见、被理解、被决策、被执行，"
                  "最后结果又怎样回到系统。", indent=True)
    body_par(doc, "阅读时不必一次记住所有术语。建议先读第 0 章建立全局印象，再按章节顺序推进；"
                  "每章末尾有“一分钟自测”，第 5 章有完整的速查表、易错点和 15 道自测题。", indent=True)'''
new_body = '''    body_par(doc, "这是一份**数据与指标的定义手册**：把第 1–4 讲里出现过的每一个数值、每一个量，"
                  "按“叫什么、怎么算、单位是什么、谁在用、用错会怎样”写成条目，"
                  "方便复习时查、做课程项目时直接抄。", indent=True)
    body_par(doc, "条目分两类，用符号区分：**★ 课件明确提到**（课堂原话或幻灯片原值）；"
                  "**☆ 课件未展开、但工程上一定会用到**（例如曝光时间、像素当量、帧率、OEE），"
                  "按通用工业口径给出，并注明是补充内容。拿不准时以课件为准。", indent=True)'''
assert old_body in s
s = s.replace(old_body, new_body)

# --- 目录条目
old_toc = '''    toc = [
        ("第0章", "先看全局：一只瓶盖的智能制造之旅", "四讲知识地图与五句话总结"),
        ("第1章", "认识智能制造：不是聪明的设备，而是会闭环的系统", "六要素、三种流、五模块、规则与 AI"),
        ("第2章", "感知：机器怎样“看见”瓶盖", "成像、像素、传统视觉七步链、人工特征"),
        ("第3章", "分析：机器怎样“看懂”瓶盖", "四类 AI 任务、混淆矩阵、精确率与召回率"),
        ("第4章", "决策、执行与反馈：让模型真正改变生产", "阈值、位置跟踪、节拍、异常降级、数据记录"),
        ("第5章", "一页纸复习、易错点与自测", "速查表、15 个易错点、15 道题、4 个练习"),
        ("附　录", "术语中英对照表", "按出现顺序整理，便于查证"),
    ]'''
new_toc = '''    toc = [
        ("第0章", "数据是什么：四种角色与五段流向", "事实／计算量／参数／记录，一条数据的来历"),
        ("第1章", "系统、流程与处置数据", "六要素、缺陷类别、三个去处、三种流、四个动作"),
        ("第2章", "成像与视觉数据", "图像矩阵、成像参数、七步链中间量、人工特征"),
        ("第3章", "模型输出与评价指标", "class／confidence／bbox、混淆矩阵、四项指标怎么算"),
        ("第4章", "决策、执行与运行数据", "阈值、时间、位置、异常、运行记录与运营指标"),
        ("第5章", "数据流、命名与 15 个陷阱", "采集到回流、字段命名、口径与采样"),
        ("第6章", "速查与自测", "数据项总表、公式卡、单位表、15 道题"),
        ("附　录", "符号、单位与记录模板", "TP/FP/FN/TN、常用单位、一条记录长什么样"),
    ]'''
assert old_toc in s
s = s.replace(old_toc, new_toc)

# --- 目录表列宽（标题变长）
s = s.replace('for c, w in zip(cells, (2.2, 7.4, 6.8)):',
              'for c, w in zip(cells, (2.0, 7.6, 6.8)):')

# --- 说明行
old_note = '''    r = p.add_run("说明：本文档依据第 1–4 讲课件及其讲稿备注整理；文中案例数字均为课件中的教学示例，"
                  "不代表通用工业标准。")'''
new_note = '''    r = p.add_run("说明：本文档依据第 1–4 讲课件及其讲稿备注整理。标 ★ 的数值取自课件原例，"
                  "均为教学示例，不代表通用工业标准；标 ☆ 的条目为工程常识补充，使用时需按实际产线标定。")'''
assert old_note in s
s = s.replace(old_note, new_note)

# --- 文档属性
old_props = '''    doc.core_properties.title = "从一只瓶盖看懂智能制造：《智能制造系统设计与实战》1–4讲初学者精讲笔记"
    doc.core_properties.subject = "智能制造系统设计与实战 课程知识点整理"
    doc.core_properties.author = "课程学习笔记"
    doc.core_properties.keywords = "智能制造;机器视觉;缺陷检测;混淆矩阵;闭环"
    doc.core_properties.comments = "依据第1-4讲课件整理，面向初学者。"'''
new_props = '''    doc.core_properties.title = "智能制造课件数据与指标定义手册：《智能制造系统设计与实战》1–4讲"
    doc.core_properties.subject = "智能制造系统设计与实战 数据与指标定义"
    doc.core_properties.author = "课程学习笔记"
    doc.core_properties.keywords = "数据字典;机器视觉;混淆矩阵;阈值;节拍;指标定义"
    doc.core_properties.comments = "依据第1-4讲课件整理，含课件值(★)与工程补充(☆)。"'''
assert old_props in s
s = s.replace(old_props, new_props)

io.open(p, 'w', encoding='utf-8').write(s)
print('front matter + props patched')
