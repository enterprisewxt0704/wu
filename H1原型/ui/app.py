# -*- coding: utf-8 -*-
"""H1 可视化操作台（tkinter 单窗口）。

界面只做三件事：
    1. 让操作员看见“这一只瓶盖是什么、系统准备怎么处置”；
    2. 让评委看见“两个模型分别怎么想、为什么融合成这个结论”；
    3. 让后续开发知道“输入/执行/记录/界面”四个插座在哪里。

布局
----
顶部   标题 + 融合方式 + 阈值预设 + 操作按钮
状态条 相机 / 光电 / 执行 / 深度模型 / 记录文件
左侧   原图 + 圆心/缺陷框
右侧   大结论卡 + 风险条 + 两个模型卡 + 计数
底部   Tab：记录表 / 诊断详情 / 参数 / 接口说明

线程模型
--------
Tkinter 只能在主线程碰控件；所有检测在后台线程跑，通过 queue 把结果送回来，
主线程用 after(80) 轮询队列。这样连续仿真时窗口不会卡死。
"""

from __future__ import annotations

import argparse
import os
import queue
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from core import config as C
from core.pipeline import Inspector, InspectionResult
from core.types import ACTION_CN, CLASS_CN, CLASSES, FEATURE_CN
from devices.sim import SimulatedLine

# ----------------------------------------------------------------------------
# 配色 / 字体
# ----------------------------------------------------------------------------
BG = "#eef2f6"
CARD = "#ffffff"
INK = "#1f2933"
MUTED = "#6b7785"
LINE = "#d5dde5"
GREEN = "#12805c"
AMBER = "#b45309"
RED = "#c0392b"
BLUE = "#1d4ed8"
ACTION_COLOR = {"PASS": GREEN, "RECHECK": AMBER, "REJECT": RED}
ACTION_BG = {"PASS": "#e6f4ee", "RECHECK": "#fff4e0", "REJECT": "#fdecea"}

MODE_LABEL = {
    "M3 双模型融合（推荐）": "M3",
    "M1 特征门卫 + 深度分诊": "M1",
    "M2 深度门卫 + 特征分诊": "M2",
    "M0 仅特征（对照）": "M0",
}
MODE_TIP = {
    "M3": "risk=max(异常分,深度缺陷分)，任一模型报警都不放行；类别按概率加权。",
    "M1": "异常分当门卫；异常分低时由深度模型报类别。",
    "M2": "深度缺陷分当门卫；它放行后由特征模型做最终分类。",
    "M0": "不加载 torch，复现原特征识别基线（97.5%）。",
}


def _fmt_ts(t: float) -> str:
    if not t:
        return ""
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"


class App(tk.Tk):
    def __init__(self, config_path: Optional[str] = None, smoke: bool = False):
        super().__init__()
        self.smoke = smoke
        self.cfg = C.load_config(config_path)
        self.inspector = Inspector.from_config(config_path, dispatcher_mode="sim",
                                               record=True, keep_debug=True)
        self.q: "queue.Queue[tuple]" = queue.Queue()
        self.stop_event = threading.Event()
        # 单线程推理池：CUDA 上下文/预热只在一个线程里建立，避免“换线程后首张又花 1~16 秒”。
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self.running = False
        self.current: Optional[InspectionResult] = None
        self.tree_results: Dict[str, InspectionResult] = {}
        self.counts = Counter()
        self.unknown = 0
        self.act_fail = 0
        self.total_ms = 0.0
        self.labeled = 0
        self.correct = 0
        self.last_result_at = 0.0

        self.title("瓶盖智能质检 · H1 原型（特征 + MobileNetV3 融合）")
        self.geometry(self.cfg.get("ui", {}).get("window", "1280x820"))
        self.minsize(1120, 720)
        self.configure(bg=BG)

        self._init_fonts()
        self._init_style()
        self._build_ui()
        self._set_mode(self.inspector.fusion_mode)
        self._update_risk_bar()

        self.after(100, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if not smoke:
            self.after(300, self._preload_deep)
        else:
            self.after(1200, self._smoke_done)

    # ==================================================================
    # 外观
    # ==================================================================
    def _init_fonts(self) -> None:
        families = set(tkfont.families())
        base = "Microsoft YaHei"
        for name in ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei"):
            if name in families:
                base = name
                break
        self.base_font = base
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family=base, size=10)
            except Exception:
                pass
        self.f_title = (base, 18, "bold")
        self.f_sub = (base, 9)
        self.f_h = (base, 11, "bold")
        self.f_big = (base, 30, "bold")
        self.f_body = (base, 10)
        self.f_small = (base, 9)
        self.f_mono = ("Consolas", 9)

    def _init_style(self) -> None:
        st = ttk.Style(self)
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure(".", font=(self.base_font, 10), background=BG, foreground=INK)
        st.configure("TFrame", background=BG)
        st.configure("Card.TFrame", background=CARD)
        st.configure("TLabel", background=BG, foreground=INK)
        st.configure("Card.TLabel", background=CARD, foreground=INK)
        st.configure("Muted.TLabel", background=BG, foreground=MUTED)
        st.configure("MutedCard.TLabel", background=CARD, foreground=MUTED)
        st.configure("TButton", padding=(10, 5), font=(self.base_font, 10))
        st.configure("Primary.TButton", background=BLUE, foreground="white")
        st.map("Primary.TButton", background=[("active", "#1746b8")])
        st.configure("Danger.TButton", background=RED, foreground="white")
        st.map("Danger.TButton", background=[("active", "#a93226")])
        st.configure("TNotebook", background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", padding=(16, 7), font=(self.base_font, 10))
        st.map("TNotebook.Tab", background=[("selected", CARD)])
        st.configure("Treeview", rowheight=24, font=(self.base_font, 9),
                     background=CARD, fieldbackground=CARD)
        st.configure("Treeview.Heading", font=(self.base_font, 9, "bold"))
        st.configure("TProgressbar", troughcolor="#dfe7ef", background=GREEN)

    # ==================================================================
    # 布局
    # ==================================================================
    def _build_ui(self) -> None:
        self._build_header()
        self._build_toolbar()
        self._build_status()
        self._build_main()
        self._build_bottom()
        self._build_footer()

    def _build_header(self) -> None:
        head = tk.Frame(self, bg=CARD, height=62)
        head.pack(side="top", fill="x")
        head.pack_propagate(False)

        left = tk.Frame(head, bg=CARD)
        left.pack(side="left", padx=16, pady=6)
        tk.Label(left, text="瓶盖智能质检 · H1 原型", font=self.f_title,
                 bg=CARD, fg=INK).pack(anchor="w")
        tk.Label(left, text="对射光电触发 · 工业相机取图 · 特征异常分 + MobileNetV3 双模型融合 · 三通道处置",
                 font=self.f_sub, bg=CARD, fg=MUTED).pack(anchor="w")

        right = tk.Frame(head, bg=CARD)
        right.pack(side="right", padx=14, pady=6)

        tk.Label(right, text="融合方式", bg=CARD, fg=MUTED,
                 font=self.f_small).grid(row=0, column=0, sticky="e", padx=(0, 4))
        self.mode_var = tk.StringVar(value=self._mode_label(self.inspector.fusion_mode))
        self.mode_box = ttk.Combobox(right, textvariable=self.mode_var, width=24,
                                     values=list(MODE_LABEL.keys()), state="readonly")
        self.mode_box.grid(row=0, column=1, padx=(0, 10))
        self.mode_box.bind("<<ComboboxSelected>>", self._on_mode)

        tk.Label(right, text="阈值预设", bg=CARD, fg=MUTED,
                 font=self.f_small).grid(row=0, column=2, sticky="e", padx=(0, 4))
        self.preset_var = tk.StringVar(value=self.cfg.get("decision", {}).get("policy", "risk_averse"))
        self.preset_box = ttk.Combobox(right, textvariable=self.preset_var, width=14,
                                       values=list(self.cfg.get("presets", {}).keys()),
                                       state="readonly")
        self.preset_box.grid(row=0, column=3, padx=(0, 10))
        self.preset_box.bind("<<ComboboxSelected>>", self._on_preset)

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg="#f7f9fb", height=46,
                       highlightbackground=LINE, highlightthickness=1)
        bar.pack(side="top", fill="x")
        bar.pack_propagate(False)
        inner = tk.Frame(bar, bg="#f7f9fb")
        inner.pack(side="left", padx=12, pady=6)
        self.btn_open = ttk.Button(inner, text="打开图片", command=self._open_image)
        self.btn_open.pack(side="left", padx=3)
        self.btn_folder = ttk.Button(inner, text="打开文件夹", command=self._open_folder)
        self.btn_folder.pack(side="left", padx=3)
        self.btn_sim = ttk.Button(inner, text="▶ 连续仿真", style="Primary.TButton",
                                  command=self._start_sim)
        self.btn_sim.pack(side="left", padx=(12, 3))
        self.btn_stop = ttk.Button(inner, text="■ 停止", style="Danger.TButton",
                                   command=self._stop)
        self.btn_stop.pack(side="left", padx=3)
        self.btn_val = ttk.Button(inner, text="跑验证集", command=lambda: self._run_split("val"))
        self.btn_val.pack(side="left", padx=(12, 3))
        self.btn_test = ttk.Button(inner, text="跑测试集",
                                   command=lambda: self._run_split("test_unlabeled"))
        self.btn_test.pack(side="left", padx=3)
        tk.Label(bar, text="提示：连续仿真会用 val + test 的图片循环，模拟光电触发与 900 ms 节拍。",
                 bg="#f7f9fb", fg=MUTED, font=self.f_small).pack(side="right", padx=12)

    def _build_status(self) -> None:
        bar = tk.Frame(self, bg="#dde5ee", height=28)
        bar.pack(side="top", fill="x")
        bar.pack_propagate(False)
        self.status_vars: Dict[str, tk.StringVar] = {}
        chips = [("camera", "相机"), ("sensor", "光电"), ("act", "执行"),
                 ("deep", "深度模型"), ("record", "记录")]
        for key, label in chips:
            var = tk.StringVar(value="…")
            self.status_vars[key] = var
            tk.Label(bar, text=f"{label}：", bg="#dde5ee", fg=MUTED,
                     font=self.f_small).pack(side="left", padx=(12, 0))
            tk.Label(bar, textvariable=var, bg="#dde5ee", fg=INK,
                     font=(self.base_font, 9, "bold")).pack(side="left")
        self.status_vars["camera"].set("仿真图片源")
        self.status_vars["sensor"].set("仿真触发")
        self.status_vars["act"].set("模拟三通道")
        self.status_vars["deep"].set("加载中…")
        self.status_vars["record"].set(str(C.out_dir(self.cfg) / "records.csv"))

    def _build_main(self) -> None:
        main = tk.Frame(self, bg=BG)
        main.pack(side="top", fill="both", expand=True, padx=12, pady=(10, 6))
        main.columnconfigure(0, weight=3, minsize=580)
        main.columnconfigure(1, weight=2, minsize=430)
        main.rowconfigure(0, weight=1)

        # ---------------- 左侧：图像 ----------------
        left = tk.Frame(main, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(left, text="① 相机画面 / 缺陷位置", font=self.f_h,
                 bg=CARD, fg=INK).pack(anchor="w", padx=12, pady=(8, 2))
        self.canvas = tk.Canvas(left, width=560, height=430, bg="#f8fafc",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=4)
        self.hint = tk.Label(left, text="打开一张图片，或点“连续仿真”让产线跑起来",
                             font=self.f_small, bg=CARD, fg=MUTED)
        self.hint.pack(anchor="w", padx=12, pady=(0, 8))

        # ---------------- 右侧：结论 ----------------
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        self.decision_card = tk.Frame(right, bg="#f1f5f9", highlightbackground=LINE,
                                      highlightthickness=1)
        self.decision_card.grid(row=0, column=0, sticky="ew")
        self.decision_label = tk.Label(self.decision_card, text="等待检测", font=self.f_big,
                                       bg="#f1f5f9", fg=MUTED)
        self.decision_label.pack(anchor="w", padx=14, pady=(8, 0))
        self.decision_reason = tk.Label(self.decision_card, text="—", font=self.f_body,
                                        bg="#f1f5f9", fg=INK, justify="left",
                                        wraplength=430, anchor="w")
        self.decision_reason.pack(anchor="w", fill="x", padx=14, pady=(0, 2))
        self.decision_detail = tk.Label(self.decision_card, text="", font=self.f_small,
                                        bg="#f1f5f9", fg=MUTED, justify="left",
                                        wraplength=430, anchor="w")
        self.decision_detail.pack(anchor="w", fill="x", padx=14, pady=(0, 8))

        # 风险条
        risk_box = tk.Frame(right, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        risk_box.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        tk.Label(risk_box, text="② 融合风险（max 异常分 / 深度缺陷分）", font=self.f_h,
                 bg=CARD, fg=INK).pack(anchor="w", padx=12, pady=(6, 0))
        self.risk_canvas = tk.Canvas(risk_box, height=46, bg=CARD, highlightthickness=0)
        self.risk_canvas.pack(fill="x", padx=12, pady=(2, 6))
        self.risk_canvas.bind("<Configure>", lambda _e: self._update_risk_bar())

        # 两个模型卡
        cards = tk.Frame(right, bg=BG)
        cards.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)
        self.feat_card, self.feat_vars = self._model_card(cards, 0, "特征模型（numpy）")
        self.deep_card, self.deep_vars = self._model_card(cards, 1, "深度模型（MobileNetV3）")

        # 计数
        counts = tk.Frame(right, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        counts.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        tk.Label(counts, text="④ 本次运行计数", font=self.f_h, bg=CARD, fg=INK).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=12, pady=(6, 2))
        self.count_vars = {}
        defs = [("total", "总数"), ("pass", "放行"), ("recheck", "复检"),
                ("reject", "剔除"), ("unknown", "未知缺陷"), ("act_fail", "执行失败"),
                ("avg", "平均耗时"), ("rate", "节拍")]
        for i, (key, label) in enumerate(defs):
            r, c = 1 + i // 4, i % 4
            tk.Label(counts, text=label, bg=CARD, fg=MUTED,
                     font=self.f_small).grid(row=r * 2, column=c, sticky="e", padx=(10, 2), pady=1)
            var = tk.StringVar(value="0")
            self.count_vars[key] = var
            tk.Label(counts, textvariable=var, bg=CARD, fg=INK,
                     font=(self.base_font, 11, "bold")).grid(row=r * 2 + 1, column=c,
                                                             sticky="w", padx=(0, 8), pady=(0, 4))
        self.acc_var = tk.StringVar(value="有标签准确率：—（跑验证集/测试集后显示）")
        tk.Label(counts, textvariable=self.acc_var, bg=CARD, fg=GREEN,
                 font=(self.base_font, 10, "bold")).grid(row=6, column=0, columnspan=4,
                                                        sticky="w", padx=12, pady=(2, 6))

    def _model_card(self, parent, col: int, title: str):
        frame = tk.Frame(parent, bg=CARD, highlightbackground=LINE, highlightthickness=1)
        frame.grid(row=0, column=col, sticky="nsew", padx=(0, 6) if col == 0 else (6, 0))
        tk.Label(frame, text=title, font=self.f_h, bg=CARD, fg=INK).pack(
            anchor="w", padx=10, pady=(6, 0))
        v_pred = tk.StringVar(value="—")
        v_sub = tk.StringVar(value="等待检测")
        v_prob = tk.StringVar(value="")
        tk.Label(frame, textvariable=v_pred, font=(self.base_font, 15, "bold"),
                 bg=CARD, fg=BLUE).pack(anchor="w", padx=10)
        tk.Label(frame, textvariable=v_sub, font=self.f_small, bg=CARD, fg=MUTED,
                 wraplength=200, justify="left").pack(anchor="w", padx=10)
        tk.Label(frame, textvariable=v_prob, font=self.f_mono, bg=CARD, fg=MUTED,
                 wraplength=200, justify="left").pack(anchor="w", padx=10, pady=(0, 6))
        return frame, {"pred": v_pred, "sub": v_sub, "prob": v_prob}

    def _build_bottom(self) -> None:
        self.nb = ttk.Notebook(self)
        self.nb.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 4))

        # ---- 记录 ----
        tab = tk.Frame(self.nb, bg=CARD)
        self.nb.add(tab, text="记录")
        top = tk.Frame(tab, bg=CARD)
        top.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(top, text="检测记录（最新在上；点一行看诊断详情）", bg=CARD, fg=MUTED,
                 font=self.f_small).pack(side="left")
        ttk.Button(top, text="导出 CSV", command=self._export).pack(side="right", padx=2)
        ttk.Button(top, text="复检确认", command=self._confirm_recheck).pack(side="right", padx=2)

        cols = ("time", "file", "action", "class", "risk", "anomaly", "deep", "ms", "act")
        heads = ("时间", "文件", "结论", "类别", "风险", "特征分", "深度缺陷分", "耗时ms", "执行")
        widths = (80, 130, 64, 64, 56, 60, 80, 60, 150)
        self.tree = ttk.Treeview(tab, columns=cols, show="headings", height=6)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center", stretch=(c == "file"))
        ys = ttk.Scrollbar(tab, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 6))
        ys.pack(side="left", fill="y", padx=(0, 8), pady=(0, 6))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # ---- 诊断 ----
        tab2 = tk.Frame(self.nb, bg=CARD)
        self.nb.add(tab2, text="诊断详情")
        self.diag_text = tk.Text(tab2, height=7, bg=CARD, fg=INK, relief="flat",
                                 font=self.f_mono, wrap="word")
        self.diag_text.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=6)
        right = tk.Frame(tab2, bg=CARD)
        right.pack(side="right", fill="y", padx=(0, 8), pady=6)
        tk.Label(right, text="摊平偏差图（红=亮缺陷，蓝=暗缺陷）", bg=CARD, fg=MUTED,
                 font=self.f_small).pack(anchor="w")
        self.unwrap_canvas = tk.Canvas(right, width=360, height=100, bg="#f8fafc",
                                       highlightthickness=1, highlightbackground=LINE)
        self.unwrap_canvas.pack(anchor="w", pady=(2, 0))

        # ---- 参数 ----
        tab3 = tk.Frame(self.nb, bg=CARD)
        self.nb.add(tab3, text="参数")
        self._build_params(tab3)

        # ---- 接口 ----
        tab4 = tk.Frame(self.nb, bg=CARD)
        self.nb.add(tab4, text="接口说明")
        self._build_interface_tab(tab4)

    def _build_params(self, parent) -> None:
        wrap = tk.Frame(parent, bg=CARD)
        wrap.pack(fill="both", expand=True, padx=12, pady=8)
        d = self.cfg.get("decision", {})
        self.low_var = tk.DoubleVar(value=float(d.get("low", 0.3607)))
        self.high_var = tk.DoubleVar(value=float(d.get("high", 0.8371)))
        self.conf_var = tk.DoubleVar(value=float(d.get("conf_min", 0.97)))
        self.weight_var = tk.DoubleVar(value=float(self.cfg.get("fusion", {}).get("weight_deep", 0.5)))

        rows = [
            ("异常分下限 low", self.low_var, 0.0, 1.0, "低于它才可能放行"),
            ("异常分上限 high", self.high_var, 0.0, 1.0, "高于它直接剔除"),
            ("置信度下限 conf_min", self.conf_var, 0.0, 1.0, "低于它标“未知缺陷”"),
            ("融合权重（深度）", self.weight_var, 0.0, 1.0, "M3 类别概率：深度权重，特征=1-w"),
        ]
        self.param_labels = {}
        for i, (label, var, lo, hi, tip) in enumerate(rows):
            tk.Label(wrap, text=label, bg=CARD, fg=INK, font=self.f_h).grid(
                row=i * 2, column=0, sticky="w", padx=(0, 10), pady=(6, 0))
            val = tk.Label(wrap, text=f"{var.get():.2f}", bg=CARD, fg=BLUE,
                           font=(self.base_font, 10, "bold"))
            val.grid(row=i * 2, column=1, sticky="w")
            self.param_labels[label] = (val, var)
            sc = ttk.Scale(wrap, from_=lo, to=hi, variable=var, orient="horizontal",
                           length=320, command=lambda _v, v=var, l=label: self._on_param_move(l, v))
            sc.grid(row=i * 2 + 1, column=0, columnspan=2, sticky="w", pady=(0, 2))
            tk.Label(wrap, text=tip, bg=CARD, fg=MUTED, font=self.f_small).grid(
                row=i * 2 + 1, column=2, sticky="w", padx=(8, 0))

        row = len(rows) * 2
        ttk.Button(wrap, text="应用阈值", style="Primary.TButton",
                   command=self._apply_params).grid(row=row, column=0, sticky="w", pady=(10, 0))
        ttk.Button(wrap, text="保存到 config.json",
                   command=self._save_config).grid(row=row, column=1, sticky="w", pady=(10, 0))
        tk.Label(wrap, text="提示：阈值只在本次运行生效时不用保存；答辩演示可先选“demo_recheck”预设看到复检分支。",
                 bg=CARD, fg=MUTED, font=self.f_small).grid(row=row + 1, column=0, columnspan=3,
                                                            sticky="w", pady=(8, 0))

    def _build_interface_tab(self, parent) -> None:
        txt = tk.Text(parent, height=7, bg=CARD, fg=INK, relief="flat", font=self.f_mono, wrap="word")
        txt.pack(fill="both", expand=True, padx=10, pady=8)
        txt.insert("1.0", INTERFACE_TEXT)
        txt.configure(state="disabled")

    def _build_footer(self) -> None:
        foot = tk.Frame(self, bg=BG, height=26)
        foot.pack(side="bottom", fill="x")
        foot.pack_propagate(False)
        self.footer = tk.StringVar(value="就绪。建议先跑“验证集”看指标，再点“连续仿真”演示闭环。")
        tk.Label(foot, textvariable=self.footer, bg=BG, fg=MUTED,
                 font=self.f_small, anchor="w").pack(side="left", padx=12)
        self.progress = ttk.Progressbar(foot, length=180, mode="determinate")
        self.progress.pack(side="right", padx=12, pady=4)
        self.progress_label = tk.StringVar(value="")
        tk.Label(foot, textvariable=self.progress_label, bg=BG, fg=MUTED,
                 font=self.f_small).pack(side="right")

    # ==================================================================
    # 事件：模式 / 预设 / 参数
    # ==================================================================
    def _mode_label(self, mode: str) -> str:
        for label, m in MODE_LABEL.items():
            if m == mode:
                return label
        return "M3 双模型融合（推荐）"

    def _set_mode(self, mode: str) -> None:
        self.inspector.set_fusion_mode(mode)
        self.mode_var.set(self._mode_label(mode))
        self.footer.set(f"融合方式：{mode} — {MODE_TIP.get(mode, '')}")

    def _on_mode(self, _event=None) -> None:
        mode = MODE_LABEL.get(self.mode_var.get(), "M3")
        self._set_mode(mode)
        if self.current:
            # 同一张图立刻按新融合方式重算，方便对比
            self._run_single(self.current.path, quiet=True)

    def _on_preset(self, _event=None) -> None:
        name = self.preset_var.get()
        p = self.cfg.get("presets", {}).get(name)
        if not p:
            return
        self.low_var.set(float(p.get("low", self.low_var.get())))
        self.high_var.set(float(p.get("high", self.high_var.get())))
        self.conf_var.set(float(p.get("conf_min", self.conf_var.get())))
        self._apply_params()
        self.footer.set(f"已套用预设 {name}：low={self.low_var.get():.3f}，high={self.high_var.get():.3f}")

    def _on_param_move(self, label: str, var) -> None:
        pair = self.param_labels.get(label)
        if pair:
            pair[0].configure(text=f"{float(var.get()):.2f}")

    def _apply_params(self) -> None:
        low, high = float(self.low_var.get()), float(self.high_var.get())
        if low >= high:
            messagebox.showwarning("阈值不合法", "low 必须小于 high。")
            return
        self.inspector.update_decision(low, high, float(self.conf_var.get()))
        self.inspector.cfg.setdefault("fusion", {})["weight_deep"] = float(self.weight_var.get())
        self._update_risk_bar()
        self.footer.set(f"阈值已应用：low={low:.3f}，high={high:.3f}，conf_min={self.conf_var.get():.3f}")

    def _save_config(self) -> None:
        self._apply_params()
        self.inspector.save_config()
        messagebox.showinfo("已保存", "参数已写入 H1原型/config.json")

    # ==================================================================
    # 运行：单图 / 批量 / 连续仿真
    # ==================================================================
    def _open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="选择瓶盖图片",
            filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp"), ("全部", "*.*")])
        if path:
            self._run_single(path)

    def _open_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择图片文件夹")
        if not folder:
            return
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        paths = []
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if os.path.splitext(name)[1].lower() in exts:
                    paths.append(os.path.join(root, name))
        paths.sort()
        if not paths:
            messagebox.showinfo("没有图片", "这个文件夹里没有 png/jpg 图片。")
            return
        self._run_batch(paths, f"文件夹（{len(paths)} 张）")

    def _run_split(self, split: str) -> None:
        paths = C.split_images(self.cfg, split)
        if not paths:
            messagebox.showwarning("没有数据", f"找不到 {split} 的图片，检查 config.json 的 dataset_root。")
            return
        self._run_batch(paths, f"{split}（{len(paths)} 张）")

    def _start_sim(self) -> None:
        # 用 val + test 交替，保证正常/缺口/划痕/污渍都能出现
        paths = C.split_images(self.cfg, "val") + C.split_images(self.cfg, "test_unlabeled")
        if not paths:
            messagebox.showwarning("没有数据", "找不到仿真图片。")
            return
        if self.running:
            messagebox.showinfo("正在运行", "先点“停止”再启动新的仿真。")
            return
        self.running = True
        self.stop_event.clear()
        self._reset_run(clear_tree=True)
        self.footer.set("连续仿真运行中…（900 ms/只，模拟传送带节拍）")

        def work():
            try:
                line = SimulatedLine(paths, self.inspector, self.cfg, loop=True)
                line.run(lambda r: self.q.put(("result", r)),
                         self.stop_event, max_samples=0,
                         on_done=lambda st: self.q.put(("done", f"仿真结束：{st.as_dict()}")))
            except Exception as exc:  # noqa: BLE001
                self.q.put(("error", f"仿真异常：{type(exc).__name__}: {exc}"))
            finally:
                self.q.put(("running", False))

        self.pool.submit(work)

    def _run_single(self, path: str, quiet: bool = False) -> None:
        if self.running:
            return
        self.running = True
        self.footer.set(f"正在检测 {os.path.basename(path)}…")

        def work():
            try:
                res = self.inspector.inspect(path, dispatch=True)
                self.q.put(("result", res))
                self.q.put(("done", f"单张完成：{os.path.basename(path)}"))
            except Exception as exc:  # noqa: BLE001
                self.q.put(("error", f"检测异常：{type(exc).__name__}: {exc}"))
            finally:
                self.q.put(("running", False))

        self.pool.submit(work)

    def _run_batch(self, paths: List[str], title: str) -> None:
        if self.running:
            messagebox.showinfo("正在运行", "先点“停止”再开始新的批量任务。")
            return
        self.running = True
        self.stop_event.clear()
        self._reset_run(clear_tree=True)
        total = len(paths)
        self.progress.configure(maximum=total, value=0)
        self.footer.set(f"{title}：开始…")
        self.q.put(("status", "正在加载深度模型（首次较慢）…"))

        def work():
            try:
                if self.cfg.get("deep", {}).get("enabled", True):
                    self.inspector.deep.load()
                    if self.inspector.deep.available:
                        self.inspector.deep.warmup()
                for i, p in enumerate(paths, 1):
                    if self.stop_event.is_set():
                        break
                    res = self.inspector.inspect(p, dispatch=False)
                    self.q.put(("result", res))
                    self.q.put(("progress", i, total))
                self.q.put(("done", f"{title} 完成"))
            except Exception as exc:  # noqa: BLE001
                self.q.put(("error", f"批量异常：{type(exc).__name__}: {exc}"))
            finally:
                self.q.put(("running", False))

        self.pool.submit(work)

    def _stop(self) -> None:
        if self.running:
            self.stop_event.set()
            self.footer.set("正在停止…（当前这一只检测完就停）")
        else:
            self.footer.set("当前没有运行中的任务。")

    def _preload_deep(self) -> None:
        def work():
            if not self.cfg.get("deep", {}).get("enabled", True):
                self.q.put(("deep_ready", False, "config 已关闭深度模型"))
                return
            ok = self.inspector.deep.load()
            if ok:
                # 预热一次，避免第一只瓶盖的 CUDA 初始化耗时把平均节拍拉偏
                self.inspector.deep.warmup()
                self.q.put(("deep_ready", True, self.inspector.deep.device_text()))
            else:
                self.q.put(("deep_ready", False, self.inspector.deep.error or "不可用"))

        self.pool.submit(work)

    # ==================================================================
    # 队列 -> 界面
    # ==================================================================
    def _poll(self) -> None:
        processed = 0
        try:
            while processed < 60:
                msg = self.q.get_nowait()
                processed += 1
                kind = msg[0]
                if kind == "result":
                    self._apply_result(msg[1])
                elif kind == "progress":
                    i, n = msg[1], msg[2]
                    self.progress.configure(value=i)
                    self.progress_label.set(f"{i}/{n}")
                elif kind == "status":
                    self.footer.set(msg[1])
                elif kind == "done":
                    self.footer.set(msg[1])
                    self.progress_label.set("")
                elif kind == "error":
                    self.footer.set(msg[1])
                    messagebox.showerror("运行出错", msg[1])
                elif kind == "deep_ready":
                    ok, text = msg[1], msg[2]
                    short = text if len(text) <= 24 else text[:24] + "…"
                    self.status_vars["deep"].set(("就绪 · " if ok else "不可用（") + short + ("" if ok else "）"))
                    if not ok:
                        self.footer.set(f"深度模型不可用，已自动回退纯特征：{text}")
                elif kind == "running":
                    self.running = bool(msg[1])
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _apply_result(self, res: InspectionResult) -> None:
        self.current = res
        self.counts[res.action] += 1
        self.total_ms += res.elapsed_ms
        if res.fusion and res.fusion.is_unknown:
            self.unknown += 1
        if res.act and not res.act.ok:
            self.act_fail += 1
        if res.label in CLASSES:
            self.labeled += 1
            if res.label == res.pred:
                self.correct += 1
        self.last_result_at = time.time()

        # 大结论
        action = res.action
        color, bg = ACTION_COLOR.get(action, MUTED), ACTION_BG.get(action, "#f1f5f9")
        self.decision_card.configure(bg=bg)
        self.decision_label.configure(text=f"{action} · {ACTION_CN.get(action, action)}",
                                      bg=bg, fg=color)
        self.decision_reason.configure(text=res.fusion.reason if res.fusion else res.error, bg=bg)
        detail = res.fusion.detail if res.fusion else ""
        if res.act:
            detail += f"　执行：{res.act.command}（{'成功' if res.act.ok else '失败'}，{res.act.latency_ms:.0f} ms）"
        self.decision_detail.configure(text=detail, bg=bg)
        for child in self.decision_card.winfo_children():
            try:
                child.configure(bg=bg)
            except Exception:
                pass

        self._update_risk_bar()
        self._update_model_cards(res)
        self._update_counts()
        self._insert_record(res)
        self._show_image(res)
        self._update_diag(res)
        self.progress_label.set(f"最近：{res.frame_id}")
        self.footer.set(f"{res.frame_id} → {action}｜{res.fusion.reason if res.fusion else ''}")

    def _update_counts(self) -> None:
        total = sum(self.counts.values())
        avg = self.total_ms / total if total else 0.0
        rate = 1000.0 / avg if avg > 0 else 0.0
        self.count_vars["total"].set(str(total))
        self.count_vars["pass"].set(str(self.counts.get("PASS", 0)))
        self.count_vars["recheck"].set(str(self.counts.get("RECHECK", 0)))
        self.count_vars["reject"].set(str(self.counts.get("REJECT", 0)))
        self.count_vars["unknown"].set(str(self.unknown))
        self.count_vars["act_fail"].set(str(self.act_fail))
        self.count_vars["avg"].set(f"{avg:.0f} ms")
        self.count_vars["rate"].set(f"{rate:.1f} 只/s" if avg else "—")
        if self.labeled:
            self.acc_var.set(f"有标签准确率：{self.correct}/{self.labeled} = "
                             f"{self.correct / self.labeled * 100:.1f}%")
        else:
            self.acc_var.set("有标签准确率：—（跑验证集/测试集后显示）")

    def _reset_run(self, clear_tree: bool = False) -> None:
        """开始新一批任务时清零计数，避免把上一次 val/仿真的数字混在一起。"""
        self.counts.clear()
        self.unknown = 0
        self.act_fail = 0
        self.total_ms = 0.0
        self.labeled = 0
        self.correct = 0
        if clear_tree:
            for iid in self.tree.get_children():
                self.tree.delete(iid)
            self.tree_results.clear()
        self._update_counts()

    def _update_model_cards(self, res: InspectionResult) -> None:
        f, d = res.feat, res.deep
        if f:
            self.feat_vars["pred"].set(f"{CLASS_CN.get(f.pred, f.pred)}  {f.conf:.2f}")
            self.feat_vars["sub"].set(f"异常分 {res.anomaly_score:.3f}；最异常：{FEATURE_CN.get(res.worst, res.worst)}")
            self.feat_vars["prob"].set(_prob_text(f.probs))
        else:
            self.feat_vars["pred"].set("—")
            self.feat_vars["sub"].set("无特征结果")
            self.feat_vars["prob"].set("")
        if d and d.available:
            self.deep_vars["pred"].set(f"{CLASS_CN.get(d.pred, d.pred)}  {d.conf:.2f}")
            self.deep_vars["sub"].set(
                f"缺陷分 {d.defect_score:.3f}；{res.backend}；{self.inspector.deep.device_text()}")
            self.deep_vars["prob"].set(_prob_text(d.probs))
        else:
            self.deep_vars["pred"].set("未启用")
            self.deep_vars["sub"].set((d.error if d else "未加载")[:70])
            self.deep_vars["prob"].set("")

    def _insert_record(self, res: InspectionResult) -> None:
        iid = self.tree.insert("", 0, values=(
            _fmt_ts(res.ts_decide),
            res.frame_id,
            res.action,
            CLASS_CN.get(res.pred, res.pred),
            f"{res.risk:.2f}",
            f"{res.anomaly_score:.3f}",
            f"{res.deep.defect_score:.3f}" if res.deep and res.deep.available else "—",
            f"{res.elapsed_ms:.0f}",
            (res.act.command if res.act else "离线"),
        ))
        self.tree_results[iid] = res
        max_rows = int(self.cfg.get("ui", {}).get("recent_rows", 300))
        children = self.tree.get_children()
        if len(children) > max_rows:
            for old in children[max_rows:]:
                self.tree.delete(old)
                self.tree_results.pop(old, None)

    def _on_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if sel:
            res = self.tree_results.get(sel[0])
            if res:
                self.current = res
                self._update_diag(res)
                self._show_image(res)

    # ==================================================================
    # 图像 / 风险条 / 诊断
    # ==================================================================
    def _show_image(self, res: InspectionResult) -> None:
        img = None
        fv = res.features
        if fv and fv.debug.get("image") is not None:
            img = np.asarray(fv.debug["image"], dtype=np.uint8).copy()
        elif res.path and os.path.exists(res.path):
            try:
                with Image.open(res.path) as im:
                    img = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
            except Exception:
                img = None
        if img is None:
            self.canvas.delete("all")
            self.canvas.create_text(280, 260, text="没有可显示的图像", fill=MUTED,
                                    font=(self.base_font, 12))
            return

        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        if fv:
            x0, y0 = fv.cx - fv.radius, fv.cy - fv.radius
            x1, y1 = fv.cx + fv.radius, fv.cy + fv.radius
            draw.ellipse([x0, y0, x1, y1], outline=(29, 78, 216), width=3)
            draw.line([(fv.cx - 10, fv.cy), (fv.cx + 10, fv.cy)], fill=(29, 78, 216), width=2)
            draw.line([(fv.cx, fv.cy - 10), (fv.cx, fv.cy + 10)], fill=(29, 78, 216), width=2)
            # 亮缺陷（划痕/亮斑）的外接框：只画“小而集中”的亮点，避免污渍把整只盖子框住
            pts = fv.debug.get("bright_pts")
            if pts is not None:
                bx, by = pts
                if len(bx) > 10:
                    bw = float(bx.max() - bx.min())
                    bh = float(by.max() - by.min())
                    area_ratio = (bw * bh) / float(img.shape[0] * img.shape[1])
                    if area_ratio < 0.08:
                        pad = 8
                        draw.rectangle([int(bx.min()) - pad, int(by.min()) - pad,
                                        int(bx.max()) + pad, int(by.max()) + pad],
                                       outline=(192, 57, 43), width=3)

        cw = max(self.canvas.winfo_width(), 560)
        ch = max(self.canvas.winfo_height(), 430)
        max_side = min(int(self.cfg.get("ui", {}).get("image_max", 430)), cw - 16, ch - 16)
        pil.thumbnail((max_side, max_side), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)
        q = res.quality
        self.hint.configure(
            text=f"质量：{'正常' if q.ok else q.summary()}｜尺寸 {q.width}×{q.height}"
                 f"｜亮度 {q.mean_lum:.0f}｜清晰度 {q.sharpness:.0f}"
                 f"｜融合 {res.fusion.mode if res.fusion else ''}")

    def _update_risk_bar(self) -> None:
        c = self.risk_canvas
        c.delete("all")
        w = max(c.winfo_width(), 420)
        h = 46
        low = float(self.low_var.get()) if hasattr(self, "low_var") else 0.3607
        high = float(self.high_var.get()) if hasattr(self, "high_var") else 0.8371
        y0, y1 = 10, 26
        c.create_rectangle(0, y0, w * low, y1, fill="#d8f0e5", outline="")
        c.create_rectangle(w * low, y0, w * high, y1, fill="#ffedcc", outline="")
        c.create_rectangle(w * high, y0, w, y1, fill="#f8d7d3", outline="")
        for x in (w * low, w * high):
            c.create_line(x, y0 - 3, x, y1 + 3, fill="#8a96a3", dash=(2, 2))
        # 阈值文字固定在左右两端，避免 low/high 靠得太近时文字叠在一起
        c.create_text(8, y1 + 12, text=f"low {low:.2f}", fill=MUTED,
                      font=(self.base_font, 8), anchor="w")
        c.create_text(w - 8, y1 + 12, text=f"high {high:.2f}", fill=MUTED,
                      font=(self.base_font, 8), anchor="e")
        risk = self.current.risk if self.current else 0.0
        x = max(0, min(w, risk * w))
        c.create_polygon(x - 5, y0 - 2, x + 5, y0 - 2, x, y0 + 5,
                         fill=ACTION_COLOR.get(self.current.action if self.current else "", INK),
                         outline="")
        c.create_line(x, y0 - 6, x, y1 + 4, fill=INK, width=2)
        c.create_text(8, y0 - 8, text=f"risk = {risk:.3f}", anchor="w",
                      fill=INK, font=(self.base_font, 9, "bold"))

    def _update_diag(self, res: InspectionResult) -> None:
        lines = []
        fu = res.fusion
        lines.append(f"结论：{res.action} {ACTION_CN.get(res.action, '')}   "
                     f"融合={fu.mode if fu else '-'}   risk={res.risk:.3f}   "
                     f"类别={CLASS_CN.get(res.pred, res.pred)}（{res.conf:.2f}）")
        if fu:
            lines.append(f"理由：{fu.reason}")
            lines.append(f"依据：{fu.detail}")
        lines.append(f"质量：{res.quality.summary()}  尺寸 {res.quality.width}×{res.quality.height}  "
                     f"亮度 {res.quality.mean_lum:.1f}  清晰度 {res.quality.sharpness:.1f}  "
                     f"偏移 {res.quality.center_offset:.2f}R")
        lines.append("")
        lines.append("特征（数值 / 阈值 / 倍数）与稳健 z：")
        thr = self.cfg.get("thresholds", {})
        if res.features:
            for k, label in (("notch", "凹陷率"), ("dark", "暗斑面积"),
                             ("purity", "直线纯度"), ("peak", "直线带点数")):
                v = getattr(res.features, k)
                t = float(thr.get(k, 0.0))
                z = res.zs.get(k, 0.0)
                lines.append(f"  {label:<8} {v:>10.4f}  阈 {t:>8.4f}   {v / t if t else 0:>5.2f}×   z={z:>6.2f}")
        lines.append("")
        if res.feat:
            lines.append(f"特征模型：{res.feat.pred} {res.feat.conf:.3f}  "
                         + "  ".join(f"{c}:{p:.3f}" for c, p in res.feat.probs.items()))
        if res.deep:
            if res.deep.available:
                lines.append(f"深度模型：{res.deep.pred} {res.deep.conf:.3f}  缺陷分 {res.deep.defect_score:.3f}  "
                             f"{self.inspector.deep.device_text()}  {res.deep.latency_ms:.1f} ms")
                lines.append("  " + "  ".join(f"{c}:{p:.3f}" for c, p in res.deep.probs.items()))
            else:
                lines.append(f"深度模型：不可用（{res.deep.error}）")
        lines.append("")
        lines.append(f"时序：触发 {_fmt_ts(res.ts_trigger)} → 采集 {_fmt_ts(res.ts_capture)} "
                     f"→ 分析 {_fmt_ts(res.ts_analyze)} → 决策 {_fmt_ts(res.ts_decide)} "
                     f"→ 执行 {_fmt_ts(res.ts_act)} → 反馈 {_fmt_ts(res.ts_feedback)}")
        lines.append(f"耗时：分析 {res.analysis_ms:.1f} ms，端到端 {res.elapsed_ms:.1f} ms；"
                     f"跟踪位置 {res.track_pos_mm:.1f} mm；延时补偿 {res.delay_comp_ms:.0f} ms；"
                     f"late={res.late}")
        if res.act:
            lines.append(f"执行：{res.act.command}  ok={res.act.ok}  {res.act.note}  "
                         f"确认 {res.act.latency_ms:.0f} ms")
        if res.error:
            lines.append(f"错误：{res.error}")

        self.diag_text.configure(state="normal")
        self.diag_text.delete("1.0", "end")
        self.diag_text.insert("1.0", "\n".join(lines))
        self.diag_text.configure(state="disabled")

        img = res.features.debug.get("unwrap_img") if res.features else None
        if img is not None:
            pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
            pil.thumbnail((360, 90), Image.LANCZOS)
            self._unwrap_photo = ImageTk.PhotoImage(pil)
            self.unwrap_canvas.delete("all")
            self.unwrap_canvas.create_image(180, 50, image=self._unwrap_photo)
        else:
            self.unwrap_canvas.delete("all")

    # ==================================================================
    # 导出 / 复检 / 关闭
    # ==================================================================
    def _export(self) -> None:
        src = C.out_dir(self.cfg) / self.cfg["paths"].get("record_csv", "records.csv")
        if not src.exists():
            messagebox.showinfo("没有记录", "还没有检测记录。")
            return
        dst = filedialog.asksaveasfilename(
            title="导出记录", defaultextension=".csv",
            initialfile="瓶盖质检记录.csv",
            filetypes=[("CSV", "*.csv")])
        if not dst:
            return
        try:
            shutil.copyfile(src, dst)
            messagebox.showinfo("已导出", f"记录已复制到：\n{dst}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("导出失败", str(exc))

    def _confirm_recheck(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("先选一行", "请在记录表里选中一条 RECHECK 记录。")
            return
        res = self.tree_results.get(sel[0])
        if not res or res.action != "RECHECK":
            messagebox.showinfo("不是复检记录", "只有 RECHECK 记录需要人工确认。")
            return
        ans = messagebox.askyesnocancel(
            "复检确认", f"{res.frame_id}\n系统建议：复检\n\n是 = 人工确认放行\n否 = 人工确认剔除\n取消 = 暂不处理")
        if ans is None:
            return
        final = "PASS" if ans else "REJECT"
        note = simpledialog.askstring("复检备注", "可选：写一句原因（例如“边缘反光，人工看过正常”）",
                                      parent=self)
        if self.inspector.feedback:
            try:
                self.inspector.feedback.submit({
                    "ts": _fmt_ts(time.time()),
                    "frame_id": res.frame_id,
                    "original_action": res.action,
                    "final_action": final,
                    "note": note or "",
                    "operator": os.environ.get("USERNAME", ""),
                })
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror("反馈写入失败", str(exc))
                return
        vals = list(self.tree.item(sel[0], "values"))
        vals[2] = f"{res.action}→{final}"
        self.tree.item(sel[0], values=vals)
        messagebox.showinfo("已记录", "人工复检结果已写入 运行结果/recheck_feedback.csv")

    def _smoke_done(self) -> None:
        print("SMOKE_OK")
        self.destroy()

    def _on_close(self) -> None:
        self.stop_event.set()
        try:
            self.pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self.inspector.close()
        except Exception:
            pass
        self.destroy()


def _prob_text(probs: Dict[str, float]) -> str:
    if not probs:
        return ""
    return "  ".join(f"{c[:3]}:{p:.2f}" for c, p in probs.items())


INTERFACE_TEXT = """H1 原型的四个“插座”（都在代码里有抽象基类/注释）

1) 检测输入：devices/base.py -> FrameSource.iter_frames()
   现在：FileImageSource / FolderSource / ReplaySource
   以后：CameraSource（工业相机）、SensorTriggerSource（对射光电）
   要改的只有“新类实现 iter_frames()”，Inspector 和界面不用动。

2) 执行输出：devices/base.py -> Dispatcher.dispatch(action, ctx)
   现在：SimulatedActuator（模拟挡板/推杆/确认）
   以后：devices/actuators.py 里的 RoboMasterConveyor / PTKServoGate / DamiaoPusher
   三个驱动的方法体已经留好，按 CAN/串口协议填实现即可。
   动作映射：PASS=不动作；RECHECK=挡板到复检通道；REJECT=挡板+推杆到废品通道。

3) 反馈记录：devices/base.py -> Recorder.record(row) / FeedbackSink.submit(row)
   现在：CsvRecorder（运行结果/records.csv）、CsvFeedbackSink（运行结果/recheck_feedback.csv）
   以后：JsonlRecorder / SQLite / MES 接口，只要实现 record()。
   schema 带 schema_version，新增列不会破坏旧记录。

4) 可视化界面：ui/app.py 只订阅 Inspector 的结果
   检测逻辑在 core/，界面可以整体换成 Web/PyQt，不影响算法与记录。
   改界面时保持“result -> 控件”的映射即可；事件通过 queue 回主线程。

再加一类缺陷 / 换模型：
   - 类别：core/types.py 的 CLASSES 与 model.pth 的 class_names 必须一致；
   - 特征：core/features.py 增加测量值，core/anomaly.py/classifier.py 同步；
   - 新模型：core/models 里实现 predict() 返回 ModelOutput，再在 pipeline 里接上。

关键文件：
   core/pipeline.py   一条瓶盖的 6 个阶段（感知/分析/决策/执行/反馈）
   core/fusion.py     M1/M2/M3 三种融合的判定规则
   core/quality.py    质量门（无图/模糊/过曝/偏心）
   devices/base.py    FrameSource / Dispatcher / Recorder / FeedbackSink
   devices/actuators.py  三种电机的接口占位
   config.json        所有阈值、路径、节拍、权重
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="只启动 1.2 秒后退出，用于自检")
    args = ap.parse_args()
    app = App(args.config, smoke=args.smoke)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
