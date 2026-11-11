# -*- coding: utf-8 -*-
"""可视化验证界面（tkinter，零额外依赖）。

界面分三块
----------
左：原图 + 叠加层（圆心十字 / 半径圆 / 亮暗像素），下面一行是“摊平后的偏差图”
右上：四个特征的数值与阈值、异常分条、四类概率条
右下：结论（PASS / RECHECK / REJECT）+ 一句理由 + 各阶段耗时
底部：批量结果表 + 统计（四类准确率、漏检、误剔、复检率）

设计上刻意把“判断”和“界面”分开：
    gui.py 只负责显示，所有数字来自 pipeline.Inspector。
    所以以后要换成网页/别的界面，只要照着 Inspector 的返回结果画即可。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
import time
from collections import Counter
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import CLASSES                              # noqa: E402
from decision import decide                                 # noqa: E402
from devices import CsvRecorder                             # noqa: E402
from pipeline import Inspector, InspectionResult            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# 配色（和文档里的深蓝保持一致）
C_BG = "#FFFFFF"
C_NAVY = "#1F3B63"
C_BLUE = "#2E5C8A"
C_GRAY = "#697077"
C_LIGHT = "#F2F5F9"
C_GREEN = "#38804A"
C_ORANGE = "#D67E1E"
C_RED = "#CC4234"
C_ACTION = {"PASS": C_GREEN, "RECHECK": C_ORANGE, "REJECT": C_RED}
CN = {"normal": "正常", "notch": "缺口", "scratch": "划痕", "stain": "污渍"}

FEATURE_CN = {"notch": "凹陷率（缺口）", "dark": "暗斑面积（污渍）",
              "purity": "直线纯度（划痕）", "peak": "直线带点数"}


class App(tk.Tk):
    """主窗口。"""

    # ---------------------------------------------------------------- 初始化
    def __init__(self, inspector: Inspector, selftest: bool = False):
        super().__init__()
        self.ins = inspector
        self.selftest = selftest
        self.title("瓶盖智能质检 · 串联判断演示（异常分 → 分类器 → 处置）")
        self.geometry("1320x940")
        self.minsize(1240, 900)
        self.configure(bg=C_BG)
        self._photo = None          # 保持引用，否则图片会被回收
        self._cur: Optional[InspectionResult] = None
        self._batch: List[Dict] = []
        self._busy = False

        self._init_fonts()
        self._build_toolbar()
        self._build_status()      # 先占住底部，避免被主体挤掉
        self._build_body()
        self.after(200, self._auto_demo)     # 打开时自动放一张示例图
        auto = os.environ.get("CAPQC_AUTOBATCH", "")
        if auto:                             # 自测用：启动后自动跑一批
            self.after(1200, lambda: self.run_split(auto))

    def _init_fonts(self) -> None:
        fam = "Microsoft YaHei UI"
        self.f_title = (fam, 16, "bold")
        self.f_h1 = (fam, 12, "bold")
        self.f_body = (fam, 10)
        self.f_small = (fam, 9)
        self.f_mono = ("Consolas", 10)
        self.f_action = (fam, 26, "bold")

    # ---------------------------------------------------------------- 工具栏
    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, bg=C_LIGHT, height=52)
        bar.pack(side="top", fill="x")
        tk.Label(bar, text="瓶盖质检演示", font=self.f_title, bg=C_LIGHT, fg=C_NAVY).pack(
            side="left", padx=14, pady=10)

        def btn(text, cmd, color=C_BLUE):
            b = tk.Button(bar, text=text, command=cmd, bg="white", fg=color,
                          font=self.f_body, relief="solid", bd=1, padx=10, pady=3,
                          activebackground=C_LIGHT)
            b.pack(side="left", padx=4, pady=10)
            return b

        btn("打开图片…", self.open_image)
        btn("打开文件夹…", self.open_folder)
        btn("跑验证集 val", lambda: self.run_split("val"))
        btn("跑 test_unlabeled", lambda: self.run_split("test_unlabeled"))
        btn("导出本次结果 CSV", self.export_batch, C_NAVY)

        # 阈值预设：切换后只影响“处置”，不改特征与分类
        tk.Label(bar, text="阈值预设", font=self.f_body, bg=C_LIGHT, fg=C_GRAY).pack(
            side="left", padx=(18, 6))
        pw = self.ins.cfg.get("presets", {})
        # 预设的显示名 → 配置里的键；显示名带上数值，答辩时一眼能看出区别
        self._presets: List[tuple] = []
        for key, text in (("risk_averse", "保守·风险厌恶"),
                          ("balanced", "均衡（最大裕度）"),
                          ("demo_recheck", "演示·会触发复检"),
                          ("course", "课件"),
                          ("starter_kit", "入门模型")):
            if key in pw:
                pv = pw[key]
                self._presets.append((f"{text}（{pv['low']:.2f} / {pv['high']:.2f}）", key))
        if not self._presets:
            self._presets = [("自定义（见当前配置）", None)]
        self.preset_var = tk.StringVar(value=self._presets[0][0])
        combo = ttk.Combobox(bar, textvariable=self.preset_var, width=26, state="readonly",
                             values=[t for t, _ in self._presets])
        combo.pack(side="left", pady=10)
        combo.bind("<<ComboboxSelected>>", self.on_preset)

    # ---------------------------------------------------------------- 主体
    def _build_body(self) -> None:
        body = tk.Frame(self, bg=C_BG)
        body.pack(side="top", fill="both", expand=True, padx=10, pady=(8, 0))

        # 左：图像显示
        left = tk.Frame(body, bg=C_BG)
        left.pack(side="left", fill="y")
        self.canvas = tk.Canvas(left, width=470, height=470, bg="#FAFBFC",
                                highlightthickness=1, highlightbackground="#DDE3EA")
        self.canvas.pack(side="top")
        self.canvas_unwrap = tk.Canvas(left, width=470, height=124, bg="#FAFBFC",
                                       highlightthickness=1, highlightbackground="#DDE3EA")
        self.canvas_unwrap.pack(side="top", pady=(8, 0))
        self.lbl_path = tk.Label(left, text="（还没有打开图片）", font=self.f_small,
                                 bg=C_BG, fg=C_GRAY, wraplength=520, justify="left")
        self.lbl_path.pack(side="top", anchor="w", pady=(6, 0))

        # 右：数据面板
        right = tk.Frame(body, bg=C_BG, width=600)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        right.pack_propagate(False)

        tk.Label(right, text="① 特征测量（每一行都是“数值 / 阈值 / 超标倍数”）",
                 font=self.f_h1, bg=C_BG, fg=C_NAVY).pack(anchor="w")
        self.feat_box = tk.Frame(right, bg=C_BG)
        self.feat_box.pack(anchor="w", fill="x", pady=(6, 10))
        self.feat_labels: Dict[str, List[tk.Label]] = {}
        for k in ("notch", "dark", "purity", "peak"):
            row = tk.Frame(self.feat_box, bg=C_BG)
            row.pack(anchor="w", fill="x", pady=1)
            tk.Label(row, text=FEATURE_CN[k], font=self.f_body, bg=C_BG, fg="#333",
                     width=16, anchor="w").pack(side="left")
            v_lbl = tk.Label(row, text="—", font=("Consolas", 11, "bold"), bg=C_BG,
                             fg="#222", width=10, anchor="e")
            v_lbl.pack(side="left")
            t_lbl = tk.Label(row, text="", font=self.f_small, bg=C_BG, fg=C_GRAY,
                             width=16, anchor="w")
            t_lbl.pack(side="left", padx=(8, 0))
            r_lbl = tk.Label(row, text="", font=self.f_body, bg=C_BG, fg=C_GRAY,
                             anchor="w")
            r_lbl.pack(side="left")
            self.feat_labels[k] = [v_lbl, t_lbl, r_lbl]

        tk.Label(right, text="② 异常分（离正常有多远；不看类别，能抓新缺陷）",
                 font=self.f_h1, bg=C_BG, fg=C_NAVY).pack(anchor="w")
        self.bar_score = tk.Canvas(right, width=560, height=54, bg=C_BG, highlightthickness=0)
        self.bar_score.pack(anchor="w", pady=(4, 2))
        self.lbl_score = tk.Label(right, text="", font=self.f_small, bg=C_BG, fg=C_GRAY)
        self.lbl_score.pack(anchor="w")

        tk.Label(right, text="③ 分类器（四类各自的概率，最高者即预测类别）",
                 font=self.f_h1, bg=C_BG, fg=C_NAVY).pack(anchor="w", pady=(10, 0))
        self.bar_cls = tk.Canvas(right, width=560, height=104, bg=C_BG, highlightthickness=0)
        self.bar_cls.pack(anchor="w", pady=(4, 2))

        tk.Label(right, text="④ 结论", font=self.f_h1, bg=C_BG, fg=C_NAVY).pack(
            anchor="w", pady=(8, 0))
        self.lbl_action = tk.Label(right, text="—", font=self.f_action, bg=C_BG, fg=C_GRAY)
        self.lbl_action.pack(anchor="w")
        self.lbl_reason = tk.Label(right, text="", font=self.f_body, bg=C_BG, fg="#333",
                                   wraplength=560, justify="left")
        self.lbl_reason.pack(anchor="w", pady=(2, 0))
        self.lbl_timing = tk.Label(right, text="", font=self.f_small, bg=C_BG, fg=C_GRAY,
                                   justify="left")
        self.lbl_timing.pack(anchor="w", pady=(4, 0))

        # 临时试算：只改本次显示，不写回配置
        demo = tk.Frame(right, bg=C_LIGHT)
        demo.pack(anchor="w", fill="x", pady=(8, 0))
        tk.Label(demo, text="临时试算（不写回配置）：", font=self.f_small, bg=C_LIGHT,
                 fg=C_GRAY).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.var_low = tk.DoubleVar(value=self.ins.cfg["decision"]["low"])
        self.var_high = tk.DoubleVar(value=self.ins.cfg["decision"]["high"])
        for i, (name, var) in enumerate((("low", self.var_low), ("high", self.var_high))):
            tk.Label(demo, text=f"{name}", font=self.f_small, bg=C_LIGHT, fg=C_GRAY).grid(
                row=0, column=1 + i * 2, padx=(8, 2))
            s = tk.Scale(demo, from_=0.0, to=1.0, resolution=0.01, orient="horizontal",
                         variable=var, length=150, bg=C_LIGHT, highlightthickness=0,
                         showvalue=True, font=self.f_small, command=lambda _=None: self._redo_decision())
            s.grid(row=0, column=2 + i * 2, padx=(0, 8))

    # ---------------------------------------------------------------- 底部
    def _build_status(self) -> None:
        bottom = tk.Frame(self, bg=C_BG, height=196)
        bottom.pack(side="bottom", fill="x", padx=10, pady=(6, 8))
        bottom.pack_propagate(False)
        cols = ("file", "true", "pred", "score", "action", "ok")
        heads = {"file": "文件", "true": "真值", "pred": "预测", "score": "异常分",
                 "action": "处置", "ok": "对/错"}
        self.tree = ttk.Treeview(bottom, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree.heading(c, text=heads[c])
            self.tree.column(c, width=150 if c == "file" else 90, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        self.lbl_stat = tk.Label(bottom, text="", font=self.f_body, bg=C_BG, fg=C_NAVY,
                                 justify="left", anchor="nw", width=40)
        self.lbl_stat.pack(side="right", fill="y", padx=(12, 0))

    # ---------------------------------------------------------------- 交互
    def _auto_demo(self) -> None:
        """启动时自动载入一张划痕样本，窗口一打开就有内容。"""
        if self.selftest:
            return
        p = os.path.join(self.ins.cfg["dataset_root"], "images", "train", "scratch", "img_2_002.png")
        if os.path.exists(p):
            self.show_path(p)

    def open_image(self) -> None:
        p = filedialog.askopenfilename(
            title="选择一张瓶盖图片",
            initialdir=os.path.join(self.ins.cfg["dataset_root"], "images", "train"),
            filetypes=[("PNG/JPEG", "*.png *.jpg *.jpeg"), ("所有文件", "*.*")])
        if p:
            self.show_path(p)

    def open_folder(self) -> None:
        d = filedialog.askdirectory(title="选择一个文件夹（里面的图片会依次检测）",
                                    initialdir=os.path.join(self.ins.cfg["dataset_root"], "images"))
        if not d:
            return
        paths = sorted(os.path.join(d, f) for f in os.listdir(d)
                       if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if not paths:
            messagebox.showinfo("提示", "这个文件夹里没有图片")
            return
        self.run_paths(paths, tag=os.path.basename(d))

    def on_preset(self, _evt=None) -> None:
        """切换阈值预设：只改 decision 段，特征阈值与分类器不动。

        · optimized   —— optimize.py 算出的“最大裕度”阈值（默认，复检率为 0）
        · demo_recheck—— tune.py 原来的阈值，会把缺陷塞进复检区，用来演示 RECHECK 分支
        · course      —— 课件里的 0.5 / 0.9，用于和课堂口径对照
        · starter_kit —— 现成模型的 0.2 / 0.8，用于和“入门模型”方案对照
        """
        pw = self.ins.cfg.get("presets", {})
        key = dict(self._presets).get(self.preset_var.get())
        if key and key in pw:
            self.ins.cfg["decision"].update(pw[key])
            for k in ("low", "high"):                     # 同步滑块，避免显示不一致
                if k in pw[key]:
                    (self.var_low if k == "low" else self.var_high).set(pw[key][k])
        if self._cur:
            self._redo_decision()

    def _redo_decision(self) -> None:
        """用临时滑块的值重算当前这张图的“处置”，并重新画结论区。"""
        if not self._cur or not self._cur.features:
            return
        cfg = dict(self.ins.cfg["decision"])
        cfg["low"] = float(self.var_low.get())
        cfg["high"] = max(float(self.var_high.get()), cfg["low"] + 0.01)
        d = decide(self._cur.score, self._cur.pred, self._cur.proba, cfg, worst=self._cur.worst)
        self._cur.decision = d
        self._paint_decision(self._cur)

    # ---------------------------------------------------------------- 单张
    def show_path(self, path: str) -> None:
        self._set_busy(True, "检测中…")
        try:
            res = self.ins.inspect(path)
        finally:
            self._set_busy(False)
        self._cur = res
        self._paint_all(res)

    def _paint_all(self, res: InspectionResult) -> None:
        self.lbl_path.config(text=res.path)
        self._paint_image(res)
        self._paint_unwrap(res)
        self._paint_features(res)
        self._paint_score(res)
        self._paint_class(res)
        self._paint_decision(res)

    def _paint_image(self, res: InspectionResult) -> None:
        self.canvas.delete("all")
        if not res.features:
            self.canvas.create_text(235, 225, text="（" + res.error + "）", fill=C_RED,
                                    font=self.f_body)
            return
        fv = res.features
        img = fv.debug["image"]
        pil = Image.fromarray(img).convert("RGB")
        scale = 470.0 / max(pil.size)
        tw, th = int(pil.width * scale), int(pil.height * scale)
        pil = pil.resize((tw, th), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        ox, oy = (470 - tw) // 2, (470 - th) // 2
        self.canvas.create_image(ox, oy, anchor="nw", image=self._photo)

        # 叠加：半径圆 + 圆心 + 亮/暗像素
        cx, cy = ox + fv.cx * scale, oy + fv.cy * scale
        R = fv.radius * scale
        self.canvas.create_oval(cx - R, cy - R, cx + R, cy + R, outline=C_NAVY, width=2)
        self.canvas.create_line(cx - 7, cy, cx + 7, cy, fill=C_ORANGE, width=2)
        self.canvas.create_line(cx, cy - 7, cx, cy + 7, fill=C_ORANGE, width=2)
        xs, ys = fv.debug["bright_pts"]
        for X, Y in zip(xs[::2], ys[::2]):
            px, py = ox + X * scale, oy + Y * scale
            self.canvas.create_oval(px - 1, py - 1, px + 1, py + 1, outline="", fill="#00C2D1")

    def _paint_unwrap(self, res: InspectionResult) -> None:
        self.canvas_unwrap.delete("all")
        if not res.features or "unwrap_img" not in res.features.debug:
            return
        img = res.features.debug["unwrap_img"]
        # 横向 720 -> 230，纵向 160 -> 82；上面是圆心、下面是边缘
        pil = Image.fromarray(img).resize((230, 82), Image.BILINEAR)
        self._photo_unwrap = ImageTk.PhotoImage(pil)
        self.canvas_unwrap.create_image(6, 6, anchor="nw", image=self._photo_unwrap)
        self.canvas_unwrap.create_text(246, 8, anchor="nw", font=self.f_small, fill=C_GRAY,
                                       justify="left",
                                       text="摊平后的偏差图（每行已减掉公共环纹）\n"
                                            "红 = 比周围亮（划痕）\n"
                                            "蓝 = 比周围暗（污渍）\n"
                                            "灰 = 正常")

    def _paint_features(self, res: InspectionResult) -> None:
        thr = self.ins.cfg["thresholds"]
        fv = res.features
        if not fv:
            return
        for k, labels in self.feat_labels.items():
            v = getattr(fv, k)
            t = float(thr.get(k, 1.0))
            ratio = (v / t) if t else 0.0
            labels[0].config(text=f"{v:.4f}" if k in ("notch", "dark") else f"{v:.1f}")
            labels[1].config(text=f"阈值 {t:.4f}" if k in ("notch", "dark") else f"阈值 {t:.1f}")
            labels[2].config(text=f"{ratio:.2f} 倍",
                             fg=C_RED if ratio >= 1.0 else C_GREEN)

    def _paint_score(self, res: InspectionResult) -> None:
        c = self.bar_score
        c.delete("all")
        w = 560
        low = self.ins.cfg["decision"]["low"]
        high = self.ins.cfg["decision"]["high"]
        c.create_rectangle(0, 10, w, 26, fill="#EDF0F4", outline="")
        c.create_rectangle(0, 10, w * low, 26, fill="#CDE7D3", outline="")     # 安全区
        c.create_rectangle(w * low, 10, w * high, 26, fill="#FBE3C8", outline="")  # 复检区
        c.create_rectangle(w * high, 10, w, 26, fill="#F5CFC9", outline="")    # 剔除区
        x = w * min(max(res.score, 0.0), 1.0)
        c.create_rectangle(x - 3, 2, x + 3, 34, fill=C_NAVY, outline="")
        # 只标两条边界值（low / high 可能很接近，写文字会挤在一起）
        for val, txt in ((low, f"low {low:.2f}"), (high, f"high {high:.2f}")):
            xv = w * val
            c.create_line(xv, 34, xv, 40, fill=C_GRAY)
            c.create_text(min(max(xv, 26), w - 26), 46, text=txt, font=self.f_small, fill=C_GRAY)
        self.lbl_score.config(
            text=f"异常分 {res.score:.3f}   最异常的特征：{FEATURE_CN.get(res.worst, res.worst)}   "
                 f"绿=PASS / 橙=RECHECK / 红=REJECT")

    def _paint_class(self, res: InspectionResult) -> None:
        c = self.bar_cls
        c.delete("all")
        for i, cls in enumerate(CLASSES):
            y = 6 + i * 25
            p = res.proba.get(cls, 0.0)
            c.create_text(0, y + 8, anchor="w", text=CN.get(cls, cls), font=self.f_small, fill="#333")
            c.create_rectangle(60, y, 470, y + 16, fill="#EDF0F4", outline="")
            c.create_rectangle(60, y, 60 + 410 * p, y + 16,
                               fill=C_BLUE if cls == res.pred else "#B9C6D6", outline="")
            c.create_text(480, y + 8, anchor="w", text=f"{p:.2f}", font=self.f_small,
                          fill=C_NAVY if cls == res.pred else C_GRAY)

    def _paint_decision(self, res: InspectionResult) -> None:
        d = res.decision
        color = C_ACTION.get(d.action, C_GRAY)
        self.lbl_action.config(text=d.action, fg=color)
        self.lbl_reason.config(text=("【未知缺陷】" if d.is_unknown else "") + d.reason)
        act = res.act
        self.lbl_timing.config(
            text=(f"预测 {CN.get(res.pred, res.pred)}（置信 {res.conf:.2f}）   "
                  f"执行：{act.command}（{'确认成功' if act.ok else '无确认！'}，"
                  f"{act.latency_ms:.1f} ms）\n"
                  f"整段耗时 {res.elapsed_ms:.0f} ms（感知→分析→决策→执行→反馈）  "
                  f"{d.detail}"))

    # ---------------------------------------------------------------- 批量
    def run_split(self, split: str) -> None:
        cfg = self.ins.cfg
        csv_path = os.path.join(cfg["dataset_root"], cfg["labels_csv"])
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                rows = [r for r in csv.DictReader(f) if r["split"] == split]
        except FileNotFoundError:
            messagebox.showerror("出错", f"找不到标签文件：{csv_path}")
            return
        if not rows:
            # test_unlabeled 没有标签，直接扫文件夹
            folder = os.path.join(cfg["dataset_root"], "images", split)
            paths = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith(".png"))
            self.run_paths(paths, tag=split, truth=None)
            return
        paths, truth = [], {}
        for r in rows:
            p = os.path.join(cfg["dataset_root"], r["relative_path"].replace("/", os.sep))
            paths.append(p)
            truth[os.path.basename(p)] = r["class_name"]
        self.run_paths(paths, tag=split, truth=truth)

    def run_paths(self, paths: List[str], tag: str = "", truth: Optional[Dict[str, str]] = None) -> None:
        if self._busy:
            messagebox.showinfo("稍等", "上一步还在跑，请稍候")
            return
        self._set_busy(True, f"批量检测 {tag}：0/{len(paths)}")
        self.tree.delete(*self.tree.get_children())
        self._batch = []
        th = threading.Thread(target=self._batch_worker, args=(paths, tag, truth), daemon=True)
        th.start()

    def _batch_worker(self, paths: List[str], tag: str, truth: Optional[Dict[str, str]]) -> None:
        rows = []
        for i, p in enumerate(paths, 1):
            res = self.ins.inspect(p)
            t = (truth or {}).get(os.path.basename(p), "")
            rows.append({"res": res, "truth": t})
            self.after(0, self._progress, i, len(paths), res, t, tag)
        self.after(0, self._batch_done, rows, tag, truth)

    def _progress(self, i: int, n: int, res: InspectionResult, truth: str, tag: str) -> None:
        ok = "" if not truth else ("✓" if res.pred == truth else "✗")
        self.tree.insert("", "end", values=(res.frame_id, CN.get(truth, truth), CN.get(res.pred, res.pred),
                                            f"{res.score:.3f}", res.action, ok))
        self.tree.yview_moveto(1.0)
        self.lbl_path.config(text=f"[{tag}] {i}/{n}  {res.frame_id}")

    def _batch_done(self, rows: List[Dict], tag: str, truth: Optional[Dict[str, str]]) -> None:
        self._batch = rows
        self._set_busy(False)
        if not rows:
            return
        n = len(rows)
        if truth:
            correct = sum(1 for r in rows if r["res"].pred == r["truth"])
            defect = [r for r in rows if r["truth"] != "normal"]
            defect_pass = [r for r in defect if r["res"].action == "PASS"]
            normal = [r for r in rows if r["truth"] == "normal"]
            normal_reject = [r for r in normal if r["res"].action == "REJECT"]
            recheck = [r for r in rows if r["res"].action == "RECHECK"]
            self.lbl_stat.config(
                text=(f"【{tag}】共 {n} 张\n"
                      f"四类准确率 {correct / n:.2%}\n"
                      f"缺陷处置率 {1 - len(defect_pass) / max(len(defect), 1):.2%}\n"
                      f"漏检 {len(defect_pass)} 张\n"
                      f"误剔 {len(normal_reject)} 张\n"
                      f"复检 {len(recheck)} 张（{len(recheck) / n:.1%}）\n"
                      f"平均 {sum(r['res'].elapsed_ms for r in rows) / n:.0f} ms/张"))
        else:
            cnt = Counter(r["res"].pred for r in rows)
            act = Counter(r["res"].action for r in rows)
            self.lbl_stat.config(
                text=(f"【{tag}】共 {n} 张（无标签）\n"
                      + "\n".join(f"{CN.get(k, k)} {v}" for k, v in cnt.items())
                      + "\n" + "\n".join(f"{k} {v}" for k, v in act.items())))
        self.lbl_path.config(text=f"[{tag}] 完成。点表格看汇总；可导出 CSV。")

    def export_batch(self) -> None:
        if not self._batch:
            messagebox.showinfo("提示", "还没有批量结果")
            return
        out = os.path.join(HERE, self.ins.cfg["paths"]["out_dir"],
                           f"gui_batch_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        rows = []
        for r in self._batch:
            d = r["res"].to_record()
            d["truth"] = r["truth"]
            rows.append(d)
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        messagebox.showinfo("已导出", out)

    # ---------------------------------------------------------------- 状态
    def _set_busy(self, busy: bool, text: str = "") -> None:
        self._busy = busy
        self.config(cursor="watch" if busy else "")
        if text:
            self.lbl_path.config(text=text)
        self.update_idletasks()


def main() -> None:
    ap = argparse.ArgumentParser(description="瓶盖质检可视化界面")
    ap.add_argument("--selftest", action="store_true", help="只构建界面、跑一张图就退出（自检用）")
    args = ap.parse_args()

    recorder = CsvRecorder(os.path.join(HERE, "out", "records.csv"))
    ins = Inspector.from_config(recorder=recorder)
    app = App(ins, selftest=args.selftest)
    if args.selftest:
        # 自检：在窗口里跑一张图、打印关键字段，然后退出
        p = os.path.join(ins.cfg["dataset_root"], "images", "train", "stain", "img_3_001.png")
        app.show_path(p)
        app.update()
        res = app._cur
        print("selftest:", res.frame_id, res.pred, f"{res.score:.3f}", res.action,
              f"{res.elapsed_ms:.0f} ms")
        app.destroy()
        return
    app.mainloop()


if __name__ == "__main__":
    main()
