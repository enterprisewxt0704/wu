# -*- coding: utf-8 -*-
"""串联流水线：把五个模块串成一个闭环，并记录每个阶段的时刻。

    感知 SENSE -> 分析 ANALYZE -> 决策 DECIDE -> 执行 ACT -> 反馈 FEEDBACK

每个阶段都是一个“可替换的零件”：
    感知 = features.measure_image + devices.FrameSource
    分析 = anomaly.Baseline + classifier.Model
    决策 = decision.decide
    执行 = devices.Dispatcher
    反馈 = devices.Recorder + ActResult 的确认检查

对外只暴露两个入口：
    Inspector.from_config()   读 model/*.json 与 config.json
    Inspector.inspect(path)   跑一张图，返回 InspectionResult（含全过程时刻）
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from anomaly import Baseline
from classifier import CLASSES, Model
from decision import Decision, decide
from devices import (ActResult, CsvRecorder, Dispatcher, Frame, FrameSource,
                     LogOnlyDispatcher, MemoryRecorder, Recorder, SimulatedActuator)
from features import FeatureVector, measure_image

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------
# 结果对象
# ----------------------------------------------------------------------------
@dataclass
class InspectionResult:
    frame_id: str = ""
    path: str = ""
    features: Optional[FeatureVector] = None
    score: float = 0.0
    zs: Dict[str, float] = field(default_factory=dict)
    worst: str = ""
    pred: str = ""
    proba: Dict[str, float] = field(default_factory=dict)
    decision: Optional[Decision] = None
    act: Optional[ActResult] = None
    # 五个阶段的时刻（秒，time.time()）与整段耗时
    t_sense: float = 0.0
    t_capture: float = 0.0
    t_analyze: float = 0.0
    t_decide: float = 0.0
    t_act: float = 0.0
    t_feedback: float = 0.0
    elapsed_ms: float = 0.0
    error: str = ""

    @property
    def action(self) -> str:
        return self.decision.action if self.decision else "ERROR"

    @property
    def conf(self) -> float:
        return max(self.proba.values()) if self.proba else 0.0

    def to_record(self) -> Dict:
        """写进 CSV/JSONL 的一行：5 个必备字段 + 特征 + 执行/耗时。"""
        f = self.features
        return {
            "ts": f"{self.t_sense:.3f}",
            "frame_id": self.frame_id,
            "image_path": self.path,
            "notch": round(f.notch, 6) if f else "",
            "dark": round(f.dark, 6) if f else "",
            "purity": round(f.purity, 4) if f else "",
            "peak": round(f.peak, 1) if f else "",
            "anomaly_score": round(self.score, 4),
            "worst_feature": self.worst,
            "pred_class": self.pred,
            "pred_conf": round(self.conf, 4),
            "class_proba": json.dumps({k: round(v, 4) for k, v in self.proba.items()},
                                      ensure_ascii=False),
            "action": self.action,
            "reason": self.decision.reason if self.decision else self.error,
            "is_unknown": int(self.decision.is_unknown) if self.decision else 0,
            "act_ok": int(self.act.ok) if self.act else 0,
            "act_command": self.act.command if self.act else "",
            "act_latency_ms": round(self.act.latency_ms, 2) if self.act else 0.0,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


# ----------------------------------------------------------------------------
# 流水线
# ----------------------------------------------------------------------------
class Inspector:
    """一整套检测系统（不含界面）。界面、命令行、批量评估都用它。"""

    def __init__(self, config: Dict, model: Model, baseline: Baseline,
                 dispatcher: Optional[Dispatcher] = None,
                 recorder: Optional[Recorder] = None,
                 keep_debug: bool = True):
        self.cfg = config
        self.model = model
        self.baseline = baseline
        self.dispatcher = dispatcher or LogOnlyDispatcher()
        self.recorder = recorder
        self.keep_debug = keep_debug

    # ---------------- 构造 ----------------
    @classmethod
    def from_config(cls, config_path: Optional[str] = None, *,
                    dispatcher: Optional[Dispatcher] = None,
                    recorder: Optional[Recorder] = None,
                    keep_debug: bool = True) -> "Inspector":
        cfg_path = config_path or os.path.join(HERE, "config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_dir = os.path.join(HERE, cfg["paths"]["model_dir"])
        model = Model.load(os.path.join(model_dir, "classifier.json"))
        baseline = Baseline.load(os.path.join(model_dir, "baseline.json"))

        if dispatcher is None:
            rt = cfg.get("runtime", {})
            if rt.get("simulate_actuator", True):
                dispatcher = SimulatedActuator(delay_ms=rt.get("actuator_delay_ms", 60),
                                               fail_rate=rt.get("actuator_fail_rate", 0.0))
            else:
                dispatcher = LogOnlyDispatcher()
        if recorder is None:
            rt = cfg.get("runtime", {})
            rec_path = os.path.join(HERE, cfg["paths"]["out_dir"], rt.get("record_csv", "records.csv"))
            recorder = CsvRecorder(rec_path)
        return cls(cfg, model, baseline, dispatcher, recorder, keep_debug)

    def reload_config(self) -> None:
        """界面切换阈值预设后调用：只重读 config.json，不重载模型。"""
        with open(os.path.join(HERE, "config.json"), "r", encoding="utf-8") as f:
            self.cfg = json.load(f)

    # ---------------- 单帧 ----------------
    def inspect(self, path: str, frame_id: Optional[str] = None,
                ts_sense: Optional[float] = None) -> InspectionResult:
        res = InspectionResult(frame_id=frame_id or os.path.basename(path), path=path)
        t0 = res.t_sense = ts_sense or time.time()

        # --- 感知：测量（圆心/半径/摊平/三个特征）
        try:
            fv = measure_image(path, self.cfg["features"], keep_debug=self.keep_debug)
        except Exception as exc:                       # 感知异常：没有图像
            res.error = f"感知失败：{exc}"
            self._finish(res, t0, error=True)
            return res
        res.features = fv
        res.t_capture = time.time()

        # --- 分析：异常分 + 分类
        feat = fv.as_features()
        score, zs = self.baseline.score(feat)
        res.score, res.zs = score, zs
        res.worst = self.baseline.worst_feature(feat)
        proba = self.model.predict_proba([feat])[0]
        res.proba = {c: float(p) for c, p in zip(CLASSES, proba)}
        res.pred = max(res.proba, key=res.proba.get)
        res.t_analyze = time.time()

        # --- 决策
        res.decision = decide(score, res.pred, res.proba, self.cfg["decision"], worst=res.worst)
        res.t_decide = time.time()

        # --- 执行
        res.act = self.dispatcher.dispatch(res.decision.action,
                                          {"frame_id": res.frame_id, "pred": res.pred})
        res.t_act = res.act.ts_act

        # --- 反馈（确认 + 记录）
        self._finish(res, t0)
        return res

    def _finish(self, res: InspectionResult, t0: float, error: bool = False) -> None:
        """反馈阶段：确认执行结果、算耗时、落盘。"""
        if error:
            res.decision = Decision("RECHECK", "感知异常（没有可用图像），需人工接管", True, res.error)
            res.act = ActResult(False, "stop_line", time.time(), 0.0, "报警：请求人工接管")
            res.t_act = res.act.ts_act
        res.t_feedback = time.time()
        res.elapsed_ms = (res.t_feedback - t0) * 1000.0
        if self.recorder is not None:
            try:
                self.recorder.record(res.to_record())
            except Exception:
                pass                      # 记录失败不能影响产线（真实系统要单独报警）

    # ---------------- 批量 ----------------
    def inspect_many(self, paths: List[str], progress=None) -> List[InspectionResult]:
        out = []
        for i, p in enumerate(paths, 1):
            out.append(self.inspect(p))
            if progress:
                progress(i, len(paths), out[-1])
        return out

    def inspect_source(self, source: FrameSource, progress=None) -> List[InspectionResult]:
        """以后接相机/光电开关时，只要传一个 FrameSource 进来。"""
        frames = list(source.iter_frames())
        out = []
        for i, fr in enumerate(frames, 1):
            res = self.inspect(fr.path, fr.frame_id, fr.ts_sense) if fr.path else None
            if res is not None:
                out.append(res)
            if progress:
                progress(i, len(frames), res)
        return out
