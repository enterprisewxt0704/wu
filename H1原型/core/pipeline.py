# -*- coding: utf-8 -*-
"""一条完整的检测流水线：感知 -> 分析 -> 决策 -> 执行 -> 反馈。

对外只暴露 Inspector：
    Inspector.from_config()             # 读 config/model，构造一套系统
    inspector.inspect(path_or_ndarray)  # 跑一只瓶盖，返回 InspectionResult
    inspector.inspect_many(paths)       # 批量（离线评估）

界面、命令行、批量评估、连续仿真都调用同一个 Inspector，
保证“界面上看到的结论”和“报告里跑出来的数字”来自同一份代码。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from . import config as C
from .anomaly import Baseline
from .classifier import Model
from .deep_model import DeepModel
from .features import FeatureVector, measure_array
from .fusion import fuse
from .quality import assess, check_center
from .types import (CLASSES, DEFECT_CLASSES, FEATURE_CN, FusionResult,
                    ModelOutput, QualityReport)

from devices.base import (ActResult, Dispatcher, FeedbackSink, Recorder,
                          RECORD_FIELDS, SCHEMA_VERSION, CsvFeedbackSink,
                          CsvRecorder, NullDispatcher)
from devices.actuators import build_dispatcher


@dataclass
class InspectionResult:
    frame_id: str = ""
    path: str = ""
    label: str = ""

    quality: QualityReport = field(default_factory=QualityReport)
    features: Optional[FeatureVector] = None
    anomaly_score: float = 0.0
    zs: Dict[str, float] = field(default_factory=dict)
    worst: str = ""

    feat: Optional[ModelOutput] = None
    deep: Optional[ModelOutput] = None
    fusion: Optional[FusionResult] = None
    act: Optional[ActResult] = None

    # 各阶段时刻（time.time()，秒）
    ts_trigger: float = 0.0
    ts_capture: float = 0.0
    ts_analyze: float = 0.0
    ts_decide: float = 0.0
    ts_act: float = 0.0
    ts_ack: float = 0.0
    ts_feedback: float = 0.0

    elapsed_ms: float = 0.0
    analysis_ms: float = 0.0
    track_pos_mm: float = 0.0
    delay_comp_ms: float = 0.0
    queue_depth: int = 0
    late: bool = False

    error: str = ""
    record_error: str = ""

    # ---------------- 便捷属性 ----------------
    @property
    def action(self) -> str:
        return self.fusion.action if self.fusion else "RECHECK"

    @property
    def pred(self) -> str:
        return self.fusion.pred if self.fusion else ""

    @property
    def risk(self) -> float:
        return self.fusion.risk if self.fusion else 0.0

    @property
    def conf(self) -> float:
        return self.fusion.conf if self.fusion else 0.0

    @property
    def proba(self) -> Dict[str, float]:
        return self.fusion.proba if self.fusion else {}

    @property
    def backend(self) -> str:
        return "feature+deep" if (self.deep and self.deep.available) else "feature"

    def to_record(self) -> Dict:
        """转成 CSV 一行。字段与 devices.base.RECORD_FIELDS 一致。"""
        q = self.quality
        f = self.features
        feat = self.feat
        deep = self.deep
        fu = self.fusion
        act = self.act

        def pj(d: Optional[Dict[str, float]]) -> str:
            return json.dumps({k: round(float(v), 6) for k, v in (d or {}).items()},
                              ensure_ascii=False)

        return {
            "schema_version": SCHEMA_VERSION,
            "ts": _fmt_ts(self.ts_trigger),
            "frame_id": self.frame_id,
            "image_path": self.path,
            "label": self.label,
            "quality_ok": int(q.ok),
            "quality_reason": q.summary(),
            "mean_lum": round(q.mean_lum, 2),
            "sharpness": round(q.sharpness, 2),
            "center_offset": round(q.center_offset, 4),
            "notch": round(f.notch, 6) if f else "",
            "dark": round(f.dark, 6) if f else "",
            "purity": round(f.purity, 4) if f else "",
            "peak": round(f.peak, 1) if f else "",
            "anomaly_score": round(self.anomaly_score, 4),
            "worst_feature": self.worst,
            "feat_class": feat.pred if feat else "",
            "feat_conf": round(feat.conf, 4) if feat else "",
            "feat_proba": pj(feat.probs if feat else {}),
            "deep_class": deep.pred if deep else "",
            "deep_conf": round(deep.conf, 4) if deep else "",
            "deep_defect": round(deep.defect_score, 4) if deep else "",
            "deep_proba": pj(deep.probs if deep else {}),
            "deep_available": int(bool(deep and deep.available)),
            "deep_latency_ms": round(deep.latency_ms, 2) if deep else "",
            "fusion_mode": fu.mode if fu else "",
            "risk": round(fu.risk, 4) if fu else "",
            "pred_class": fu.pred if fu else "",
            "pred_conf": round(fu.conf, 4) if fu else "",
            "class_proba": pj(fu.proba if fu else {}),
            "action": fu.action if fu else "RECHECK",
            "reason": fu.reason if fu else self.error,
            "is_unknown": int(fu.is_unknown) if fu else 1,
            "rule": fu.rule if fu else "",
            "act_ok": int(act.ok) if act else 0,
            "act_command": act.command if act else "",
            "act_latency_ms": round(act.latency_ms, 2) if act else "",
            "act_note": act.note if act else "",
            "track_pos_mm": round(self.track_pos_mm, 2),
            "delay_comp_ms": round(self.delay_comp_ms, 2),
            "queue_depth": self.queue_depth,
            "late": int(self.late),
            "ts_trigger": round(self.ts_trigger, 6),
            "ts_capture": round(self.ts_capture, 6),
            "ts_analyze": round(self.ts_analyze, 6),
            "ts_decide": round(self.ts_decide, 6),
            "ts_act": round(self.ts_act, 6),
            "ts_ack": round(self.ts_ack, 6),
            "ts_feedback": round(self.ts_feedback, 6),
            "analysis_ms": round(self.analysis_ms, 2),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "backend": self.backend,
        }


def _fmt_ts(t: float) -> str:
    if not t:
        return ""
    lt = time.localtime(t)
    return time.strftime("%Y-%m-%d %H:%M:%S", lt) + f".{int((t % 1) * 1000):03d}"


class Inspector:
    def __init__(self, cfg: Dict, root: Path,
                 dispatcher: Optional[Dispatcher] = None,
                 recorder: Optional[Recorder] = None,
                 feedback: Optional[FeedbackSink] = None,
                 keep_debug: bool = True):
        self.cfg = cfg
        self.root = Path(root)
        self.keep_debug = keep_debug

        self.feat_model = Model.load(str(C.classifier_path(cfg)))
        self.baseline = Baseline.load(str(C.baseline_path(cfg)))
        self.deep = DeepModel(cfg, self.root)
        self.labels = C.load_labels(cfg)

        self.dispatcher = dispatcher or build_dispatcher(cfg, "sim")
        self.recorder = recorder
        self.feedback = feedback
        self.fusion_mode = cfg.get("fusion", {}).get("default_mode", "M3")
        self.last_error = ""

    # ------------------------------------------------------------------
    # 构造 / 配置
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path: Optional[str] = None, *,
                    dispatcher_mode: str = "sim",
                    record: bool = True,
                    keep_debug: bool = True) -> "Inspector":
        cfg = C.load_config(config_path)
        dispatcher = build_dispatcher(cfg, dispatcher_mode)
        recorder = None
        feedback = None
        if record:
            out = C.out_dir(cfg)
            recorder = CsvRecorder(str(out / cfg["paths"].get("record_csv", "records.csv")))
            feedback = CsvFeedbackSink(str(out / cfg["paths"].get("feedback_csv", "recheck_feedback.csv")))
        return cls(cfg, C.PROJECT_ROOT, dispatcher, recorder, feedback, keep_debug)

    def set_fusion_mode(self, mode: str) -> None:
        self.fusion_mode = mode

    def update_decision(self, low: Optional[float] = None,
                        high: Optional[float] = None,
                        conf_min: Optional[float] = None) -> None:
        d = self.cfg.setdefault("decision", {})
        if low is not None:
            d["low"] = float(low)
        if high is not None:
            d["high"] = float(high)
        if conf_min is not None:
            d["conf_min"] = float(conf_min)

    def save_config(self) -> None:
        C.save_config(self.cfg)

    def close(self) -> None:
        if self.recorder:
            self.recorder.close()
        if self.feedback:
            self.feedback.close()

    # ------------------------------------------------------------------
    # 单只瓶盖
    # ------------------------------------------------------------------
    def inspect(self, image_or_path: Union[str, Path, np.ndarray],
                frame_id: Optional[str] = None,
                ts_trigger: Optional[float] = None,
                label: Optional[str] = None,
                dispatch: bool = True,
                queue_depth: int = 0) -> InspectionResult:
        res = InspectionResult(frame_id=frame_id or self._frame_id(image_or_path),
                               path=str(image_or_path) if isinstance(image_or_path, (str, Path)) else "")
        res.label = label if label is not None else self.labels.get(res.frame_id, "")
        res.ts_trigger = ts_trigger or time.time()
        res.queue_depth = queue_depth
        t0 = res.ts_trigger

        # ---------------- 1. 感知：读图 + 质量门 ----------------
        image: Optional[np.ndarray] = None
        try:
            if isinstance(image_or_path, np.ndarray):
                image = image_or_path
            else:
                image = _read_rgb(image_or_path)
            res.ts_capture = time.time()
            res.quality = assess(image, self.cfg)
        except Exception as exc:  # noqa: BLE001
            res.error = f"感知失败：{type(exc).__name__}: {exc}"
            res.quality = QualityReport(ok=False, reasons=[res.error])
            self._finish_early(res, t0, dispatch, "感知异常（没有可用图像），需人工接管")
            return res

        # ---------------- 2. 特征测量 ----------------
        try:
            res.features = measure_array(image, self.cfg["features"],
                                         keep_debug=self.keep_debug)
            check_center(res.quality, res.features, self.cfg)
        except Exception as exc:  # noqa: BLE001
            res.error = f"特征测量失败：{type(exc).__name__}: {exc}"
            res.quality.ok = False
            res.quality.reasons.append(res.error)
            self._finish_early(res, t0, dispatch, "特征测量异常，需人工接管")
            return res

        # ---------------- 3. 分析：特征模型 + 深度模型 ----------------
        feat = res.features.as_features()
        score, zs = self.baseline.score(feat)
        res.anomaly_score = score
        res.zs = zs
        res.worst = self.baseline.worst_feature(feat)

        feat_proba = self.feat_model.predict_proba([feat])[0]
        fprobs = {c: float(p) for c, p in zip(CLASSES, feat_proba)}
        fpred = max(fprobs, key=fprobs.get)
        res.feat = ModelOutput(
            name="feature", available=True, probs=fprobs, pred=fpred,
            conf=float(fprobs[fpred]),
            defect_score=float(sum(fprobs[c] for c in DEFECT_CLASSES)),
        )

        if self.cfg.get("deep", {}).get("enabled", True):
            try:
                res.deep = self.deep.predict(image)
            except Exception as exc:  # noqa: BLE001
                res.deep = ModelOutput.unavailable("deep", f"{type(exc).__name__}: {exc}")
        else:
            res.deep = ModelOutput.unavailable("deep", "config 中 deep.enabled=false")

        res.ts_analyze = time.time()
        res.analysis_ms = (res.ts_analyze - res.ts_capture) * 1000.0

        # ---------------- 4. 决策 ----------------
        try:
            res.fusion = fuse(self.fusion_mode, score, res.feat, res.deep,
                              self.cfg, res.quality, res.worst)
        except Exception as exc:  # noqa: BLE001
            res.error = f"决策异常：{type(exc).__name__}: {exc}"
            res.fusion = FusionResult(
                mode="ERROR", risk=1.0, pred="", conf=0.0, proba={},
                action="RECHECK", reason=f"决策模块异常，保守转复检：{res.error}",
                is_unknown=True, rule="decision_error",
            )
        res.ts_decide = time.time()

        # ---------------- 5. 执行 ----------------
        line = self.cfg.get("line", {})
        speed = max(float(line.get("conveyor_speed_mm_s", 120.0)), 1e-6)
        total_mm = float(line.get("sensor_to_camera_mm", 40.0)) + \
            float(line.get("camera_to_actuator_mm", 180.0))
        res.delay_comp_ms = total_mm / speed * 1000.0
        res.track_pos_mm = speed * (res.ts_decide - res.ts_trigger) * 1000.0 / 1000.0
        res.late = (res.ts_decide - res.ts_trigger) * 1000.0 > res.delay_comp_ms

        if dispatch:
            ctx = {
                "frame_id": res.frame_id,
                "action": res.action,
                "pred": res.pred,
                "track_pos_mm": res.track_pos_mm,
                "delay_comp_ms": res.delay_comp_ms,
                "late": res.late,
                "queue_depth": res.queue_depth,
                "quality_ok": res.quality.ok,
                "stop_line": not res.quality.ok,
            }
            try:
                res.act = self.dispatcher.dispatch(res.action, ctx)
            except Exception as exc:  # noqa: BLE001
                res.act = ActResult(False, "dispatch_error", time.time(), 0.0,
                                    f"{type(exc).__name__}: {exc}")
        res.ts_act = res.act.ts_act if res.act else time.time()
        res.ts_ack = res.act.ts_ack if res.act else 0.0

        # ---------------- 6. 反馈：落盘 ----------------
        self._finish(res, t0)
        return res

    # ------------------------------------------------------------------
    # 批量
    # ------------------------------------------------------------------
    def inspect_many(self, paths: Sequence[str], progress=None,
                     dispatch: bool = False) -> List[InspectionResult]:
        out: List[InspectionResult] = []
        for i, p in enumerate(paths, 1):
            res = self.inspect(p, dispatch=dispatch)
            out.append(res)
            if progress:
                progress(i, len(paths), res)
        return out

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _frame_id(self, image_or_path) -> str:
        if isinstance(image_or_path, (str, Path)):
            return os.path.basename(str(image_or_path))
        return f"frame_{int(time.time() * 1000)}"

    def _finish_early(self, res: InspectionResult, t0: float,
                      dispatch: bool, reason: str) -> None:
        """读图/特征失败：也要走完执行与记录，不能静默丢掉。"""
        res.fusion = FusionResult(
            mode="ERROR", risk=1.0, pred="", conf=0.0, proba={},
            action="RECHECK", reason=reason, is_unknown=True,
            detail=res.error, rule="sense_error",
        )
        res.ts_analyze = res.ts_decide = time.time()
        if dispatch:
            try:
                res.act = self.dispatcher.dispatch("RECHECK", {
                    "frame_id": res.frame_id, "stop_line": True,
                    "quality_ok": False, "reason": reason,
                })
            except Exception as exc:  # noqa: BLE001
                res.act = ActResult(False, "dispatch_error", time.time(), 0.0, str(exc))
        res.ts_act = res.act.ts_act if res.act else time.time()
        res.ts_ack = res.act.ts_ack if res.act else 0.0
        self._finish(res, t0)

    def _finish(self, res: InspectionResult, t0: float) -> None:
        res.ts_feedback = time.time()
        res.elapsed_ms = (res.ts_feedback - t0) * 1000.0
        if self.recorder is not None:
            try:
                self.recorder.record(res.to_record())
            except Exception as exc:  # noqa: BLE001
                res.record_error = f"{type(exc).__name__}: {exc}"


def _read_rgb(path: Union[str, Path]) -> np.ndarray:
    with Image.open(str(path)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def load_image(path: Union[str, Path]) -> np.ndarray:
    return _read_rgb(path)
