# -*- coding: utf-8 -*-
"""设备层基类：把“检测输入 / 执行输出 / 反馈记录”做成三个可替换插座。

现在没有真实硬件，所以提供：
    FileImageSource   从图片/文件夹输入（离线演示）
    NullDispatcher    只出结论、不动作（批量评估）
    SimulatedActuator 模拟气缸/挡板，带延时和确认（连续仿真）
    CsvRecorder       一行一条记录，立刻 flush（断电不丢）

以后有硬件时，只要新写一个类实现同样的方法，pipeline 和界面都不用改：
    CameraSource / SensorTriggerSource   -> 实现 FrameSource.iter_frames()
    PlcActuator / RoboMaster / PTK / 达妙 -> 实现 Dispatcher.dispatch()
    SqliteRecorder / JsonlRecorder       -> 实现 Recorder.record()

记录 schema 带 schema_version：以后加字段（比如曝光、编码器计数）时，
旧 CSV 仍能读，新 CSV 多一列，不会把历史数据搞乱。
"""

from __future__ import annotations

import abc
import csv
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional

import numpy as np

SCHEMA_VERSION = "h1.2"

# 记录 CSV 的列顺序。界面表格只显示其中一部分，完整行留给报告/追溯。
RECORD_FIELDS = [
    "schema_version", "ts", "frame_id", "image_path", "label",
    "quality_ok", "quality_reason", "mean_lum", "sharpness", "center_offset",
    "notch", "dark", "purity", "peak",
    "anomaly_score", "worst_feature",
    "feat_class", "feat_conf", "feat_proba",
    "deep_class", "deep_conf", "deep_defect", "deep_proba", "deep_available", "deep_latency_ms",
    "fusion_mode", "risk", "pred_class", "pred_conf", "class_proba",
    "action", "reason", "is_unknown", "rule",
    "act_ok", "act_command", "act_latency_ms", "act_note",
    "track_pos_mm", "delay_comp_ms", "queue_depth", "late",
    "ts_trigger", "ts_capture", "ts_analyze", "ts_decide", "ts_act", "ts_ack", "ts_feedback",
    "analysis_ms", "elapsed_ms", "backend",
]


# ----------------------------------------------------------------------------
# 1) 输入
# ----------------------------------------------------------------------------
@dataclass
class Frame:
    frame_id: str
    path: Optional[str] = None
    image: Optional[np.ndarray] = None       # RGB uint8 (H,W,3)
    ts_sense: float = 0.0                    # 光电开关上升沿时刻
    meta: Dict[str, str] = field(default_factory=dict)


class FrameSource(abc.ABC):
    """输入设备接口：实现 iter_frames() 就能接任何图像来源。"""

    @abc.abstractmethod
    def iter_frames(self) -> Iterator[Frame]:
        ...


class FileImageSource(FrameSource):
    """单张/多张图片。界面“打开图片/文件夹”和批量评估都走这里。"""

    def __init__(self, paths: Iterable[str], repeat: int = 1):
        self.paths = list(paths)
        self.repeat = max(1, int(repeat))

    def iter_frames(self) -> Iterator[Frame]:
        for _ in range(self.repeat):
            for p in self.paths:
                yield Frame(frame_id=os.path.basename(p), path=p, ts_sense=time.time())


# ----------------------------------------------------------------------------
# 2) 输出
# ----------------------------------------------------------------------------
@dataclass
class ActResult:
    ok: bool
    command: str
    ts_act: float
    ts_ack: float = 0.0
    note: str = ""

    @property
    def latency_ms(self) -> float:
        return (self.ts_ack - self.ts_act) * 1000.0 if self.ts_ack else 0.0


class Dispatcher(abc.ABC):
    """执行设备接口：PASS 不动作，RECHECK 进复检区，REJECT 剔除。"""

    @abc.abstractmethod
    def dispatch(self, action: str, ctx: Dict) -> ActResult:
        ...


class NullDispatcher(Dispatcher):
    """离线评估：不动作、不等待，只返回一个成功结果。"""

    def dispatch(self, action: str, ctx: Dict) -> ActResult:
        t = time.time()
        return ActResult(True, f"log:{action}", t, t, "离线模式：不驱动执行机构")


class SimulatedActuator(Dispatcher):
    """模拟三通道执行机构（挡板 + 推杆 + 确认传感器）。

    真机替换点：把 time.sleep 换成“写 CAN/串口指令 + 等确认 IO”，
    返回的 ActResult 保持一致，LineController 和记录都不用改。
    """

    BIN = {"PASS": "bin_OK", "RECHECK": "bin_RECHECK", "REJECT": "bin_REJECT"}

    def __init__(self, delay_ms: float = 60.0, fail_rate: float = 0.0,
                 seed: int = 0, timeout_ms: float = 300.0):
        self.delay_ms = float(delay_ms)
        self.fail_rate = float(fail_rate)
        self.timeout_ms = float(timeout_ms)
        self.rng = random.Random(seed)

    def dispatch(self, action: str, ctx: Dict) -> ActResult:
        ts_act = time.time()
        time.sleep(self.delay_ms / 1000.0)
        if action == "PASS":
            return ActResult(True, "none", ts_act, time.time(), "正常放行，不动作")

        target = self.BIN.get(action, "bin_UNKNOWN")
        failed = self.rng.random() < self.fail_rate
        if failed:
            return ActResult(False, f"push@{target}", ts_act, 0.0,
                             "确认传感器未回报：执行机构可能卡住/瓶盖未进通道")
        return ActResult(True, f"push@{target}", ts_act, time.time(),
                         "确认传感器回报正常")


# ----------------------------------------------------------------------------
# 3) 反馈记录
# ----------------------------------------------------------------------------
class Recorder(abc.ABC):
    @abc.abstractmethod
    def record(self, row: Dict) -> None:
        ...

    def close(self) -> None:
        pass


class CsvRecorder(Recorder):
    """一行一条，立刻 flush。Excel 可直接打开做追溯和统计。"""

    def __init__(self, path: str, fields: Optional[List[str]] = None):
        self.path = path
        self.fields = fields or RECORD_FIELDS
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._rotate_if_schema_mismatch()
        self._new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="", encoding="utf-8-sig")
        self._w = csv.DictWriter(self._fh, fieldnames=self.fields, extrasaction="ignore")
        if self._new:
            self._w.writeheader()

    def _rotate_if_schema_mismatch(self) -> None:
        """旧版本 CSV 的列不一样时，改名备份，避免把两套表头混在一起。"""
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return
        try:
            with open(self.path, "r", encoding="utf-8-sig", newline="") as f:
                header = next(csv.reader(f))
            if header == self.fields:
                return
        except Exception:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        os.replace(self.path, self.path + f".old_{stamp}")

    def record(self, row: Dict) -> None:
        self._w.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class MemoryRecorder(Recorder):
    def __init__(self) -> None:
        self.rows: List[Dict] = []

    def record(self, row: Dict) -> None:
        self.rows.append(dict(row))


class JsonlRecorder(Recorder):
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def record(self, row: Dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class FeedbackSink(abc.ABC):
    """人工复检结果回填接口（H1 的“反馈记录”）。"""

    @abc.abstractmethod
    def submit(self, row: Dict) -> None:
        ...


class CsvFeedbackSink(FeedbackSink):
    FIELDS = ["ts", "frame_id", "original_action", "final_action", "note", "operator"]

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="", encoding="utf-8-sig")
        self._w = csv.DictWriter(self._fh, fieldnames=self.FIELDS, extrasaction="ignore")
        if new:
            self._w.writeheader()

    def submit(self, row: Dict) -> None:
        self._w.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
