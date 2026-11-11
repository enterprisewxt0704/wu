# -*- coding: utf-8 -*-
"""设备与记录接口（“感知输入 / 执行输出 / 反馈记录”三个可替换点）。

为什么要单独抽象
----------------
课程要求 H1 里“执行与反馈”要自己补上；而现场真正的相机、光电开关、气缸
现在没有，所以这一层全部做成**接口 + 假实现**：
    · 现在：FileImageSource（读图）、SimulatedActuator（模拟气缸）、CsvRecorder（写 CSV）
    · 以后：CameraSource（相机）、SensorTriggerSource（光电触发）、PlcActuator（真 PLC）
只要新类实现同样的方法，pipeline.py 一行都不用改 —— 这就是“预留接口”。

三个接口
--------
FrameSource   产出 Frame（图像 + 来源 + 时间戳）
Dispatcher    接收处置，执行动作，返回 ActResult（是否成功 / 有无确认）
Recorder      把一条完整记录落盘（CSV / JSONL / 数据库）
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

# ----------------------------------------------------------------------------
# 1) 输入：图像从哪来
# ----------------------------------------------------------------------------
@dataclass
class Frame:
    """一帧待检测图像。真实产线上还会带“触发信号”“编码器计数”等字段。"""

    frame_id: str
    path: Optional[str] = None
    image: Optional[np.ndarray] = None          # 统一用 float32 RGB (H,W,3)
    ts_sense: float = 0.0                       # 到位信号时刻（感知开始）
    meta: Dict[str, str] = field(default_factory=dict)


class FrameSource(abc.ABC):
    """输入设备接口。实现 iter_frames() 即可接入任何来源。"""

    @abc.abstractmethod
    def iter_frames(self) -> Iterator[Frame]:
        ...


class FileImageSource(FrameSource):
    """单张图 / 图片列表：界面上“打开图片”“打开文件夹”走这里。"""

    def __init__(self, paths: Iterable[str]):
        self.paths = list(paths)

    def iter_frames(self) -> Iterator[Frame]:
        for p in self.paths:
            yield Frame(frame_id=os.path.basename(p), path=p, ts_sense=time.time())


class CameraSource(FrameSource):
    """【预留】相机输入：接 USB 相机/海康/大恒 SDK 时实现这一层。

    实现要点（写在注释里，答辩时可直接讲“我知道怎么接”）：
      1. 由光电开关（到位信号）触发，不要用固定延时轮询；
      2. 取流要非阻塞：相机内部缓冲，取“触发那一刻之后的第一帧”；
      3. 每帧必须带时间戳（相机 SDK 或主机时钟），否则无法回放。
    """

    def __init__(self, camera_index: int = 0, exposure_ms: float = 3.0):
        self.camera_index = camera_index
        self.exposure_ms = exposure_ms

    def iter_frames(self) -> Iterator[Frame]:
        raise NotImplementedError(
            "尚未接入真实相机。请用 CameraSource 的子类实现 iter_frames()："
            "打开设备 -> 等待触发 -> 抓帧 -> yield Frame(image=..., ts_sense=...)")


class SensorTriggerSource(FrameSource):
    """【预留】光电开关触发：把“有没有瓶盖到位”变成一次检测事件。

    真实实现通常是：串口/网口/PLC 读到一个上升沿 -> 调用相机抓帧 -> yield Frame。
    """

    def __init__(self, port: str = "COM3"):
        self.port = port

    def iter_frames(self) -> Iterator[Frame]:
        raise NotImplementedError("尚未接入光电开关硬件；接口签名见 FrameSource。")


# ----------------------------------------------------------------------------
# 2) 输出：处置怎么执行
# ----------------------------------------------------------------------------
@dataclass
class ActResult:
    """一次执行的结果，feedback 阶段要用它做“确认”。"""

    ok: bool
    command: str                # 例如 "push@bin_REJECT" / "none"
    ts_act: float               # 动作发出时刻
    ts_ack: float = 0.0         # 确认传感器回报时刻
    note: str = ""

    @property
    def latency_ms(self) -> float:
        return (self.ts_ack - self.ts_act) * 1000.0 if self.ts_ack else 0.0


class Dispatcher(abc.ABC):
    """执行设备接口。PASS 不动作；RECHECK 分流到复检区；REJECT 剔除。"""

    @abc.abstractmethod
    def dispatch(self, action: str, ctx: Dict) -> ActResult:
        ...


class LogOnlyDispatcher(Dispatcher):
    """只记录、不动作（离线评估/界面里默认用它）。"""

    def dispatch(self, action: str, ctx: Dict) -> ActResult:
        t = time.time()
        return ActResult(ok=True, command=f"log:{action}", ts_act=t, ts_ack=t,
                         note="仅记录，无真实执行机构")


class SimulatedActuator(Dispatcher):
    """模拟气缸/分拣机构：带延时、可注入失败，用来演示“异常三分支”。

    真实 PLC 版本只要把 time.sleep 换成“写线圈 + 读确认传感器”即可。
    """

    BIN = {"PASS": "bin_OK", "RECHECK": "bin_RECHECK", "REJECT": "bin_REJECT"}

    def __init__(self, delay_ms: float = 60.0, fail_rate: float = 0.0, seed: int = 0):
        self.delay_ms = delay_ms
        self.fail_rate = fail_rate
        self.rng = random.Random(seed)

    def dispatch(self, action: str, ctx: Dict) -> ActResult:
        ts_act = time.time()
        time.sleep(self.delay_ms / 1000.0)                    # 动作耗时
        failed = self.rng.random() < self.fail_rate
        ts_ack = time.time() + 0.002                           # 假设确认传感器 2 ms 后回报
        if action == "PASS":
            return ActResult(True, "none", ts_act, ts_ack, "正常放行，不动作")
        target = self.BIN.get(action, "bin_UNKNOWN")
        if failed:
            return ActResult(False, f"push@{target}", ts_act, 0.0,
                             "确认传感器未回报：气缸未动作或瓶盖未进入目标通道")
        return ActResult(True, f"push@{target}", ts_act, ts_ack, "确认传感器回报正常")


# ----------------------------------------------------------------------------
# 3) 反馈：记录怎么留
# ----------------------------------------------------------------------------
RECORD_FIELDS = [
    "ts", "frame_id", "image_path", "notch", "dark", "purity", "peak",
    "anomaly_score", "worst_feature", "pred_class", "pred_conf", "class_proba",
    "action", "reason", "is_unknown", "act_ok", "act_command", "act_latency_ms",
    "elapsed_ms",
]


class Recorder(abc.ABC):
    @abc.abstractmethod
    def record(self, row: Dict) -> None:
        ...

    def close(self) -> None:
        pass


class CsvRecorder(Recorder):
    """一条记录一行，可直接拿 Excel 打开做追溯/画图（反馈模块的最小实现）。"""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="", encoding="utf-8-sig")
        self._w = csv.DictWriter(self._fh, fieldnames=RECORD_FIELDS, extrasaction="ignore")
        if self._new:
            self._w.writeheader()

    def record(self, row: Dict) -> None:
        self._w.writerow(row)
        self._fh.flush()          # 立刻落盘：断电/崩溃也不丢已经判过的数据

    def close(self) -> None:
        self._fh.close()


class MemoryRecorder(Recorder):
    """把记录留在内存里（界面批量跑时用，最后一次性写盘）。"""

    def __init__(self) -> None:
        self.rows: List[Dict] = []

    def record(self, row: Dict) -> None:
        self.rows.append(dict(row))


class JsonlRecorder(Recorder):
    """每行一个 JSON：以后接数据库/时序库时更好扩展。"""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def record(self, row: Dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
