# -*- coding: utf-8 -*-
"""检测输入的具体实现。

已经可用
--------
FileImageSource  单图 / 图片列表
FolderSource     文件夹（按文件名排序，保证回放顺序固定）
ReplaySource     循环回放一组图，模拟“传送带上不断来瓶盖”
VideoSource      视频文件（需要 opencv-python；没装时给出清楚的提示）

预留（接口已定，硬件到了补方法体即可）
------------------------------------
CameraSource        工业相机 / USB 相机
SensorTriggerSource 对射式红外光电开关（到位信号）

接真机的正确顺序（答辩可以照着讲）
--------------------------------
1. 光电开关检测到瓶盖前沿 -> 上升沿中断/PLC 输入点；
2. 触发相机软触发（不要自由运行 + 固定延时，节拍一变就拍不到）；
3. 相机曝光完成 -> 取一帧 RGB + 时间戳；
4. 这一帧进入 Inspector，同时记下触发时刻，用于后面的跟踪延时补偿；
5. 执行机构在“瓶盖走到挡板/推杆”的时刻动作，而不是判决完立刻动作。
"""

from __future__ import annotations

import os
import time
from typing import Iterable, Iterator, List, Optional

import numpy as np

from .base import Frame, FrameSource


class FolderSource(FrameSource):
    def __init__(self, folder: str, exts: Optional[Iterable[str]] = None):
        self.folder = folder
        self.exts = {e.lower() for e in (exts or [".png", ".jpg", ".jpeg", ".bmp"])}

    def paths(self) -> List[str]:
        if not os.path.isdir(self.folder):
            return []
        out = []
        for root, _dirs, files in os.walk(self.folder):
            for name in files:
                if os.path.splitext(name)[1].lower() in self.exts:
                    out.append(os.path.join(root, name))
        return sorted(out)

    def iter_frames(self) -> Iterator[Frame]:
        for p in self.paths():
            yield Frame(frame_id=os.path.basename(p), path=p, ts_sense=time.time())


class ReplaySource(FrameSource):
    """把一组图片循环播放，用来做“连续处理 ≥20 个样本”的仿真演示。"""

    def __init__(self, paths: Iterable[str], cycles: int = 1):
        self.paths = list(paths)
        self.cycles = max(1, int(cycles))

    def iter_frames(self) -> Iterator[Frame]:
        for _ in range(self.cycles):
            for p in self.paths:
                yield Frame(frame_id=os.path.basename(p), path=p, ts_sense=time.time())


class VideoSource(FrameSource):
    """视频输入：逐帧解码后当作图像序列。

    需要 opencv-python（pip install opencv-python）。课程环境没装也不影响主流程，
    所以这里做成“用到才 import”，没装时抛出带安装提示的异常。
    """

    def __init__(self, path: str, stride: int = 1):
        self.path = path
        self.stride = max(1, int(stride))

    def iter_frames(self) -> Iterator[Frame]:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise NotImplementedError(
                "视频输入需要 opencv-python：pip install opencv-python。"
                "也可以先用 FolderSource 把视频抽成图片。"
            ) from exc
        cap = cv2.VideoCapture(self.path)
        idx = 0
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if idx % self.stride == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    yield Frame(frame_id=f"{os.path.basename(self.path)}#{idx}",
                                image=rgb, ts_sense=time.time())
                idx += 1
        finally:
            cap.release()


class CameraSource(FrameSource):
    """【预留】真实相机。子类只需实现 iter_frames()。

    以海康 MVS 为例（USB3/GigE 都类似）：
        cam = MvCamera(); cam.MV_CC_CreateHandle(); cam.MV_CC_OpenDevice()
        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_ON)
        cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0)
        # 等光电开关上升沿 -> MV_CC_TriggerSoftwareExecute()
        frame = cam.MV_CC_GetImageBuffer(); rgb = convert(frame)
        yield Frame(image=rgb, ts_sense=trigger_ts)

    若用 USB 相机（OpenCV）：
        cap = cv2.VideoCapture(index); cap.set(cv2.CAP_PROP_EXPOSURE, ...)
        同样由光电触发，而不是死循环 read()。
    """

    def __init__(self, camera_index: int = 0, exposure_ms: float = 3.0,
                 serial: str = ""):
        self.camera_index = camera_index
        self.exposure_ms = exposure_ms
        self.serial = serial

    def open(self) -> None:
        raise NotImplementedError("尚未接入真实相机：请在子类中实现 open()")

    def close(self) -> None:
        pass

    def iter_frames(self) -> Iterator[Frame]:
        raise NotImplementedError(
            "尚未接入真实相机。实现 CameraSource.iter_frames()："
            "打开设备 -> 等触发 -> 抓一帧 RGB + 时间戳 -> yield Frame(...)")


class SensorTriggerSource(FrameSource):
    """【预留】对射式红外光电开关触发。

    常见接法：
      · NPN 常开 -> PLC 输入点 / 单片机 GPIO / USB 采集卡；
      · 串口模块 -> 读到 "1" 上升沿；
      · 与相机同一触发源 -> 相机硬触发，主机只收图。
    去抖：机械/光电抖动一般 1–5 ms，软件再去抖 10–50 ms；
    重复触发：同一只瓶盖在挡光期间只算一次，等“离开 -> 再来”才产生下一事件。
    """

    def __init__(self, port: str = "COM3", baud: int = 115200,
                 debounce_ms: float = 20.0):
        self.port = port
        self.baud = baud
        self.debounce_ms = debounce_ms

    def iter_frames(self) -> Iterator[Frame]:
        raise NotImplementedError(
            "尚未接入光电开关。实现 SensorTriggerSource.iter_frames()："
            "等上升沿 -> 去抖 -> 触发相机 -> yield Frame(image=..., ts_sense=...)")


class ArraySource(FrameSource):
    """把内存里的 ndarray 列表当输入（单元测试、算法调试用）。"""

    def __init__(self, images: Iterable[np.ndarray]):
        self.images = list(images)

    def iter_frames(self) -> Iterator[Frame]:
        for i, img in enumerate(self.images):
            yield Frame(frame_id=f"array_{i:04d}", image=img, ts_sense=time.time())
