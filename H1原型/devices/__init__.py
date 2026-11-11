# -*- coding: utf-8 -*-
"""设备层：输入 / 执行 / 记录 / 仿真产线。

常用入口：
    from devices.base import FileImageSource, NullDispatcher, CsvRecorder
    from devices.sources import FolderSource, CameraSource, SensorTriggerSource
    from devices.actuators import build_dispatcher, CompositeDispatcher
    from devices.sim import SimulatedLine
"""

from .base import (ActResult, CsvFeedbackSink, CsvRecorder, Dispatcher,
                   FeedbackSink, FileImageSource, Frame, FrameSource,
                   JsonlRecorder, MemoryRecorder, NullDispatcher, Recorder,
                   SimulatedActuator)
from .sim import LineStats, SimulatedLine

__all__ = [
    "ActResult", "CsvFeedbackSink", "CsvRecorder", "Dispatcher", "FeedbackSink",
    "FileImageSource", "Frame", "FrameSource", "JsonlRecorder", "MemoryRecorder",
    "NullDispatcher", "Recorder", "SimulatedActuator", "LineStats", "SimulatedLine",
]
