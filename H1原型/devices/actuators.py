# -*- coding: utf-8 -*-
"""执行输出：大疆 RoboMaster 电机、PTK 舵机、达妙电机的接口占位。

现在没有实物，所以这里**不假装已经能转**：每个驱动的方法体都抛
NotImplementedError，并在异常信息里写清楚“要接哪个协议、改哪个方法”。
这样答辩时既能讲清楚执行链路，也不会把“没验证过的代码”说成已调通。

真实链路建议
------------
PASS    传送带继续转，不动作
RECHECK PTK 舵机把挡板打到“复检通道”，瓶盖滑进复检盒
REJECT  达妙推杆伸出，把瓶盖推入废品通道；RoboMaster 电机维持传送带节拍
确认    挡板/推杆上的限位开关或电机编码器到位 -> ts_ack；超时 -> 报警 + RECHECK

为什么用 CompositeDispatcher
----------------------------
三种动作可能同时涉及“传送带 + 挡板 + 推杆”，把它们组合在一个类里，
LineController 只看到 dispatch(action, ctx)，不关心底下是 CAN 还是串口。
"""

from __future__ import annotations

import abc
import time
from typing import Dict, Optional

from .base import ActResult, Dispatcher, NullDispatcher, SimulatedActuator


class MotorDriver(abc.ABC):
    """所有执行机构的共同接口。"""

    name = "motor"

    @abc.abstractmethod
    def connect(self) -> None:
        ...

    def close(self) -> None:
        pass

    @abc.abstractmethod
    def describe(self) -> str:
        ...


class RoboMasterConveyor(MotorDriver):
    """大疆 RoboMaster M3508 / C620 电调，CAN 总线。

    典型参数：CAN 波特率 1 Mbps，电机 ID 1–8，电调 ID 1–8；
    控制帧 0x200 + ID，8 字节：电流高/低字节 ×4。
    这里只留接口：真实控制要按 RoboMaster 开发板/电调手册的 CAN 帧格式实现。
    """

    name = "RoboMaster M3508/C620"

    def __init__(self, channel: str = "can0", motor_id: int = 1,
                 gear_ratio: float = 3591.0 / 187.0):
        self.channel = channel
        self.motor_id = motor_id
        self.gear_ratio = gear_ratio

    def connect(self) -> None:
        raise NotImplementedError(
            "RoboMaster CAN 驱动未接入：请实现 connect()（打开 CAN 通道、设置 1 Mbps）"
            "与 set_speed_rpm()/stop()（按 C620 电流帧格式发帧）。")

    def describe(self) -> str:
        return f"{self.name}，{self.channel}，ID={self.motor_id}，减速比 {self.gear_ratio:.2f}"

    def set_speed_rpm(self, rpm: float) -> None:
        raise NotImplementedError("按 C620 手册把目标转速换算成电流，写入 0x200+ID 帧")

    def stop(self) -> None:
        raise NotImplementedError("发零电流帧，并等待电机停转")


class PTKServoGate(MotorDriver):
    """PTK 总线舵机（挡板）。也可换成 PWM 舵机，改 set_angle 的实现即可。

    三通道挡板建议角度：
        PASS     0°   （直通）
        RECHECK  35°  （复检盒）
        REJECT   70°  （废品盒）
    真实角度要在机构上标定，不能照抄。
    """

    name = "PTK 总线舵机"

    def __init__(self, port: str = "COM4", baud: int = 1000000,
                 channel: int = 1, angles: Optional[Dict[str, float]] = None):
        self.port = port
        self.baud = baud
        self.channel = channel
        self.angles = angles or {"PASS": 0.0, "RECHECK": 35.0, "REJECT": 70.0}

    def connect(self) -> None:
        raise NotImplementedError(
            "PTK 舵机未接入：请实现 connect()（打开串口、设置波特率）"
            "与 set_angle()（按 PTK 协议写位置/时间）。")

    def describe(self) -> str:
        return f"{self.name}，{self.port}@{self.baud}，通道 {self.channel}"

    def set_angle(self, angle_deg: float, move_ms: int = 120) -> None:
        raise NotImplementedError("按 PTK 协议发送目标角度与运动时间")

    def move_for(self, action: str, move_ms: int = 120) -> None:
        self.set_angle(self.angles.get(action, 0.0), move_ms)


class DamiaoPusher(MotorDriver):
    """达妙电机（DM4310/DM4340 等）推杆，CAN 总线，MIT/位置模式。

    真实实现要点：
      · 使能电机 -> 位置/速度模式 -> 目标位置；
      · 读回编码器位置作为到位确认；
      · 推杆行程末端加限位，防止堵转。
    """

    name = "达妙 DM 系列电机"

    def __init__(self, channel: str = "can1", motor_id: int = 1,
                 extend_pos: float = 0.08, retract_pos: float = 0.0):
        self.channel = channel
        self.motor_id = motor_id
        self.extend_pos = extend_pos
        self.retract_pos = retract_pos

    def connect(self) -> None:
        raise NotImplementedError(
            "达妙 CAN 驱动未接入：请实现 connect()（打开 CAN、使能电机）"
            "与 push()/retract()（MIT 或位置模式，读编码器确认到位）。")

    def describe(self) -> str:
        return f"{self.name}，{self.channel}，ID={self.motor_id}"

    def push(self) -> None:
        raise NotImplementedError("发目标位置 extend_pos，等待编码器到位")

    def retract(self) -> None:
        raise NotImplementedError("发目标位置 retract_pos，等待编码器到位")


class CompositeDispatcher(Dispatcher):
    """把传送带/挡板/推杆组合成一套三通道执行机构。

    现在只搭骨架；接真机时把三个 driver 的 NotImplementedError 逐个实现，
    本类和 LineController 不需要改。超时/确认逻辑已经在这里写好。
    """

    def __init__(self, conveyor: MotorDriver, gate: PTKServoGate,
                 pusher: DamiaoPusher, timeout_ms: float = 300.0):
        self.conveyor = conveyor
        self.gate = gate
        self.pusher = pusher
        self.timeout_ms = timeout_ms
        self.connected = False

    def connect(self) -> None:
        for d in (self.conveyor, self.gate, self.pusher):
            d.connect()
        self.connected = True

    def close(self) -> None:
        for d in (self.conveyor, self.gate, self.pusher):
            d.close()
        self.connected = False

    def dispatch(self, action: str, ctx: Dict) -> ActResult:
        ts = time.time()
        if not self.connected:
            return ActResult(False, "not_connected", ts, 0.0,
                             "执行机构未连接：请先调用 connect()，或改用模拟执行器")
        try:
            if action == "PASS":
                return ActResult(True, "none", ts, time.time(), "放行，不动作")
            if action == "RECHECK":
                self.gate.move_for("RECHECK")
                return ActResult(True, "gate@RECHECK", ts, time.time(), "挡板打到复检通道")
            if action == "REJECT":
                self.gate.move_for("REJECT")
                self.pusher.push()
                # 真机这里应该轮询限位/编码器，最多等 timeout_ms
                return ActResult(True, "gate@REJECT+push", ts, time.time(), "挡板+推杆动作")
            return ActResult(False, f"unknown:{action}", ts, 0.0, "未知动作")
        except NotImplementedError as exc:
            return ActResult(False, f"{action}:not_implemented", ts, 0.0, str(exc))
        except Exception as exc:  # noqa: BLE001
            return ActResult(False, f"{action}:error", ts, 0.0, f"{type(exc).__name__}: {exc}")


def build_dispatcher(cfg: Dict, mode: str = "sim") -> Dispatcher:
    """按运行模式造执行器。

    mode = offline  批量评估，不动作
    mode = sim      连续仿真，模拟延时/确认
    mode = hardware 真机（会因驱动未实现而返回失败，界面会显示原因）
    """
    if mode == "offline":
        return NullDispatcher()
    if mode == "hardware":
        line = cfg.get("line", {})
        return CompositeDispatcher(
            RoboMasterConveyor(),
            PTKServoGate(),
            DamiaoPusher(),
            timeout_ms=float(line.get("actuator_timeout_ms", 300)),
        )
    rt = cfg.get("runtime", {})
    return SimulatedActuator(
        delay_ms=float(rt.get("actuator_delay_ms", 60)),
        fail_rate=float(rt.get("actuator_fail_rate", 0.0)),
        timeout_ms=float(cfg.get("line", {}).get("actuator_timeout_ms", 300)),
    )
