# -*- coding: utf-8 -*-
"""连续仿真产线：没有实物也能演示“一只接一只”的完整闭环。

它做三件事：
    1. 按节拍（默认 900 ms/只）从图片列表取下一只瓶盖；
    2. 调用 Inspector 完成感知/分析/决策/执行/记录；
    3. 把结果交给界面，界面只负责画出来。

为什么不在界面线程里 sleep
--------------------------
Tkinter 是单线程的；在按钮回调里 time.sleep 会把窗口卡死。
所以 SimulatedLine 在后台线程跑，界面用队列收结果（见 ui/app.py）。
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

from .base import Frame


@dataclass
class LineStats:
    total: int = 0
    passed: int = 0
    recheck: int = 0
    rejected: int = 0
    unknown: int = 0
    act_fail: int = 0
    sum_elapsed_ms: float = 0.0

    def add(self, res) -> None:
        self.total += 1
        action = res.action
        if action == "PASS":
            self.passed += 1
        elif action == "RECHECK":
            self.recheck += 1
        elif action == "REJECT":
            self.rejected += 1
        if res.fusion and res.fusion.is_unknown:
            self.unknown += 1
        if res.act and not res.act.ok:
            self.act_fail += 1
        self.sum_elapsed_ms += res.elapsed_ms

    @property
    def avg_ms(self) -> float:
        return self.sum_elapsed_ms / self.total if self.total else 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "total": self.total,
            "passed": self.passed,
            "recheck": self.recheck,
            "rejected": self.rejected,
            "unknown": self.unknown,
            "act_fail": self.act_fail,
            "avg_ms": round(self.avg_ms, 1),
        }


class SimulatedLine:
    def __init__(self, paths: Iterable[str], inspector, cfg: Dict,
                 cycle_ms: Optional[float] = None, loop: bool = True):
        self.paths: List[str] = [p for p in paths if os.path.exists(p)]
        if not self.paths:
            raise ValueError("仿真产线没有可用图片")
        self.inspector = inspector
        self.cfg = cfg
        self.cycle_ms = float(cycle_ms if cycle_ms is not None
                              else cfg.get("line", {}).get("sim_cycle_ms", 900))
        self.loop = loop
        self.index = 0
        self.stats = LineStats()
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.index = 0
            self.stats = LineStats()

    def step(self):
        with self._lock:
            path = self.paths[self.index % len(self.paths)]
            frame_id = os.path.basename(path)
            # 同一张图循环回放时，给 frame_id 加轮次，记录里能区分“第几次”
            round_no = self.index // len(self.paths) + 1
            if round_no > 1:
                frame_id = f"{round_no:02d}_{frame_id}"
            self.index += 1
        res = self.inspector.inspect(
            path, frame_id=frame_id, ts_trigger=time.time(),
            queue_depth=0,   # 顺序仿真：同一时刻只有一只在检，队列深度为 0
        )
        self.stats.add(res)
        return res

    def run(self, on_result: Callable, stop_event: threading.Event,
            max_samples: int = 0, on_done: Optional[Callable] = None) -> None:
        """后台线程入口。每来一只调用 on_result(res)，结束时 on_done(stats)。"""
        count = 0
        try:
            while not stop_event.is_set():
                res = self.step()
                count += 1
                try:
                    on_result(res)
                except Exception:
                    pass
                if max_samples and count >= max_samples:
                    break
                if not self.loop and self.index >= len(self.paths):
                    break
                stop_event.wait(max(self.cycle_ms, 1.0) / 1000.0)
        finally:
            if on_done:
                try:
                    on_done(self.stats)
                except Exception:
                    pass
