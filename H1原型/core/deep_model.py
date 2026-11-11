# -*- coding: utf-8 -*-
"""深度模型后端：加载课程基础版的 MobileNetV3-Small（model.pth）。

它不是“大模型”，而是一个 6 MB 的轻量 CNN：在 ImageNet 预训练权重上，
只训练最后的四分类输出层。课程基础版就是用它，本工程把它作为**第二意见**，
与特征模型做 M1/M2/M3 三种融合；torch 没装时自动回退到纯特征（M0）。

为什么 lazy import torch
------------------------
启动界面时不该等 3 秒的 CUDA 初始化；只有第一次真正要跑深度模型时才加载。
这样“只跑特征”的机器、答辩现场没有 GPU 的机器也能打开界面。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .types import CLASSES, ModelOutput


class DeepModel:
    def __init__(self, cfg: Dict, root: Path):
        self.cfg = cfg.get("deep", {})
        self.root = Path(root)
        self.model_path = Path(self.cfg.get("model_path", "模型/model.pth"))
        if not self.model_path.is_absolute():
            self.model_path = self.root / self.model_path
        self.device_name = str(self.cfg.get("device", "auto"))
        self.input_size = int(self.cfg.get("input_size", 512))
        self.batch_size = int(self.cfg.get("batch_size", 16))

        self._lock = threading.Lock()
        self._loaded = False
        self._torch = None
        self._model = None
        self._device = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self.error = ""
        self.load_ms = 0.0

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return self._loaded and self._model is not None

    def load(self) -> bool:
        """加载模型；返回是否成功。失败时把原因放在 self.error，不抛异常。"""
        if self._loaded:
            return self.available
        with self._lock:
            if self._loaded:
                return self.available
            t0 = time.time()
            try:
                import torch
                from torchvision import models

                if not self.model_path.exists():
                    raise FileNotFoundError(f"找不到模型文件：{self.model_path}")

                ckpt = torch.load(str(self.model_path), map_location="cpu", weights_only=True)
                names = list(ckpt.get("class_names", []))
                if names and names != list(CLASSES):
                    raise ValueError(
                        f"model.pth 的类别顺序 {names} 与本工程 {list(CLASSES)} 不一致"
                    )
                mean = ckpt.get("mean", [0.485, 0.456, 0.406])
                std = ckpt.get("std", [0.229, 0.224, 0.225])
                self._mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
                self._std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

                model = models.mobilenet_v3_small(weights=None)
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_features, len(CLASSES))
                model.load_state_dict(ckpt["model_state_dict"])

                if self.device_name == "auto":
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    device = torch.device(self.device_name)
                model.to(device).eval()

                self._torch = torch
                self._model = model
                self._device = device
                self._loaded = True
                self.load_ms = (time.time() - t0) * 1000.0
                return True
            except Exception as exc:  # noqa: BLE001 - 加载失败要显示在界面上，不能崩
                self.error = f"{type(exc).__name__}: {exc}"
                self._loaded = True          # 标记“尝试过”，避免每次刷新都重试
                return False

    def device_text(self) -> str:
        if not self.available:
            return "不可用"
        if self._device is not None and self._device.type == "cuda":
            return f"GPU {self._torch.cuda.get_device_name(0)}"
        return "CPU"

    # ------------------------------------------------------------------
    # 预处理 / 推理
    # ------------------------------------------------------------------
    def _prepare(self, image: np.ndarray) -> np.ndarray:
        """RGB uint8 -> 归一化 float32 CHW。不裁剪，只把非 512 的图缩到 512。"""
        from PIL import Image

        a = np.asarray(image)
        if a.ndim == 2:
            a = np.repeat(a[:, :, None], 3, axis=2)
        if a.shape[2] == 4:
            a = a[:, :, :3]
        if a.shape[:2] != (self.input_size, self.input_size):
            a = np.asarray(
                Image.fromarray(a.astype(np.uint8)).resize(
                    (self.input_size, self.input_size), Image.BILINEAR
                )
            )
        x = a.astype(np.float32) / 255.0
        x = (x - self._mean) / np.maximum(self._std, 1e-6)
        return np.transpose(x, (2, 0, 1))

    def predict(self, image: np.ndarray) -> ModelOutput:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: Sequence[np.ndarray]) -> List[ModelOutput]:
        """批量推理。界面单张、离线评估一批都用它。"""
        n = len(images)
        if n == 0:
            return []
        if not self.load():
            return [ModelOutput.unavailable("deep", self.error or "深度模型不可用") for _ in images]

        t0 = time.time()
        outs: List[ModelOutput] = []
        torch = self._torch
        with self._lock, torch.inference_mode():
            for start in range(0, n, self.batch_size):
                chunk = images[start:start + self.batch_size]
                x = np.stack([self._prepare(im) for im in chunk], axis=0)
                tensor = torch.from_numpy(x).to(self._device)
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                for row in probs:
                    d = {c: float(p) for c, p in zip(CLASSES, row)}
                    defect = float(sum(d[c] for c in CLASSES[1:]))
                    pred = max(d, key=d.get)
                    outs.append(
                        ModelOutput(
                            name="deep",
                            available=True,
                            probs=d,
                            pred=pred,
                            conf=float(d[pred]),
                            defect_score=defect,
                        )
                    )
        per = (time.time() - t0) * 1000.0 / max(n, 1)
        for o in outs:
            o.latency_ms = per
        return outs

    def warmup(self) -> None:
        """用一张灰图预热，避免第一只瓶盖的耗时把节拍统计拉偏。"""
        if self.load():
            try:
                self.predict(np.full((self.input_size, self.input_size, 3), 128, np.uint8))
            except Exception:
                pass
