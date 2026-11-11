# -*- coding: utf-8 -*-
"""配置读取与路径解析。

约定：config.json 里的相对路径一律相对 **H1原型/ 根目录**，不是相对当前命令行目录。
这样无论从哪个目录启动 app.py，都能找到模型和数据集。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Optional[str] = None) -> Dict:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: Dict, path: Optional[str] = None) -> None:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.json"
    tmp = cfg_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cfg_path)


def resolve(cfg: Dict, value: str) -> Path:
    """把配置里的相对路径解析成绝对路径；已经是绝对路径就原样返回。"""
    p = Path(value)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def model_path(cfg: Dict) -> Path:
    return resolve(cfg, cfg["deep"]["model_path"])


def classifier_path(cfg: Dict) -> Path:
    return resolve(cfg, cfg["paths"]["model_dir"]) / "classifier.json"


def baseline_path(cfg: Dict) -> Path:
    return resolve(cfg, cfg["paths"]["model_dir"]) / "baseline.json"


def out_dir(cfg: Dict) -> Path:
    p = resolve(cfg, cfg["paths"]["out_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_root(cfg: Dict) -> Path:
    return resolve(cfg, cfg["dataset_root"])


def split_images(cfg: Dict, split: str) -> list[str]:
    """返回某个 split 的图片列表（按文件名排序，保证每次回放顺序一致）。

    课程数据集的目录是 数据/dataset/images/{train,val,test_unlabeled}/<类别>/*.png，
    所以这里递归查找，不假设图片直接放在 split 根目录。
    """
    root = dataset_root(cfg) / "images" / split
    if not root.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return [str(p) for p in sorted(root.rglob("*")) if p.suffix.lower() in exts]


def load_labels(cfg: Dict) -> Dict[str, str]:
    """读 labels_train_val.csv，返回 {文件名: 类别}。没有文件就返回空字典。"""
    import csv

    path = dataset_root(cfg) / cfg.get("labels_csv", "labels_train_val.csv")
    labels: Dict[str, str] = {}
    if not path.exists():
        return labels
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("file_name") or row.get("filename") or row.get("file") or "").strip()
            label = (row.get("class_name") or row.get("label") or row.get("class") or "").strip()
            if name and label:
                labels[name] = label
    return labels
