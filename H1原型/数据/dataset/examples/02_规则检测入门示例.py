"""提取可解释基础特征；本示例不提供完整四分类答案。"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

default = Path(__file__).parents[1] / "images/train/normal/img_0_001.png"
parser = argparse.ArgumentParser()
parser.add_argument("image", nargs="?", type=Path, default=default)
path = parser.parse_args().image
gray = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).mean(axis=2)
border = np.concatenate((gray[:30].ravel(), gray[-30:].ravel(), gray[:, :30].ravel(), gray[:, -30:].ravel()))
bg = float(np.median(border))
foreground = np.abs(gray - bg) > 28
features = {
    "背景灰度估计": bg,
    "前景面积比例": float(foreground.mean()),
    "深色区域比例": float(((gray < bg - 75) & foreground).mean()),
    "平均边缘强度": float((np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean()) / 2),
}
print(f"图片：{path}")
for key, value in features.items():
    print(f"{key}: {value:.4f}")
print("提示：请自行比较不同类别特征，并设计判断或学习方法。")

