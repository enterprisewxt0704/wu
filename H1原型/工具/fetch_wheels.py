# -*- coding: utf-8 -*-
"""下载 H1 原型所需的全部离线 wheel。

为什么不用 pip 直接装：
本机 DSH 沙箱下，pip 访问 PyPI 时写入的 *.whl.metadata 临时文件会被拒绝读取
（Errno 13），因此改为「先下载 wheel 到本地，再 pip --no-index 安装」。
这个脚本只负责下载，不安装；安装见 install_offline.bat / README。

用法：
    ../支持文件/Python环境/Scripts/python.exe 工具/fetch_wheels.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
WHEEL_DIR = ROOT.parent / "支持文件" / "离线安装包" / "wheels"
WHEEL_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "pip/25.0.1 (h1-prototype offline wheel fetcher)"}

# 固定版本：与课程 README、特征识别工程保持一致，避免“今天装出来是另一个版本”。
TORCH_URL = (
    "https://download-r2.pytorch.org/whl/cu128/"
    "torch-2.11.0%2Bcu128-cp312-cp312-win_amd64.whl"
)
TORCHVISION_URL = (
    "https://download-r2.pytorch.org/whl/cu128/"
    "torchvision-0.26.0%2Bcu128-cp312-cp312-win_amd64.whl"
)

# torch 2.11.0+cu128 的 Requires-Dist（来自 wheel 元数据）
# 另加 torchvision 需要的 numpy / pillow。
DEPS = {
    "filelock": None,
    "typing-extensions": None,
    "setuptools": "81.0.0",          # torch 要求 setuptools<82
    "sympy": None,
    "mpmath": None,
    "networkx": None,
    "jinja2": None,
    "MarkupSafe": None,
    "fsspec": None,
    "numpy": "2.3.5",                # 与特征识别/现有环境一致
    "pillow": "13.0.0",              # 课程 README 指定
}


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def _pick_dep_wheel(name: str, version: str | None) -> tuple[str, str]:
    """返回 (filename, url)。优先 cp312 win_amd64，其次 py3-none-any。"""
    meta = _get_json(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
    ver = version or meta["info"]["version"]
    files = _get_json(
        f"https://pypi.org/pypi/{urllib.parse.quote(name)}/{ver}/json"
    )["urls"]

    def score(f: dict) -> int:
        fn = f["filename"]
        if f["packagetype"] != "bdist_wheel":
            return -1
        if "cp312-cp312-win_amd64" in fn:
            return 3
        if "py3-none-any" in fn or "py2.py3-none-any" in fn:
            return 2
        if fn.endswith("none-any.whl"):
            return 1
        return -1

    best = max(files, key=score)
    if score(best) < 0:
        raise RuntimeError(f"{name}=={ver} 没有可用的 Windows/py3 wheel")
    return best["filename"], best["url"]


def download(url: str, filename: str) -> pathlib.Path:
    dest = WHEEL_DIR / filename
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length") or 0)
        if dest.exists() and total and dest.stat().st_size == total:
            print(f"[skip] {filename} ({total/1e6:.1f} MB)", flush=True)
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        done = 0
        last = time.time()
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last >= 2.0:
                    pct = f"{done*100/total:5.1f}%" if total else "  ?  "
                    print(
                        f"[get ] {filename} {pct} "
                        f"{done/1e6:8.1f}/{total/1e6:.1f} MB",
                        flush=True,
                    )
                    last = now
        tmp.replace(dest)
        print(f"[done] {filename} ({done/1e6:.1f} MB)", flush=True)
        return dest


def main() -> int:
    print(f"wheel dir: {WHEEL_DIR}", flush=True)
    download(TORCH_URL, "torch-2.11.0+cu128-cp312-cp312-win_amd64.whl")
    download(TORCHVISION_URL, "torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl")
    for name, ver in DEPS.items():
        fn, url = _pick_dep_wheel(name, ver)
        download(url, fn)
    print("\nall wheels are in:", WHEEL_DIR, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
