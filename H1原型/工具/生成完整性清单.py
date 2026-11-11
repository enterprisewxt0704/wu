# -*- coding: utf-8 -*-
"""生成 H1 原型的「完整性清单」和本地资源缓存。

用法（在 H1原型 目录下）：
    ..\支持文件\Python环境\Scripts\python.exe 工具\生成完整性清单.py
    ..\支持文件\Python环境\Scripts\python.exe 工具\生成完整性清单.py --no-cache

产物：
    H1原型\完整性清单.json
    ..\支持文件\安装缓存\H1原型资源.zip

注意：
- 清单只覆盖「运行必需 + 文档 + 工具 + 测试 + 模型 + 数据」；
  不覆盖 `运行结果/`（结果文件会变）、`__pycache__/`、`.pyc`。
- 资源缓存用于断网/误删时自动恢复，不是交付物；删掉它不影响程序正常运行。
- 每次更新代码/模型/数据后，重新运行本脚本刷新清单和缓存。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import sys
import tempfile
import zipfile
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_NAME = "完整性清单.json"
CACHE_REL = pathlib.Path("../支持文件/安装缓存/H1原型资源.zip")
SUPPORT_REL = pathlib.Path("../支持文件/Python环境")
WHEEL_REL = pathlib.Path("../支持文件/离线安装包/wheels")

# 需要纳入完整性检查的目录（相对 H1原型）
DIRS = ["core", "devices", "ui", "工具", "测试", "文档", "模型", "数据"]
# 根目录下需要纳入的文件
ROOT_FILES = ["config.json", "README.md", "启动.bat", "check_integrity.py"]
# 这些路径只检查“是否存在”，不校验哈希：用户会改配置/文档/工具，不应被完整性检查拦住
MUTABLE_PREFIXES = ("文档/", "测试/", "工具/")
MUTABLE_FILES = {"config.json", "README.md", "启动.bat"}

# 支持包版本（与当前环境一致；检查器只校验这些）
PACKAGES = [
    "torch",
    "torchvision",
    "numpy",
    "Pillow",
    "sympy",
    "networkx",
    "jinja2",
    "filelock",
    "fsspec",
    "setuptools",
    "typing_extensions",
    "mpmath",
    "MarkupSafe",
]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for dirname in DIRS:
        base = ROOT / dirname
        if not base.exists():
            print(f"[WARN] 目录不存在，跳过：{dirname}", flush=True)
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            parts = p.relative_to(ROOT).parts
            if "__pycache__" in parts or p.suffix in (".pyc", ".pyo"):
                continue
            files.append(p)
    for name in ROOT_FILES:
        p = ROOT / name
        if p.is_file():
            files.append(p)
        else:
            print(f"[WARN] 文件不存在，跳过：{name}", flush=True)
    return files


def is_mutable(rel: str) -> bool:
    return rel in MUTABLE_FILES or rel.startswith(MUTABLE_PREFIXES)


def package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="只写清单，不重建资源 zip")
    args = ap.parse_args()

    files = iter_files()
    entries = []
    total = 0
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        size = p.stat().st_size
        print(f"[hash] {rel} ({size/1e6:.2f} MB)", flush=True)
        entries.append(
            {
                "path": rel,
                "size": size,
                "sha256": sha256(p),
                "mutable": is_mutable(rel),
            }
        )
        total += size

    wheels = []
    wheel_dir = (ROOT / WHEEL_REL).resolve()
    if wheel_dir.exists():
        for p in sorted(wheel_dir.glob("*.whl")):
            wheels.append({"name": p.name, "size": p.stat().st_size})

    manifest = {
        "schema": 1,
        "project": "H1原型",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_base": "",
        "cache_zip": CACHE_REL.as_posix(),
        "files": entries,
        "support": {
            "python": (SUPPORT_REL / "Scripts/python.exe").as_posix(),
            "venv_dir": SUPPORT_REL.as_posix(),
            "wheel_dir": WHEEL_REL.as_posix(),
            "packages": package_versions(),
            "wheels": wheels,
        },
    }

    manifest_path = ROOT / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[OK] 清单：{manifest_path}（{len(entries)} 个文件，{total/1e6:.1f} MB）", flush=True)

    if not args.no_cache:
        cache_path = (ROOT / CACHE_REL).resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(suffix=".zip", dir=str(cache_path.parent))
        os.close(fd)
        pathlib.Path(tmp_name).unlink(missing_ok=True)
        tmp = pathlib.Path(tmp_name)
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in files:
                zf.write(p, p.relative_to(ROOT).as_posix())
            zf.writestr(MANIFEST_NAME, manifest_path.read_bytes())
        tmp.replace(cache_path)
        print(f"[OK] 缓存：{cache_path}（{cache_path.stat().st_size/1e6:.1f} MB）", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
