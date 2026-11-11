# -*- coding: utf-8 -*-
"""H1 原型文件完整性检查 / 自动恢复。

检查两类东西：
1. 项目文件：源码、模型、数据等运行必需文件按 `完整性清单.json` 的 sha256 校验；
   `config.json`、README、文档、工具、测试这类允许手改的文件只查存在。
2. 支持环境：`支持文件/Python环境` 是否存在，torch/torchvision/numpy/Pillow 等包是否齐，
   离线 wheel 是否齐。

发现缺失时：
- 项目文件：优先从 `支持文件/安装缓存/H1原型资源.zip` 恢复；配置了 `source_base` 时从 HTTP 下载。
- Python 包：先调用 `工具/fetch_wheels.py` 下载缺的 wheel，再用 pip --no-index --no-deps 安装。
- 校验失败但文件存在（例如运行必需源码被误改）：默认只报告，不覆盖；加 `--force` 才恢复。

用法（在 H1原型 目录下）：
    ..\\支持文件\\Python环境\\Scripts\\python.exe check_integrity.py --check
    ..\\支持文件\\Python环境\\Scripts\\python.exe check_integrity.py --auto
    ..\\支持文件\\Python环境\\Scripts\\python.exe check_integrity.py --auto --force

退出码：0 正常/已修复；1 仍有问题或清单/缓存损坏。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = "完整性清单.json"
DEFAULT_CACHE = pathlib.Path("../支持文件/安装缓存/H1原型资源.zip")
UA = {"User-Agent": "H1-prototype-integrity-check/1.0"}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config() -> dict:
    p = ROOT / "config.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def integrity_cfg(cfg: dict) -> dict:
    v = cfg.get("integrity")
    return v if isinstance(v, dict) else {}


def cache_path(cfg: dict, manifest: dict | None = None) -> pathlib.Path:
    raw = integrity_cfg(cfg).get("cache_zip") or (manifest or {}).get("cache_zip") or str(DEFAULT_CACHE)
    return (ROOT / raw).resolve()


def load_manifest(cfg: dict) -> tuple[dict, str | None]:
    """返回 (manifest, 提示)。清单缺失时尝试从缓存 zip 里取。"""
    name = integrity_cfg(cfg).get("manifest") or DEFAULT_MANIFEST
    p = ROOT / name
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8")), None
        except Exception as e:
            raise SystemExit(f"[FAIL] 清单不是合法 JSON：{p}\n{e}")

    cache = cache_path(cfg)
    if cache.exists():
        try:
            with zipfile.ZipFile(cache) as zf:
                data = zf.read(DEFAULT_MANIFEST)
            return json.loads(data.decode("utf-8")), f"清单缺失，已从缓存读取：{cache}"
        except Exception as e:
            raise SystemExit(f"[FAIL] 清单缺失，缓存里也读不到：{cache}\n{e}")

    raise SystemExit(
        "[FAIL] 找不到完整性清单，也没有本地资源缓存。\n"
        f"  期望：{p}\n  缓存：{cache}\n"
        "请从完整包恢复，或运行 工具\\生成完整性清单.py 重新生成。"
    )


def check_bytes(data: bytes, entry: dict) -> bool:
    return len(data) == int(entry["size"]) and sha256_bytes(data) == entry["sha256"]


def atomic_write(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with open(fd, "wb") as f:
            f.write(data)
        pathlib.Path(tmp_name).replace(path)
    finally:
        pathlib.Path(tmp_name).unlink(missing_ok=True)


def restore_from_cache(rel: str, entry: dict, cache: pathlib.Path) -> tuple[bool, str]:
    if not cache.exists():
        return False, "无本地缓存"
    try:
        with zipfile.ZipFile(cache) as zf:
            try:
                data = zf.read(rel)
            except KeyError:
                return False, "缓存里没有这个文件"
    except Exception as e:
        return False, f"缓存损坏：{e}"
    if not check_bytes(data, entry):
        return False, "缓存里的文件校验失败"
    atomic_write(ROOT / rel, data)
    return True, "已从缓存恢复"


def download_file(rel: str, entry: dict, source_base: str, timeout: int) -> tuple[bool, str]:
    if not source_base:
        return False, "未配置 source_base，无法联网下载"
    url = source_base.rstrip("/") + "/" + urllib.parse.quote(rel)
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
    except Exception as e:
        return False, f"下载失败：{e}"
    if not check_bytes(data, entry):
        return False, "下载内容校验失败"
    atomic_write(ROOT / rel, data)
    return True, f"已下载 {url}"


def report_project_files(
    manifest: dict,
    cfg: dict,
    *,
    auto: bool,
    force: bool,
    verbose: bool,
    problems: list[str],
    fixed: list[str],
) -> None:
    entries = manifest.get("files") or []
    cache = cache_path(cfg, manifest)
    source_base = integrity_cfg(cfg).get("source_base") or manifest.get("source_base") or ""
    timeout = int(integrity_cfg(cfg).get("download_timeout_s") or 120)
    log(f"[1/2] 项目文件：{len(entries)} 个")
    for i, entry in enumerate(entries, 1):
        rel = entry["path"]
        p = ROOT / rel
        if not p.exists():
            msg = f"缺失 {rel}"
            if auto:
                ok, why = restore_from_cache(rel, entry, cache)
                if not ok:
                    ok, why = download_file(rel, entry, source_base, timeout)
                if ok:
                    fixed.append(f"{rel}: {why}")
                    log(f"  [FIX ] {rel}  {why}")
                else:
                    problems.append(f"{msg}；{why}")
                    log(f"  [MISS] {rel}  {why}")
            else:
                problems.append(msg)
                log(f"  [MISS] {rel}")
            continue

        if entry.get("mutable"):
            if verbose:
                log(f"  [ OK ] {rel}（可修改文件，只查存在）")
            continue

        size_ok = p.stat().st_size == int(entry["size"])
        hash_ok = size_ok and sha256_file(p) == entry["sha256"]
        if not hash_ok:
            what = "大小不符" if not size_ok else "sha256 不符"
            if auto and force:
                ok, why = restore_from_cache(rel, entry, cache)
                if not ok:
                    ok, why = download_file(rel, entry, source_base, timeout)
                if ok:
                    fixed.append(f"{rel}: {why}")
                    log(f"  [FIX ] {rel}  {why}")
                else:
                    problems.append(f"校验失败 {rel}（{what}）；{why}")
                    log(f"  [BAD ] {rel}（{what}）{why}")
            else:
                problems.append(f"校验失败 {rel}（{what}）")
                log(f"  [BAD ] {rel}（{what}；加 --force 可恢复）")
        elif verbose:
            log(f"  [ OK ] {rel}")


def run_python(python: pathlib.Path, args: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(python), *args],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
    )


def package_versions(python: pathlib.Path, names: list[str]) -> dict[str, str]:
    # 分发名 -> import 名；用 find_spec 防止“只有 dist-info、包目录被删”漏检。
    module_names = {
        "Pillow": "PIL",
        "MarkupSafe": "markupsafe",
        "typing_extensions": "typing_extensions",
        "setuptools": "setuptools",
    }
    modules = {n: module_names.get(n, n) for n in names}
    code = (
        "import importlib.metadata as m, importlib.util as u, json\n"
        f"names = {names!r}\n"
        f"modules = {modules!r}\n"
        "out = {}\n"
        "for n in names:\n"
        "    mod = modules.get(n, n)\n"
        "    try:\n"
        "        found = u.find_spec(mod) is not None\n"
        "    except Exception:\n"
        "        found = False\n"
        "    try:\n"
        "        ver = m.version(n)\n"
        "    except Exception:\n"
        "        ver = ''\n"
        "    out[n] = ver if found else ''\n"
        "print(json.dumps(out, ensure_ascii=False))\n"
    )
    try:
        r = run_python(python, ["-c", code], 60)
    except Exception as e:
        return {}
    if r.returncode != 0:
        return {}
    lines = [x for x in r.stdout.splitlines() if x.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except Exception:
        return {}


def ensure_wheels(
    python: pathlib.Path,
    wheel_dir: pathlib.Path,
    expected: list[dict],
    problems: list[str],
    fixed: list[str],
    auto: bool,
) -> None:
    missing = []
    for w in expected:
        p = wheel_dir / w["name"]
        if not p.exists() or p.stat().st_size != int(w["size"]):
            missing.append(w["name"])
    if not missing:
        return
    log(f"  [WHEEL] 缺失/不完整：{', '.join(missing)}")
    if not auto:
        problems.extend(f"wheel 缺失：{n}" for n in missing)
        return

    fetch = ROOT / "工具" / "fetch_wheels.py"
    if not fetch.exists():
        problems.append("wheel 缺失，且找不到 工具/fetch_wheels.py")
        return
    log("  [WHEEL] 调用 工具/fetch_wheels.py 下载（torch 约 2.6 GB，可能较久）...")
    try:
        r = run_python(python, [str(fetch)], 3600)
    except Exception as e:
        problems.append(f"下载 wheel 失败：{e}")
        return
    if r.returncode != 0:
        problems.append("下载 wheel 失败，见 工具/fetch_wheels.py 输出")
        log((r.stderr or r.stdout or "")[-1200:])
        return
    still = [
        w["name"]
        for w in expected
        if not (wheel_dir / w["name"]).exists()
        or (wheel_dir / w["name"]).stat().st_size != int(w["size"])
    ]
    if still:
        problems.append("下载后仍缺 wheel：" + ", ".join(still))
    else:
        fixed.append("已补齐离线 wheel：" + ", ".join(missing))


def install_wheels(
    python: pathlib.Path,
    wheel_dir: pathlib.Path,
    problems: list[str],
    fixed: list[str],
    only: list[str] | None = None,
) -> None:
    wheels = sorted(wheel_dir.glob("*.whl"))
    if only:
        wanted = {n.lower().replace("-", "_") for n in only}
        picked = []
        for p in wheels:
            stem = p.name.split("-")[0].lower().replace("-", "_")
            if stem in wanted:
                picked.append(p)
        if picked:
            wheels = picked
    if not wheels:
        problems.append("没有可安装的 wheel")
        return
    log(f"  [PIP ] 离线安装 {len(wheels)} 个 wheel（torch 较大时请稍候）...")
    args = [
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
        "--disable-pip-version-check",
        "--no-warn-script-location",
        *[str(p) for p in wheels],
    ]
    try:
        r = run_python(python, args, 3600)
    except Exception as e:
        problems.append(f"pip 安装失败：{e}")
        return
    if r.returncode != 0:
        problems.append("pip 安装失败")
        log((r.stdout + "\n" + r.stderr)[-2000:])
        return
    fixed.append(f"已离线安装 {len(wheels)} 个 wheel")


def report_support(
    manifest: dict,
    cfg: dict,
    *,
    auto: bool,
    problems: list[str],
    fixed: list[str],
) -> None:
    sup = manifest.get("support") or {}
    python_rel = integrity_cfg(cfg).get("support_python") or sup.get("python") or ""
    python = (ROOT / python_rel).resolve() if python_rel else None
    log("[2/2] 支持环境")
    if not python or not python.exists():
        problems.append(f"Python 环境缺失：{python or '(未配置)'}")
        log(f"  [MISS] Python 解释器：{python or '(未配置)'}")
        log("        无法仅靠本脚本创建解释器；请按 文档/环境与离线安装.md 重建。")
        return
    log(f"  [ OK ] Python：{python}")

    venv_dir = (ROOT / (sup.get("venv_dir") or "")).resolve() if sup.get("venv_dir") else python.parents[1]
    if not (venv_dir / "pyvenv.cfg").exists():
        problems.append(f"venv 标记缺失：{venv_dir / 'pyvenv.cfg'}")
        log(f"  [BAD ] pyvenv.cfg 缺失：{venv_dir / 'pyvenv.cfg'}")
    else:
        log("  [ OK ] pyvenv.cfg")

    if not integrity_cfg(cfg).get("check_support", True):
        log("  [SKIP] check_support=false")
        return

    expected_pkgs = sup.get("packages") or {}
    if expected_pkgs:
        got = package_versions(python, list(expected_pkgs))
        bad = [n for n, v in expected_pkgs.items() if v and got.get(n, "") != v]
        if not bad:
            log(f"  [ OK ] Python 包：{len(expected_pkgs)} 个版本一致")
        else:
            for n in bad:
                log(f"  [BAD ] 包 {n}: 期望 {expected_pkgs[n]}，实际 {got.get(n, '未安装')}")
            if auto:
                wheel_dir = (ROOT / (sup.get("wheel_dir") or "")).resolve() if sup.get("wheel_dir") else python.parents[1] / "wheels"
                ensure_wheels(python, wheel_dir, sup.get("wheels") or [], problems, fixed, auto)
                install_wheels(python, wheel_dir, problems, fixed, only=bad)
                got2 = package_versions(python, list(expected_pkgs))
                bad2 = [n for n, v in expected_pkgs.items() if v and got2.get(n, "") != v]
                if bad2:
                    problems.append("安装后仍不完整：" + ", ".join(bad2))
                else:
                    fixed.append("Python 包已补齐")
            else:
                problems.append("Python 包缺失/版本不符：" + ", ".join(bad))
    else:
        log("  [SKIP] 清单没有记录 Python 包版本")


def main() -> int:
    ap = argparse.ArgumentParser(description="H1 原型完整性检查 / 自动恢复")
    ap.add_argument("--check", action="store_true", help="只检查，不修改（默认）")
    ap.add_argument("--auto", action="store_true", help="自动恢复缺失文件 / 下载并安装缺的包")
    ap.add_argument("--force", action="store_true", help="文件存在但哈希不符时也覆盖恢复（默认不覆盖）")
    ap.add_argument("--verbose", action="store_true", help="列出每个正常文件")
    ap.add_argument("--no-support", action="store_true", help="跳过支持环境检查")
    args = ap.parse_args()
    auto = bool(args.auto) and not args.check

    cfg = load_config()
    manifest, note = load_manifest(cfg)
    if note:
        log(f"[INFO] {note}")
        if auto and not (ROOT / (integrity_cfg(cfg).get("manifest") or DEFAULT_MANIFEST)).exists():
            p = ROOT / (integrity_cfg(cfg).get("manifest") or DEFAULT_MANIFEST)
            cache = cache_path(cfg, manifest)
            with zipfile.ZipFile(cache) as zf:
                atomic_write(p, zf.read(DEFAULT_MANIFEST))
            log(f"[FIX ] 已恢复 {p}")

    log(f"H1 原型完整性检查（{'自动修复' if auto else '只检查'}）")
    log(f"根目录：{ROOT}")
    log("")

    problems: list[str] = []
    fixed: list[str] = []
    report_project_files(
        manifest,
        cfg,
        auto=auto,
        force=args.force,
        verbose=args.verbose,
        problems=problems,
        fixed=fixed,
    )
    if not args.no_support and integrity_cfg(cfg).get("check_support", True):
        report_support(manifest, cfg, auto=auto, problems=problems, fixed=fixed)
    else:
        log("[2/2] 支持环境：跳过")

    log("")
    if fixed:
        log(f"[已修复] {len(fixed)} 项")
    if problems:
        log(f"[FAIL] 仍有 {len(problems)} 个问题：")
        for x in problems:
            log(f"  - {x}")
        return 1
    log("[PASS] 文件与支持环境完整。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
