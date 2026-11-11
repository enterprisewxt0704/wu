# -*- coding: utf-8 -*-
"""开发期截图脚本：启动 GUI -> 跑一张图或 val -> 截图 -> 关闭。

用法：
    python 工具/snap_gui.py                         # 跑 val，存 运行结果/gui_val.png
    python 工具/snap_gui.py --image <path> --preset demo_recheck --out 运行结果/gui_recheck.png
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import ImageGrab

from ui.app import App


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="")
    ap.add_argument("--preset", default="")
    ap.add_argument("--out", default=str(ROOT / "运行结果" / "gui_val.png"))
    ap.add_argument("--tab", type=int, default=0, help="底部 Tab 索引：0 记录 / 1 诊断 / 2 参数 / 3 接口")
    args = ap.parse_args()

    app = App()
    if args.preset:
        app.preset_var.set(args.preset)
        app._on_preset()
    if args.tab:
        app.after(200, lambda: app.nb.select(args.tab))
    action = (lambda: app._run_single(args.image)) if args.image else (lambda: app._run_split("val"))
    start_deadline = time.time() + 20

    def start_when_ready():
        state = app.status_vars["deep"].get()
        if state.startswith("就绪") or state.startswith("不可用") or time.time() > start_deadline:
            action()
        else:
            app.after(300, start_when_ready)

    app.after(600, start_when_ready)
    out = Path(args.out)
    deadline = time.time() + 40

    def check():
        if (not app.running) and app.current is not None:
            app.update_idletasks()
            app.lift()
            app.attributes("-topmost", True)
            app.focus_force()
            app.update()
            time.sleep(0.5)
            app.update()
            x, y = app.winfo_rootx(), app.winfo_rooty()
            w, h = app.winfo_width(), app.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            img.save(out)
            print("SAVED", out, img.size)
            app.after(200, app.destroy)
            return
        if time.time() > deadline:
            print("TIMEOUT", app.running, app.current)
            app.destroy()
            return
        app.after(300, check)

    app.after(1500, check)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
