@echo off
chcp 65001 >nul
cd /d %~dp0

rem 优先用项目自带的 Python 运行时；找不到就退回系统 PATH 里的 python
set "PY=C:\Users\wuxuanting\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if not exist "%PY%" set "PY=python"

echo 正在启动瓶盖质检可视化界面...
"%PY%" gui.py
if errorlevel 1 (
  echo.
  echo 启动失败。请确认已安装 Python 3.9+ 与 numpy、Pillow、tkinter。
  pause
)
