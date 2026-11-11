@echo off
REM H1 prototype launcher (ASCII only so it works on any Windows code page).
setlocal
cd /d "%~dp0"

set "PY="
for /f "delims=" %%P in ('dir /b /s "%~dp0..\python.exe" 2^>nul') do (
    if exist "%%~dpP..\pyvenv.cfg" if not defined PY set "PY=%%P"
)

if not defined PY (
    echo [ERROR] Python environment not found.
    echo Keep the support folder next to this project, then try again.
    echo See README.md "Quick start" or docs for rebuilding the environment.
    pause
    exit /b 1
)

if exist "check_integrity.py" (
    echo [1/2] Checking files and support packages ...
    "%PY%" "check_integrity.py" --auto
    if errorlevel 1 (
        echo.
        echo [ERROR] Integrity check failed. Fix the items above, then retry.
        echo Tip: run "check_integrity.py --auto --force" to restore modified files.
        pause
        exit /b 1
    )
) else (
    echo [WARN] check_integrity.py is missing; skip the integrity check.
)

echo [2/2] Starting the H1 prototype ...
"%PY%" "ui\app.py" %*

if errorlevel 1 (
    echo.
    echo [ERROR] Launch failed. See README.md "Quick start".
    pause
)
endlocal
