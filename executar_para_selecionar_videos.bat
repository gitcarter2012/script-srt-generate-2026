@echo off
setlocal
cd /d "%~dp0"
set "SCRIPT_PATH=%~dp0gerar_srt.py"
py "%SCRIPT_PATH%" %*
echo.
pause
