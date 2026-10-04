@echo off
rem ============================================================
rem  LuoK pattern-memory project : similarity viewer launcher
rem  Starts a local server and opens the browser.
rem  Close this window to stop the server.
rem ============================================================
cd /d "%~dp0"
set "PY=C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] python not found: %PY%
  pause
  exit /b 1
)
echo.
echo   LuoK similarity viewer
echo   URL : http://127.0.0.1:8765/
echo   Keep this window open. Press Ctrl+C to stop.
echo.
"%PY%" scripts\serve.py --open --port 8765
pause
