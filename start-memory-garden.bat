@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   Memory Garden  starting...
echo   URL: http://127.0.0.1:8766
echo   Stop: close this window, or press Ctrl+C,
echo         or double-click close-memory-garden.bat
echo ============================================
echo.

uv run memory-garden serve

echo.
echo Server stopped.
pause
