@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set MG_PUBLIC_DEMO_MODE=true
set MG_DEMO_USE_MODEL=false
echo Memory Garden - synthetic offline demo
echo Open http://127.0.0.1:8876
echo Original notes and the regular database are not used.
uv run memory-garden serve --port 8876
pause
endlocal
