@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set MG_PUBLIC_DEMO_MODE=true
set MG_DEMO_USE_MODEL=true
echo Memory Garden - Agent demo with synthetic notes
echo Uses your configured generation model. Test messages and synthetic excerpts are sent to it.
echo Private Vault and existing conversations are not loaded. Cloud embedding and rerank stay off.
echo Open http://127.0.0.1:8876
uv run memory-garden serve --port 8876
pause
endlocal
