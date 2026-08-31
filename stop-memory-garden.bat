@echo off
chcp 65001 >nul

set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8766 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    set FOUND=1
)

if %FOUND%==1 (
    echo Memory Garden stopped.
) else (
    echo Memory Garden is not running.
)
pause
