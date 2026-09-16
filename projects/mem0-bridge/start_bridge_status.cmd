@echo off
rem ============================================================
rem  Idempotent launcher for the bridge-status sidecar (port 18900)
rem  Used by scheduled task "\Mem0 Stack Auto-Start" - Bridge-Status helper
rem  Safe to run repeatedly: checks /healthz first, never spawns a
rem  duplicate process.
rem ============================================================
setlocal

set "SCRIPT=C:\Users\Administrator\.openclaw\workspace-main\projects\mem0-bridge\bridge_status_server.py"
set "PORT=18900"

rem --- 1) Idempotency check: if /healthz answers, it is already running. ---
for /f "usebackq delims=" %%r in (`powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:18900/healthz' -TimeoutSec 2; 'RUNNING' } catch { 'DOWN' }"`) do set "STATE=%%r"

if /i "%STATE%"=="RUNNING" (
    rem Already listening; do not spawn a duplicate.
    exit /b 0
)

rem --- 2) Resolve python. Prefer full path; fall back to PATH. ---
set "PY=python"
if exist "C:\Python314\python.exe" set "PY=C:\Python314\python.exe"
if exist "C:\Python31\python.exe"   set "PY=C:\Python31\python.exe"

rem --- 3) Start. WEBHOOK_URLS flows through from the task environment if set. ---
start "" /min "%PY%" "%SCRIPT%"

endlocal
exit /b 0