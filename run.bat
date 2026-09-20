@echo off
title Smart Learning Disability Screening System Launcher
echo ========================================================
echo  Starting Smart Learning Disability Screening System
echo ========================================================
echo.

echo [1/3] Starting FastAPI Backend on port 8000...
start "LD Backend (FastAPI)" cmd /k "cd /d %~dp0backend && python -m uvicorn main:app --host 127.0.0.1 --port 8000"

echo [2/3] Starting Frontend HTTP Server on port 5500...
start "LD Frontend (HTTP 5500)" cmd /k "cd /d %~dp0frontend && python -m http.server 5500 --bind 127.0.0.1"

echo [3/3] Waiting for servers to initialize...
timeout /t 3 /nobreak > nul

echo Opening browser at http://127.0.0.1:5500 ...
start http://127.0.0.1:5500

echo.
echo ========================================================
echo  System is LIVE!
echo  - Frontend: http://127.0.0.1:5500
echo  - Backend API: http://127.0.0.1:8000
echo  - API Docs: http://127.0.0.1:8000/docs
echo ========================================================
echo.
pause
