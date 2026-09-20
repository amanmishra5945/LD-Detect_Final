# run.ps1 — PowerShell One-Click Launcher for Smart LD Screening System
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting Smart Learning Disability Screening System" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# 1. Start Backend
Write-Host "`n[1/3] Starting FastAPI Backend on port 8000..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\backend'; python -m uvicorn main:app --host 127.0.0.1 --port 8000"

# 2. Start Frontend
Write-Host "[2/3] Starting Frontend HTTP Server on port 5500..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\frontend'; python -m http.server 5500 --bind 127.0.0.1"

# 3. Wait and launch browser
Write-Host "[3/3] Waiting for servers to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "Opening application in Chrome / default browser..." -ForegroundColor Green
if (Get-Command "chrome.exe" -ErrorAction SilentlyContinue) {
    Start-Process "chrome.exe" -ArgumentList "http://127.0.0.1:5500"
} else {
    Start-Process "http://127.0.0.1:5500"
}

Write-Host "`n========================================================" -ForegroundColor Green
Write-Host " System is LIVE!" -ForegroundColor Green
Write-Host " - Frontend:   http://127.0.0.1:5500" -ForegroundColor Green
Write-Host " - Backend:    http://127.0.0.1:8000" -ForegroundColor Green
Write-Host " - API Docs:   http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
