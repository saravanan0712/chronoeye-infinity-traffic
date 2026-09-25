@echo off
title ChronoEye Infinity Master Launcher
echo =======================================================================
echo          CHRONOEYE INFINITY - ONE-COMMAND MASTER LAUNCHER
echo =======================================================================
echo.

cd /d "%~dp0"

IF EXIST "chronoeye\Scripts\python.exe" (
    SET PYTHON_EXE=chronoeye\Scripts\python.exe
) ELSE (
    SET PYTHON_EXE=python
)

echo Starting Backend Server on http://127.0.0.1:8000 ...
start "ChronoEye Backend" cmd /k "cd /d "%~dp0backend" && set PYTHONPATH=. && ..\%PYTHON_EXE% -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload"

echo Starting Frontend Dashboard on http://localhost:3000 ...
start "ChronoEye Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

timeout /t 3 >nul

echo Opening browser at http://localhost:3000 ...
start http://localhost:3000

echo.
echo =======================================================================
echo ChronoEye Infinity is now running!
echo  - Dashboard UI : http://localhost:3000
echo  - Backend API  : http://127.0.0.1:8000
echo  - Swagger Docs : http://127.0.0.1:8000/docs
echo =======================================================================
echo.
pause
