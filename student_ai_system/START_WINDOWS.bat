@echo off
title MIT College Student AI System
color 0A
cls

echo.
echo  ============================================================
echo   MIT College of Railway Engineering, Barshi
echo   Student AI Performance System v3
echo  ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    python3 --version >nul 2>&1
    if errorlevel 1 (
        echo  [ERROR] Python not found!
        echo.
        echo  Please install Python 3.9+ from https://www.python.org/downloads/
        echo  Make sure to check "Add Python to PATH" during installation.
        echo.
        pause
        start https://www.python.org/downloads/
        exit /b 1
    )
    set PYTHON=python3
) else (
    set PYTHON=python
)

echo  [1/3] Python found. Checking dependencies...
echo.

:: Install dependencies quietly
%PYTHON% -m pip install flask pandas numpy scikit-learn joblib openpyxl reportlab werkzeug --quiet --disable-pip-version-check 2>nul
if errorlevel 1 (
    echo  [WARN] Some packages may not have installed. Trying pip3...
    pip3 install flask pandas numpy scikit-learn joblib openpyxl reportlab werkzeug --quiet 2>nul
)

echo  [2/3] Dependencies ready.
echo.

:: Check if models exist, train if not
if not exist "models\dropout_model.pkl" (
    echo  [3/3] Training ML models for first run (this takes ~30 seconds)...
    %PYTHON% train_model.py
    echo.
) else (
    echo  [3/3] ML models found.
    echo.
)

echo  ============================================================
echo   Starting server...
echo   Opening: http://127.0.0.1:5000
echo  ============================================================
echo.
echo  Press Ctrl+C to stop the application.
echo.

:: Open browser after 2 seconds
start /b cmd /c "timeout /t 2 >nul && start http://127.0.0.1:5000"

:: Start Flask app
%PYTHON% app.py

pause
