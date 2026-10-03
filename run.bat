@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating the project virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv. Install Python 3 and try again.
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
python -c "import cv2, flask, ultralytics" >nul 2>&1
if errorlevel 1 (
    echo Installing project dependencies. This may take a while the first time...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed.
        exit /b 1
    )
)

echo Starting the traffic flow web app at http://127.0.0.1:5000
start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:5000'"
python app.py
endlocal
