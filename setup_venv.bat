@echo off
rem ==============================================================================
rem OceanGo Virtual Environment Setup for Windows 11
rem ==============================================================================
echo ===================================================
echo OceanGo Environment Setup (Windows 11)
echo ===================================================

rem 1. Check Python existence
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python was not found in your PATH.
    echo Please install Python (Python 3.8 - 3.14+ supported) from python.org and check "Add python.exe to PATH".
    pause
    exit /b 1
)

rem 2. Create virtual environment
if not exist ".venv" (
    echo Creating virtual environment in .venv...
    python -m venv .venv
) else (
    echo Existing virtual environment found in .venv.
)

rem 3. Activate virtual environment
call .venv\Scripts\activate.bat

rem 4. Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

rem 5. Optional choice for NVIDIA CUDA PyTorch
echo.
echo Select PyTorch version to install:
echo [1] Windows with NVIDIA GPU (CUDA 12.1 acceleration - Recommended for NVIDIA)
echo [2] Standard / CPU only
set /p CHOICE="Enter 1 or 2 [Default is 1]: "
if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="1" (
    echo Installing PyTorch with CUDA 12.1 support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
    echo Installing standard PyTorch...
    pip install torch torchvision
)

echo Installing remaining requirements (numpy, pygame-ce)...
pip install -r requirements.txt

rem 6. Run diagnostic
echo.
echo Running diagnostic test:
python t.py

echo.
echo ===================================================
echo Setup complete! To activate your environment in a new terminal:
echo    .venv\Scripts\activate
echo To run OceanGo:
echo    python OceanGo.py --play
echo ===================================================
pause
