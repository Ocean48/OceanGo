@echo off
rem ==============================================================================
rem OceanGo V2 Virtual Environment Setup for Windows 11
rem Supports NVIDIA RTX 50-series Blackwell (5060 Ti / 5090), RTX 30/40, & CPU
rem ==============================================================================
echo ===================================================
echo OceanGo V2 Environment Setup (Windows 11)
echo ===================================================

rem 1. Check Python existence
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: Python was not found in your PATH.
    echo Please install Python (Python 3.9 - 3.14+ supported) from python.org and check "Add python.exe to PATH".
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

rem 5. Detect NVIDIA GPU if present
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo.
    echo Detected NVIDIA GPU via nvidia-smi:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
)

rem 6. Select PyTorch build (CUDA 12.8 cu128 required for RTX 5060 Ti / RTX 5090 Blackwell)
echo.
echo Select PyTorch version to install:
echo [1] Windows with NVIDIA GPU (CUDA 12.8 / cu128 - Supports RTX 5060 Ti, RTX 5090, RTX 30/40) [Default]
echo [2] Standard / CPU only
set /p CHOICE="Enter 1 or 2 [Default is 1]: "
if "%CHOICE%"=="" set CHOICE=1

if "%CHOICE%"=="1" (
    echo Installing PyTorch with CUDA 12.8 ^(cu128^) support...
    pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
) else (
    echo Installing standard CPU PyTorch...
    pip install --upgrade torch
)

rem 7. Install numpy and pygame (falling back to pygame-ce if prebuilt SDL wheel needed on newest Python)
echo Installing numpy and pygame...
pip install --upgrade numpy
pip install pygame
if %ERRORLEVEL% neq 0 (
    echo Standard pygame wheel unavailable for this Python build; installing pygame-ce...
    pip install pygame-ce
)

rem 8. Run hardware diagnostic
echo.
echo Running OceanGo V2 hardware diagnostic:
python t.py

echo.
echo ===================================================
echo Setup complete! To activate your environment in a new terminal:
echo    .venv\Scripts\activate
echo Quick end-to-end verification:
echo    python OceanGo.py --dry-run
echo Play interactively against OceanGo V2:
echo    python OceanGo.py --play
echo Start 9x9 -^> 13x13 -^> 19x19 curriculum training:
echo    python OceanGo.py --train --board-size 9 --iters 20
echo ===================================================
pause
