#!/usr/bin/env bash
# ==============================================================================
# OceanGo V2 Virtual Environment Setup for macOS (Apple Silicon MPS) & Linux (CUDA 12.8 / CPU)
# ==============================================================================
set -e

echo "=== OceanGo V2 Environment Setup (macOS / Linux) ==="

# 1. Locate a suitable Python binary (Python 3.9 - 3.14+)
PYTHON_BIN=""
for cmd in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_BIN="$cmd"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3 was not found on your system."
    echo "Please install Python 3 (Python 3.9 - 3.14+ supported)."
    exit 1
fi

PY_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
OS_KERNEL=$(uname -s)
echo "Using Python binary: $PYTHON_BIN (version $PY_VER on $OS_KERNEL)"

# 2. Create virtual environment if it does not exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv/..."
    "$PYTHON_BIN" -m venv .venv
else
    echo "Existing virtual environment found in .venv/."
fi

# 3. Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 4. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 5. Install platform-optimal PyTorch (CUDA 12.8 cu128 on Linux w/ NVIDIA GPU; MPS-enabled wheel on macOS)
if [ "$OS_KERNEL" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "Detected NVIDIA GPU on Linux — installing PyTorch with CUDA 12.8 (cu128, supports RTX 5060 Ti / 5090 & RTX 30/40)..."
    pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
else
    echo "Installing PyTorch..."
    pip install --upgrade torch
fi

# 6. Install numpy and pygame (with automatic fallback to pygame-ce on newest Python versions)
echo "Installing numpy and pygame..."
pip install --upgrade numpy
if ! pip install pygame; then
    echo "Standard pygame build failed (e.g. missing SDL headers on newest Python); installing prebuilt pygame-ce..."
    pip install pygame-ce
fi

# 7. Run hardware diagnostic
echo -e "\nRunning OceanGo V2 hardware diagnostic:"
python t.py

echo -e "\nSetup complete! To activate your environment in any terminal, run:"
echo "    source .venv/bin/activate"
echo "Quick end-to-end verification:"
echo "    python OceanGo.py --dry-run"
echo "Play interactively:"
echo "    python OceanGo.py --play"
