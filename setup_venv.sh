#!/usr/bin/env bash
# ==============================================================================
# OceanGo Virtual Environment Setup for macOS / Linux
# ==============================================================================
set -e

echo "=== OceanGo Environment Setup (macOS / Linux) ==="

# 1. Locate a suitable Python binary (Python 3.8 - 3.14+)
PYTHON_BIN=""
for cmd in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PYTHON_BIN="$cmd"
        break
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo "Error: Python 3 was not found on your system."
    echo "Please install Python 3 (Python 3.8 - 3.14+ supported)."
    exit 1
fi

PY_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Using Python binary: $PYTHON_BIN (version $PY_VER)"

# 2. Create virtual environment if it does not exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv/..."
    $PYTHON_BIN -m venv .venv
else
    echo "Existing virtual environment found in .venv/."
fi

# 3. Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 4. Upgrade pip and install requirements
echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

# 5. Run hardware diagnostic
echo -e "\nRunning hardware acceleration diagnostic:"
python t.py

echo -e "\nSetup complete! To activate your environment in any terminal, run:"
echo "    source .venv/bin/activate"
echo "To test or run OceanGo:"
echo "    python OceanGo.py --play"
