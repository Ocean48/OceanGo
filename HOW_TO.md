# How to Run OceanGo

OceanGo contains a full AlphaZero-style training pipeline alongside an interactive graphical Go game engine. It is designed to run seamlessly on both **macOS** (Apple Silicon MPS / Intel) and **Windows 11** (NVIDIA CUDA / CPU).

---

## 1. Prerequisites & Environment Setup

- **Python Version**: Python 3.8 – 3.14+ is supported (including Python 3.14).
- **Hardware Acceleration**:
  - **macOS (Apple Silicon M1/M2/M3/M4)**: Uses Apple Metal Performance Shaders (MPS) for GPU acceleration.
  - **Windows 11 (NVIDIA GPU)**: Uses NVIDIA CUDA for GPU acceleration.
  - **CPU Fallback**: OceanGo will automatically fall back to CPU if no compatible GPU accelerator is detected.

---

## 2. Virtual Environment Setup

### Option A: Automated One-Click Setup

- **macOS / Linux**:
  ```bash
  chmod +x setup_venv.sh
  ./setup_venv.sh
  ```
- **Windows 11**:
  Double-click `setup_venv.bat` or run in Command Prompt:
  ```cmd
  setup_venv.bat
  ```

---

### Option B: Manual Setup

#### On macOS
1. **Create the virtual environment**:
   ```bash
   python3 -m venv .venv
   ```
2. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(Note: The official macOS PyTorch wheel automatically includes Apple Silicon MPS GPU acceleration. We use `pygame-ce`, which has pre-built wheels for Python 3.14+ to prevent `'SDL.h' file not found` C-compilation errors).*

#### On Windows 11
1. **Create the virtual environment**:
   ```cmd
   python -m venv .venv
   ```
2. **Activate the virtual environment**:
   - In Command Prompt (`cmd`):
     ```cmd
     .venv\Scripts\activate
     ```
   - In PowerShell:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
3. **Install dependencies**:
   - If you have an **NVIDIA GPU (CUDA)**:
     ```cmd
     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
     pip install -r requirements.txt
     ```
   - If you are running on **CPU only**:
     ```cmd
     pip install -r requirements.txt
     ```

---

## 3. Verify Hardware Acceleration

Run the diagnostic script to check that PyTorch detects your hardware accelerator:
```bash
python t.py
```
- On **macOS Apple Silicon**: You will see `SUCCESS: PyTorch detected Apple Silicon GPU with MPS (Metal Performance Shaders) acceleration!`
- On **Windows 11 with NVIDIA GPU**: You will see `SUCCESS: PyTorch detected NVIDIA GPU with CUDA acceleration!` along with your GPU name.

---

## 4. Command Line Usage

### A. Playing Interactively Against the AI
To immediately play a game on the interactive Pygame board against the latest model:
```bash
python OceanGo.py --play
```
*(Note: `python GPU.py --play` also works as a backwards-compatible shortcut).*
- **Controls**:
  - Left click on board intersections to place stones.
  - Press **`P`** on your keyboard to pass your turn.
- To play against a specific historical snapshot:
  ```bash
  python OceanGo.py --model checkpoints/policy_value_net_iter_3.pth --play
  ```

### B. Training the AI
To start the AlphaZero self-play and training sequence from scratch (or continue from the latest `policy_value_net.pth`):
```bash
python OceanGo.py
```
This automatically initiates:
1. Multi-core self-play across CPU workers utilizing a Centralized GPU Queue architecture.
2. Real-time progress displays for games, loss metrics, and ETA.
3. Batch evaluation on the GPU accelerator (CUDA or MPS), preventing memory bottlenecks.
4. Symmetrical board data augmentation (rotations and reflections).
5. Saving iteration weights sequentially into `checkpoints/` and updating `policy_value_net.pth`.
6. Time-stamped logging (`logs_YYYYMMDD_HHMMSS.log`).

### C. Inspecting Model Parameters
To print the parameter shapes and statistics of a trained model:
```bash
python policy_print.py
```

### D. Customizing Hyperparameters
Modify parameters at the top of `OceanGo.py`:
- `BOARD_SIZE`: Go board dimension (default `9` for 9x9; supports `13` or `19`).
- `N_ITER`: Number of complete training loop iterations (default `50`).
- `N_MCTS_SIMS`: Number of MCTS predictions tree depth loops (default `800`).
- `TRAIN_SAMPLE_SIZE`: Board states sampled from replay buffer per training phase (default `250,000`).
- `MAX_REPLAY_BUFFER_SIZE`: Total replay buffer size (default `500,000`).

---

## 5. Legacy Prototype
The original pure-CPU prototype using random rollout MCTS without neural networks is archived in `OceanGo_legacy.py` for reference.
