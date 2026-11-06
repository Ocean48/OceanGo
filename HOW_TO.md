# How to Run OceanGo

OceanGo contains a full AlphaZero-style training pipeline alongside an interactive graphical Go game engine. It is designed to run seamlessly on both **macOS** (Apple Silicon MPS / Intel) and **Windows 11** (NVIDIA CUDA / CPU).

> For a full mathematical breakdown of the MCTS search, PUCT formula, and dual-head ResNet architecture, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

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

### A. Quick Start / Mode Selection
Simply running:
```bash
python OceanGo.py
```
will present an interactive prompt:
```
Select OceanGo Mode:
  [1] Play Go as Black (You move first vs AI White) [Default]
  [2] Play Go as White (AI moves first as Black)
  [3] Start AlphaZero AI Self-Play Training
```
Pressing **Enter** (or `1`) boots up the window with you playing Black. Entering `2` boots up the window with you playing White (AI moves first).

### B. Playing Interactively Against the AI
To jump straight into the Pygame window without prompting:
```bash
# Play as Black (You move first, AI is White) [Default]
python OceanGo.py --play

# Play as White (AI moves first as Black)
python OceanGo.py --play --color white
```
*(Note: `python GPU.py --play` also works as a backwards-compatible shortcut).*
- **Controls & UI**:
  - **Left Click**: Place a stone on any legal intersection.
  - **Pass Button / `P` key**: Pass your turn. When both players pass consecutively, the game ends automatically.
  - **End & Score Button / `E` key**: Immediately ends the game and computes the Chinese area score with 7.5 Komi.
  - **Resign Button**: Concede the match.
  - **Game Over & Board Review**:
    - Automatically displays the winner and detailed Chinese area score breakdown (stones + territory + Komi).
    - **Scoring Squares**: Every single point contributing to a player's score is visually marked with a square:
      - **Black Squares**: Living Black stones, Black empty territory, and captured/dead White stones in Black territory.
      - **White Squares**: Living White stones, White empty territory, and captured/dead Black stones in White territory.
    - **Dead Stone Detection & Toggle**: Obvious dead groups are detected automatically at game end. While reviewing the board, you can also **click on any stone group** to toggle it between Dead and Alive; the scoring squares and final scores update instantly!
    - **Review Board**: Click **Review Board**, the **`[X]`** close button, press **`ESC`** / **`SPACE`**, or click outside the dialog to dismiss the pop-up and inspect the full board and territories.
    - **Swap Side (`S`)**: Click **Swap Side** or press **`S`** to immediately restart a game playing as the other color!
    - **Play Again (`R`) / Exit (`Q`)**: Start a new game with the same color, or quit cleanly.
- To play against a specific historical snapshot:
  ```bash
  python OceanGo.py --model checkpoints/policy_value_net_iter_3.pth --play
  ```

### C. Training the AI
To start the AlphaZero self-play and training sequence directly:
```bash
python OceanGo.py --train
```

#### Training Options:
- `--dry-run`: Runs a fast 10-second end-to-end sanity check (1 game, backpropagation, and save test) to verify hardware and directories before committing to a full run.
- `--board-size <N>`: Sets the Go board dimension: `9`, `13`, or `19` (default: `9`). Automatically scales Dirichlet exploration noise and opening temperature cutoffs for the selected board size.
- `--games <N>`: Sets the number of self-play games per iteration (default: `500`).
- `--sims <N>`: Sets MCTS simulations per move (default: `800`).
- `--iters <N>`: Sets number of training iterations (default: `50`).
- `--save-dir <path>`: Specifies custom directory for checkpoints and models (e.g. Google Drive `/content/drive/MyDrive/OceanGo/checkpoints`).

### D. Training on Google Colab (Cloud GPUs)
For complete instructions on running in the cloud with zero credit waste, see the dedicated **[`COLAB_GUIDE.md`](COLAB_GUIDE.md)**.
A pre-configured Jupyter notebook is included in the root directory: **[`OceanGo_Colab.ipynb`](OceanGo_Colab.ipynb)**.
1. Open [Google Colab](https://colab.research.google.com) and upload `OceanGo_Colab.ipynb`.
2. Select a GPU runtime (`Runtime -> Change runtime type -> GPU`).
3. Run Step 4 (`--dry-run`) to verify CUDA GPU acceleration for 0 compute units.
4. Run Step 5 to launch full training with automatic Google Drive checkpoint syncing.

### E. Inspecting Model Parameters
To print the parameter shapes and statistics of a trained model:
```bash
python policy_print.py
```

### F. Customizing Hyperparameters
Modify parameters at the top of `OceanGo.py`:
- `BOARD_SIZE`: Go board dimension (default `9` for 9x9; supports `13` or `19`).
- `N_ITER`: Number of complete training loop iterations (default `50`).
- `N_MCTS_SIMS`: Number of MCTS predictions tree depth loops (default `800`).
- `TRAIN_SAMPLE_SIZE`: Board states sampled from replay buffer per training phase (default `250,000`).
- `MAX_REPLAY_BUFFER_SIZE`: Total replay buffer size (default `500,000`).

---

## 5. Legacy Prototype
The original pure-CPU prototype using random rollout MCTS without neural networks is archived in `OceanGo_legacy.py` for reference.
