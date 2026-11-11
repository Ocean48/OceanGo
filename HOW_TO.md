# How to Run OceanGo V2

OceanGo V2 is a self-contained, single-machine, **board-size-agnostic AlphaZero/KataGo Go engine** supporting progressive curriculum training (`9x9 -> 13x13 -> 19x19`) alongside an interactive Pygame UI. It profiles CPU threads, system RAM, and GPU VRAM automatically at startup across **Windows 11 / Linux (NVIDIA CUDA)** and **macOS (Apple Silicon MPS)**.

> For a full mathematical breakdown of the Global-Pooling ResNet, incremental Zobrist/bitmask Go engine, and MCTS optimizations, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## 1. Prerequisites & Environment Setup

- **Python Version**: Python 3.9 – 3.14+ supported (including Python 3.14).
- **Hardware Acceleration**:
  - **Windows 11 / Linux (NVIDIA GPU)**: Uses CUDA `bfloat16` Tensor Cores + `channels_last` layout. **Important for RTX 5060 Ti / RTX 5090 (Blackwell `sm_120`)**: use PyTorch CUDA 12.8 (`cu128`) wheels (installed automatically by `setup_venv.bat` / `setup_venv.sh`).
  - **macOS (Apple Silicon M1/M2/M3/M4)**: Uses Metal Performance Shaders (`MPS`) acceleration.
  - **Google Colab**: Supports Cloud TPUs (`v5e-1` XLA `bfloat16`) and NVIDIA GPUs (`T4`, `L4`, `A100`).

---

## 2. Virtual Environment Setup

### Option A: Automated One-Click Setup (Recommended)

- **Windows 11 (RTX 5060 Ti / RTX 5090 / RTX 30 & 40 series / CPU)**:
  Double-click `setup_venv.bat` or run in Command Prompt:
  ```cmd
  setup_venv.bat
  ```
  *(Select `[1]` to install CUDA 12.8 `cu128` PyTorch with full Blackwell & Ampere/Ada Tensor Core support).*

- **macOS / Linux**:
  ```bash
  chmod +x setup_venv.sh
  ./setup_venv.sh
  ```
  *(Automatically installs MPS-enabled PyTorch on macOS, or CUDA 12.8 `cu128` PyTorch on Linux when `nvidia-smi` is detected).*

---

### Option B: Manual Setup

#### On Windows 11 / Linux (NVIDIA GPU)
```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pygame
```

#### On macOS (Apple Silicon)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Verify Hardware & Run End-to-End Dry Run

1. **Check GPU / VRAM / bfloat16 Tensor Core status**:
   ```bash
   python t.py
   ```
2. **Run the 5-second end-to-end V2 verification** (tests shared-memory multiprocessing workers, GPU server batching, 8x GPU symmetry + multi-head backprop, Arena evaluation, and V2 checkpoint persistence):
   ```bash
   python OceanGo.py --dry-run --board-size 9
   python OceanGo.py --dry-run --board-size 19
   ```

---

## 4. Progressive Curriculum Training (`9x9 -> 13x13 -> 19x19`)

Because OceanGo V2 uses a **KataGo Global-Pooling ResNet (`15 blocks x 192 channels`)**, **100% of learned weights transfer seamlessly across board sizes**. When `--board-size` increases, OceanGo automatically transfers all trunk and head weights, initializes a fresh bit-packed replay buffer for the new board dimension, resets SGD momentum, and runs a 2-iteration transition LR warmup:

```bash
# Stage 1: Master fast tactical reading, life-and-death, and liberties on 9x9
python OceanGo.py --train --board-size 9 --games 500 --sims 600 --iters 25

# Stage 2: Automatically transfer 100% of 9x9 weights into 13x13 (whole-board influence & joseki)
python OceanGo.py --train --board-size 13 --games 400 --sims 600 --iters 25

# Stage 3: Automatically transfer 100% of 13x13 weights into full 19x19 Go
python OceanGo.py --train --board-size 19 --games 250 --sims 600 --iters 50
```

> **Auto-Resume**: Stopping and restarting any command automatically resumes from the highest `policy_value_net_iter_X.pth` in `--save-dir`, restoring both optimizer momentum and the compact bit-packed replay buffer (`train_state_v2_{N}x{N}.pth`).

### Key Training CLI Flags
| Flag | Default | Description |
|:---|:---:|:---|
| `--board-size` | `9` | Go board dimension (`9`, `13`, or `19`) |
| `--blocks` | `15` | Number of residual blocks (with `GlobalPoolResBlock` every 3rd block) |
| `--channels` | `192` | Convolutional trunk channels |
| `--games` | `500` | Self-play games generated per iteration |
| `--sims` | `600` | Full-search MCTS simulations per move (`25%` full / `75%` fast playout cap) |
| `--iters` | `50` | Training iterations to execute |
| `--workers` | `auto` | Parallel CPU self-play workers (`sys_cores - 2`) |
| `--batch-size` | `auto` | Training minibatch size (`256` on >=12GB CUDA GPUs, `128` otherwise) |
| `--replay-ratio` | `1.5` | Controlled training samples per newly generated self-play position |
| `--eval-interval` | `5` | Run 20-game Arena evaluation & Elo update every $K$ iterations (`0` disables) |
| `--compile` | `off` | Enable `torch.compile` graph fusion on supported Linux/Windows setups |
| `--save-dir` | `checkpoints` | Directory for `.pth` models and replay buffer state |

---

## 5. Playing Interactively Against the AI

```bash
# Play as Black (uses checkpoint's native board size automatically)
python OceanGo.py --play

# Play as White against a specific checkpoint on any board size (e.g., 19x19)
python OceanGo.py --play --color white --board-size 19 --play-symmetries 8
```

### Interactive Controls & Features
- **Left Click**: Place a stone on any legal intersection.
- **`P` Key / Pass Button**: Pass your turn (two consecutive passes end the game).
- **`O` Key (Live Ownership Overlay)**: Toggles live neural territory ownership predictions from the V2 Auxiliary Ownership Head right on the board!
- **`E` Key / Count Button**: Score the board immediately under `7.5` Komi Chinese area rules.
- **Subtree Reuse & 8x Symmetry Ensemble**: During `--play`, the AI automatically reuses MCTS subtrees across both your moves and its own moves while ensembling all 8 dihedral symmetries (`--play-symmetries 8`) for maximum playing strength.
- **Post-Game Board Review**: Click **Review** (or `[X]`) after Game Over and click any stone group to toggle **Alive / Dead** with instant territory square updates; press **`R`** to play again or **`S`** to swap colors.

---

## 6. Inspecting Model Metadata & Parameters

To inspect checkpoint metadata (`version`, `blocks`, `channels`, `board_size`, `iteration`, `elo_est`) and verify cross-board (`9x9`, `13x13`, `19x19`) forward shapes:
```bash
python policy_print.py
```
