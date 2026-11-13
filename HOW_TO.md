# How to Run OceanGo V2

OceanGo V2 is a self-contained, single-machine, **board-size-agnostic AlphaZero/KataGo Go engine** supporting progressive curriculum training (`9x9 -> 13x13 -> 19x19`) alongside an interactive Pygame UI. It profiles CPU threads, system RAM, and GPU VRAM automatically at startup across **Windows 11 / Linux (NVIDIA CUDA)** and **macOS (Apple Silicon MPS)**.

> For a full mathematical breakdown of the Global-Pooling ResNet, incremental Zobrist/bitmask Go engine, and MCTS optimizations, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## 1. Prerequisites & Environment Setup

- **Python Version**: Python 3.9 – 3.14+ supported (including Python 3.14).
- **Hardware Acceleration**:
  - **Windows 11 / Linux (NVIDIA GPU)**: Uses native CUDA `bfloat16` Tensor Cores (`channels_last` on Ampere/Ada/Hopper; `contiguous` `NCHW` automatically on consumer Blackwell `sm_120` for cuDNN stability). **Important for RTX 5060 Ti / RTX 5090 (Blackwell `sm_120`)**: use PyTorch CUDA 12.8 (`cu128`) wheels (installed automatically by `setup_venv.bat` / `setup_venv.sh`).
  - **macOS (Apple Silicon M1/M2/M3/M4)**: Uses Metal Performance Shaders (`MPS`) `bfloat16` acceleration.
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

## 5. Optimal Hyperparameters for Maximum Strength Across All 3 Board Sizes (`9x9 -> 13x13 -> 19x19`)

Because Stage 2 (`13x13`) and Stage 3 (`19x19`) automatically inherit the exact trunk depth (`--blocks`) and filter width (`--channels`) saved in Stage 1 (`9x9`), choosing your backbone capacity at **Stage 1** determines the ceiling of your final `19x19` model.

### A. Best-Performance 3-Stage Curriculum Commands

#### Preset 1: Flagship Maximum Strength (`18 blocks × 256 channels`, ~22M params)
Recommended for **RTX 5060 Ti (16GB)**, **RTX 5090 (32GB)**, **RTX 4070/4080/4090**, or **Cloud L4 / A100 / TPU `v5e-1`**. Pairs AlphaZero's `256`-channel trunk width with KataGo Global-Pooling and deep MCTS search across all three board sizes:

```bash
# Stage 1 (9x9 — Tactical Reading, Captures, Ko & Life-and-Death | Global Iters 1 -> 40)
python OceanGo.py --train --board-size 9 --blocks 18 --channels 256 --games 600 --sims 800 --iters 40 --lr 0.01 --replay-ratio 1.5 --eval-interval 5 --eval-games 20

# Stage 2 (13x13 — Joseki, Ladders & Whole-Board Influence | Global Iters 41 -> 85)
python OceanGo.py --train --board-size 13 --games 500 --sims 800 --iters 45 --lr 0.01 --replay-ratio 1.5 --eval-interval 5 --eval-games 20

# Stage 3 (19x19 — Full-Board Fuseki, Fighting & Endgame Yose | Global Iters 86 -> 145)
python OceanGo.py --train --board-size 19 --games 300 --sims 1000 --iters 60 --lr 0.005 --replay-ratio 1.8 --eval-interval 5 --eval-games 20
```

*(Tip: Instead of running the three stage commands manually one by one, use `./train_curriculum.sh` on Linux/macOS or `.\train_curriculum.bat` on Windows 11 — see **[Section 6](#6-automated-3-stage-curriculum-scripts-linux-windows-11--macos)** and **[Section 7 (`--compile` Guide)](#7-using---compile-torchcompile-graph-fusion)** below).*

#### Preset 2: Balanced High-Efficiency (`15 blocks × 192 channels`, ~10.8M params)
Recommended for **Apple Silicon Macs (`MPS`)**, **8–12 GB VRAM GPUs**, or faster turnaround:

```bash
# Stage 1 (9x9 | Global Iters 1 -> 30)
python OceanGo.py --train --board-size 9 --blocks 15 --channels 192 --games 500 --sims 600 --iters 30 --lr 0.01

# Stage 2 (13x13 | Global Iters 31 -> 65)
python OceanGo.py --train --board-size 13 --games 400 --sims 600 --iters 35 --lr 0.01

# Stage 3 (19x19 | Global Iters 66 -> 115)
python OceanGo.py --train --board-size 19 --games 250 --sims 800 --iters 50 --lr 0.005 --replay-ratio 1.8
```

### B. Why These Stage Parameters Maximize Final Strength

| Stage | Board | `--iters` | `--lr` Schedule Behavior (`stage_iter`) | Strategic Purpose |
|:---:|:---:|:---:|:---|:---|
| **Stage 1** | `9x9` | `40` | `0.01` (iters `1–25`) -> `0.003` (iters `26–40`) | Fast short games rapidly teach exact liberty counting, eye formation, ko fights, and dead-stone ownership before dropping LR (`0.3x`) to lock in clean local filters. |
| **Stage 2** | `13x13` | `45` | 2-iter warmup (`0.0025` -> `0.006`) -> `0.01` (`3–25`) -> `0.003` (`26–45`) | Gently adapts BatchNorm and GlobalPool scaling $s(N)$ without shock, then learns corner-to-side interactions, ladders, and territory vs. influence balance. |
| **Stage 3** | `19x19` | `60` | 2-iter warmup (`0.00125` -> `0.003`) -> `0.005` (`3–25`) -> `0.0015` (`26–45`) -> `0.0005` (`46–60`) | Lower base `--lr 0.005` preserves pre-trained tactical filters while spanning **all three** LR decay tiers down to `0.0005` (`0.1x`) with deeper `--sims 1000` reading and `1.8x` sample reuse. |

---

### C. Model Save Staging & Directory Layout During Training

All three curriculum stages run in the **same `checkpoints/` directory** without ever overwriting earlier iteration history or mixing incompatible board-size buffers:

```text
OceanGo/
├── policy_value_net.pth                        # Always mirrors the latest completed iteration
├── logs_YYYYMMDD_HHMMSS.log                    # Detailed per-run timestamped training log
└── checkpoints/
    ├── policy_value_net.pth                    # Latest V2 checkpoint (auto-detected on resume / --play)
    ├── policy_value_net_iter_1.pth             # ──┐ Stage 1 (9x9) snapshots: iter 1 .. 40
    ├── ...                                     #   │
    ├── policy_value_net_iter_40.pth            # ──┘
    ├── policy_value_net_best_9x9.pth           # Peak Arena-promoted 9x9 champion model
    ├── train_state_v2_9x9.pth                  # 9x9 SGD momentum + compact uint16 replay buffer
    │
    ├── policy_value_net_iter_41.pth            # ──┐ Stage 2 (13x13) snapshots: iter 41 .. 85
    ├── ...                                     #   │ (100% weight transfer from iter_40)
    ├── policy_value_net_iter_85.pth            # ──┘
    ├── policy_value_net_best_13x13.pth         # Peak Arena-promoted 13x13 champion model
    ├── train_state_v2_13x13.pth                # 13x13 SGD momentum + compact uint16 replay buffer
    │
    ├── policy_value_net_iter_86.pth            # ──┐ Stage 3 (19x19) snapshots: iter 86 .. 145
    ├── ...                                     #   │ (100% weight transfer from iter_85)
    ├── policy_value_net_iter_145.pth           # ──┘
    ├── policy_value_net_best_19x19.pth         # Peak Arena-promoted 19x19 champion model
    └── train_state_v2_19x19.pth                # 19x19 SGD momentum + compact uint16 replay buffer
```

#### Key Staging Behaviors to Know:
1. **Continuous Global `iter_X` Numbering vs. Reset `Stage Iter`**:
   - `Iteration` (`policy_value_net_iter_X.pth`) increments monotonically (`1 -> 40 -> 85 -> 145`) so every historical checkpoint across your entire curriculum is preserved.
   - `Stage Iter` resets to `1` whenever `--board-size` changes so each new board size gets its full `--iters` count, a 2-iteration warmup, and a fresh LR schedule.
2. **Per-Board Champion Models (`policy_value_net_best_{N}x{N}.pth`)**:
   - Every `--eval-interval 5` iterations, the candidate model plays `20` paired games (`10` Black + `10` White from varied 2-move openings) against the current reference model.
   - Whenever the candidate achieves **`>= 52%` win rate**, `[PROMOTED NEW REFERENCE]` is printed and `policy_value_net_best_{N}x{N}.pth` is updated. After finishing all 3 stages, you can play against your final universal model *or* load each board's dedicated champion (`--model checkpoints/policy_value_net_best_9x9.pth`).
3. **Safe Interrupt & Resume (`Ctrl+C`)**:
   - Stopping mid-stage (e.g., during Iteration `53` on `13x13`) is completely safe—all completed iterations up to `iter_52.pth` and `train_state_v2_13x13.pth` are already on disk. Re-running the Stage 2 command resumes cleanly at `Iteration 53/85` with full replay buffer and SGD momentum intact.

---

### D. What to Expect to See in the Terminal

#### 1. Startup & Curriculum Stage Transition
When launching Stage 2 (`--board-size 13`) after Stage 1 (`9x9`), expect to see:
```text
========================================
OceanGo V2 Initializing...
OS Platform: Windows (AMD64)
Computing Device: CUDA | NVIDIA GeForce RTX 5060 Ti (16.0 GB VRAM, sm_120)
Precision & Layout: bfloat16 | contiguous
System Memory: 32.0 GB RAM | Replay Cap: 500,000 states
CPU Threads: 16 available | Using 14 self-play workers
========================================

[Curriculum Transfer] Loaded 100% of weights from 9x9 checkpoint 'checkpoints/policy_value_net_iter_40.pth' into 13x13 (18 blocks x 256 channels)!
[Curriculum Transition] Initialized fresh 13x13 replay buffer, reset SGD momentum, and scheduled 2-iteration LR warmup.
```
Notice on the first two transition iterations (`Stage Iter: 1` and `Stage Iter: 2`), `LR` automatically warms up (`0.25x` then `0.60x` of base `--lr`) before stepping up to full `--lr` on `Stage Iter: 3`.

#### 2. Per-Iteration Self-Play, Training & Arena Output
```text
==== Iteration 45/85 (Board: 13x13 | Stage Iter: 5 | LR: 0.01000 | Elo: 1245) ====
  Running 500 self-play games across 14 shared-memory workers (Full: 800 sims, Fast: 160 sims)...
  Self-Play: [████████████████████████████] 500/500 (100.0%) | 1.420 games/s | Elapsed: 00:05:52 | ETA: 00:00:00
  Self-Play Telemetry: +58,410 unique states (Buffer: 284,120) | Avg Len: 116.8 | Resigned: 312/500 | False-Resign: 1.8% (2/111) | Search Entropy: 1.64
  Training on 87,808 GPU-augmented samples (343 steps, ratio=1.50x new states)...
  Train [████████████████████] 343/343 (100.0%) | Loss: 2.841 (P:1.982 V:0.541 O:0.418 H:1.89) | 18.4 step/s
  Running Arena Evaluation (20 paired games @ 160 sims vs reference model)...
  Arena Result (9.4s): 13W-7L-0D (Win Rate: 65.0%, ΔElo: +108 -> Est. Elo: 1353) [PROMOTED NEW REFERENCE]
  Saved V2 checkpoint -> checkpoints/policy_value_net_iter_45.pth (Duration: 381.2s)
```

#### 3. How to Read the Health Metrics
- **`P` (Policy Cross-Entropy Loss)** & **`H` / `Search Entropy`**:
  - Starts near $\ln(N^2 + 1)$ (`~4.4` on `9x9`, `~5.1` on `13x13`, `~5.9` on `19x19`) on Iteration 1, then drops steadily into the `1.4 – 2.6` range as MCTS and the Policy Head focus on strong candidate moves. Note a temporary bump in `P` right when transitioning to a larger board (`9 -> 13 -> 19`) is completely normal because the action space expands (`82 -> 170 -> 362`).
- **`V` (Value `BCEWithLogits` Loss)**:
  - Random/even guessing corresponds to $\ln(2) \approx 0.693$. Healthy training sits around **`0.45 – 0.66`** (never dropping to `0.00`, because evenly matched self-play games remain competitive!). Unlike V1's MSE loss, `BCEWithLogits` stays gradient-active even in decisive positions.
- **`O` (Ownership `BCEWithLogits` Loss)**:
  - Starts near `~0.69` and steadily falls toward **`0.20 – 0.40`** as the network learns to classify live groups, dead stones, and territory boundaries accurately.
- **`Resigned` & `False-Resign %`**:
  - On Iteration 1–2, `Resigned` will be near `0` while the value head is uncertain; as value confidence grows, up to `~50–70%` of games resign early (saving time), while `25%` always play to double-pass. Healthy **`False-Resign` stays `< 5%`**.
- **`Est. Elo`**:
  - Starts at `1000` and accumulates Arena $\Delta\text{Elo}$ gains across all three stages (`9x9 -> 13x13 -> 19x19`). Occasional non-promoted Arena checks (`Win Rate < 52%`) are normal statistical variance over 20 games; `ref_net` holds the strongest milestone until beaten.

---

## 6. Automated 3-Stage Curriculum Scripts (Linux, Windows 11 & macOS)

Instead of manually waiting for Stage 1 (`9x9`) to finish before starting Stage 2 (`13x13`) and Stage 3 (`19x19`), OceanGo V2 includes automated, resume-aware pipeline scripts for all three operating systems:

- **`train_curriculum.sh`** — for **Linux** and **macOS**
- **`train_curriculum.bat`** — for **Windows 11** (`CMD` and `PowerShell`)

### A. What the Automated Scripts Handle for You
1. **Sequential Stage Chaining (`9x9 -> 13x13 -> 19x19`)**: Automatically launches the next stage immediately after the current stage finishes cleanly (`exit 0`), and halts immediately (`set -euo pipefail` / `!ERRORLEVEL!`) if an error or `Ctrl+C` occurs.
2. **Smart Crash / `Ctrl+C` Resume**: Before each stage, the script inspects the latest healthy V2 checkpoint in `checkpoints/`. If you stop mid-run (e.g., at Iteration `58` on `13x13`) and re-run the script later, it **automatically skips Stage 1 (`9x9`)**, finishes the remaining `27` iterations of Stage 2 (`59 -> 85`) with full replay buffer + SGD momentum intact, and then continues into Stage 3 (`86 -> 145`). *(Plain `&&` one-liners do not skip completed stages if restarted from the top).*
3. **Auto-Activates `.venv`** and defaults unconditionally to **Flagship (`18×256`, `40/45/60` iters) + `--compile` enabled** across both `./train_curriculum.sh` and `.\train_curriculum.bat` (while `OceanGo.py` automatically handles Triton vs. fast eager `bfloat16` per GPU backend):

| Platform | Script | Default `--preset` | Default `--compile` Flag | Runtime Compiler Behavior | Recommended Sleep / SSH Wrapper |
|:---|:---|:---|:---|:---|:---|
| **Linux (NVIDIA CUDA)** | `./train_curriculum.sh` | `flagship` (`18×256`, `40/45/60` iters) | **Enabled** (`--compile`) | Full `Triton` graph fusion | `tmux` or `nohup ... &` |
| **Windows 11 (NVIDIA CUDA)** | `.\train_curriculum.bat` | `flagship` (`18×256`, `40/45/60` iters) | **Enabled** (`--compile`) | Uses `triton-windows` + auto-detected MSVC `cl.exe` if installed, else auto-falls back to eager `bfloat16` | Windows *Power & sleep → Never* |
| **macOS (Apple Silicon `MPS`)** | `./train_curriculum.sh` | `flagship` (`18×256`, `40/45/60` iters) | **Enabled** (`--compile`) | `OceanGo.py` auto-selects faster eager `MPS` `bfloat16` | `caffeinate -dims ./train_curriculum.sh` |

---

### B. Running Unattended on Each OS

*(Tip: If starting a brand-new curriculum after an earlier experiment, move or delete old `.pth` files in `checkpoints/` first—or pass `--save-dir checkpoints_run2`—so it starts cleanly at Iteration 1).*

#### 1. Linux (Local Terminal or Remote SSH Server)
```bash
chmod +x train_curriculum.sh

# Option A (Recommended): Run inside tmux so you can detach/reattach anytime
tmux new -s oceango
./train_curriculum.sh 2>&1 | tee -a curriculum_console.log
# Detach: press Ctrl+B, then D
# Reattach later: tmux attach -t oceango

# Option B: Run in background with nohup (survives SSH disconnect)
nohup ./train_curriculum.sh > curriculum_console.log 2>&1 &
tail -f curriculum_console.log
```

#### 2. Windows 11 (Command Prompt or PowerShell)
Set **Settings → System → Power & sleep → Put my device to sleep → Never**, then run:
```bat
:: Flagship 18x256 curriculum + --compile (default)
.\train_curriculum.bat

:: Balanced 15x192 curriculum (8-12 GB VRAM GPUs)
.\train_curriculum.bat --preset balanced

:: Custom checkpoint directory or worker count (any extra flags forward directly to OceanGo.py)
.\train_curriculum.bat --save-dir checkpoints_v2 --workers 20
```

#### 3. macOS (Apple Silicon M1/M2/M3/M4)
Wrap with macOS's built-in `caffeinate -dims` so your Mac never enters idle sleep during multi-hour self-play:
```bash
chmod +x train_curriculum.sh

# Flagship 18x256 curriculum (default)
caffeinate -dims ./train_curriculum.sh 2>&1 | tee -a curriculum_console.log

# Or Balanced 15x192 curriculum for faster turnaround on base M1/M2/M3/M4 Macs:
caffeinate -dims ./train_curriculum.sh --preset balanced 2>&1 | tee -a curriculum_console.log
```

#### 4. Manual Command Chaining Without a Script (`&&`)
If you ever want to chain custom commands directly in a single uninterrupted session:
- **Linux / macOS (`bash` / `zsh`) or Windows `CMD` / `PowerShell 7+`**: join commands with `&&` so the next stage only starts if the previous stage succeeded (`exit 0`):
  ```bash
  python -u OceanGo.py --train --compile --board-size 9 --blocks 18 --channels 256 --games 600 --sims 800 --iters 40 --lr 0.01 --replay-ratio 1.5 --eval-interval 5 --eval-games 20 && \
  python -u OceanGo.py --train --compile --board-size 13 --games 500 --sims 800 --iters 45 --lr 0.01 --replay-ratio 1.5 --eval-interval 5 --eval-games 20 && \
  python -u OceanGo.py --train --compile --board-size 19 --games 300 --sims 1000 --iters 60 --lr 0.005 --replay-ratio 1.8 --eval-interval 5 --eval-games 20
  ```
  *(Note: Avoid chaining with `;` in bash or Windows PowerShell 5.1, because `;` launches Stage 2 even if Stage 1 failed or was stopped with `Ctrl+C`).*

---

## 7. Using `--compile` (`torch.compile` Graph Fusion)

Passing `--compile` enables PyTorch 2's **TorchInductor** compiler (`torch.compile`) around `PolicyValueNet`.

### A. What `--compile` Optimizes in OceanGo V2
- **Kernel Fusion**: Fuses `Conv2d + BatchNorm2d + ReLU`, `KataGlobalPool` spatial mean/max/scale reductions, and residual skip additions into fused Triton GPU kernels, cutting VRAM bandwidth overhead and boosting CUDA throughput by **~15–25%** across both MCTS batched inference and 8x-augmented training steps.
- **Dynamic MCTS Batch Warmup**: Because parallel CPU self-play workers submit variable batch sizes (`B = 1 .. 512`) every millisecond, OceanGo pre-warms `torch.compile` at `B = 2` and `B = 8` on startup and raises `torch._dynamo.config.recompile_limit` to `64` so a single dynamic-batch graph serves all worker batch sizes without recompilation stalls.
- **100% Portable Checkpoints & Safe Fallback**: OceanGo automatically strips the compiler's `_orig_mod` wrapper via `unwrap_model()` whenever saving `.pth` files and sets `torch._dynamo.config.suppress_errors = True`. A model trained on Linux with `--compile` loads identically on Windows, macOS, or `--play` without `--compile`.

---

### B. Platform-Specific `--compile` Setup & Behavior

#### 1. Linux with NVIDIA GPU (`RTX 30 / 40 / 5060 Ti / 5090`) — **Recommended**
PyTorch's Linux `cu128` wheels bundle **Triton** out of the box. You only need a standard C/C++ compiler and Python headers installed so Triton can JIT-compile its tiny CUDA launcher stub:

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y build-essential python3-dev

# Fedora / RHEL
sudo dnf install -y gcc gcc-c++ python3-devel

# Arch Linux
sudo pacman -S --needed base-devel python
```

**Verify `--compile` in 10 seconds before starting a full run:**
```bash
python OceanGo.py --dry-run --board-size 9 --compile
```
Look for this confirmation line at startup:
```text
[Compiler] torch.compile graph fusion active (dynamic batch + backward warmup verified).
```
> **First-Iteration Warmup Note**: Expect a one-time **15–40 second** pause on the very first self-play batch and first training step of each board size (`9x9`, `13x13`, `19x19`) while Triton compiles and caches kernels under `/tmp/torchinductor_<user>`. Every subsequent iteration runs immediately at full compiled speed.

---

#### 2. Windows 11 with NVIDIA GPU — **Automatic Eager Fallback or Optional `triton-windows` + MSVC**
- **Out of the box (zero extra setup)**: If you run `python OceanGo.py --train --compile` (or `.\train_curriculum.bat`) without `triton-windows` or MSVC `cl.exe` installed, OceanGo detects this at startup and automatically runs at full native eager CUDA `bfloat16` Tensor Core speed without crashing.
- **Optional — Enable full `torch.compile` on Windows 11**:
  1. Install **[Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)** and check the **Desktop development with C++** workload (provides `cl.exe`).
  2. Install `triton-windows` in `.venv` (already bundled with PyTorch 2.11+ `cu128`, or install via `pip install -U triton-windows`).
  3. Run `.\train_curriculum.bat` directly from Command Prompt or PowerShell — it automatically locates and activates Visual Studio's x64 C++ Build Tools (`vcvars64.bat` via `vswhere.exe`) before launching training! *(If running `python OceanGo.py --compile` directly without `train_curriculum.bat`, launch from **x64 Native Tools Command Prompt for VS 2022** so `cl.exe` is in `PATH`).*

---

#### 3. macOS (Apple Silicon `MPS`) — **Automatic Eager `bfloat16` Selection**
Triton only targets CUDA/ROCm GPUs, not Apple Metal (`MPS`), and PyTorch's experimental Inductor `mps` backend runs ~4× slower than eager execution on dynamic MCTS batch sizes. When `--compile` is passed on macOS (such as the default in `./train_curriculum.sh`), `OceanGo.py` detects `MPS` at startup and automatically selects fast eager `MPS` `bfloat16` execution.

---

### C. When to Use `--compile` vs. Omit `--compile`
- **Use `--compile`**: Long training runs (`--train`) on **Linux CUDA** (or **Windows CUDA** with `triton-windows` + MSVC), and high-simulation `--play` (`--sims 1600+`) on Linux CUDA.
- **Omit `--compile` (or pass `--no-compile`)**: Quick `--dry-run` checks, short interactive `--play` sessions where you want instant startup with zero 15s JIT warmup, **macOS (`MPS`)**, and **Cloud TPUs** (`torch_xla` manages its own XLA graph compilation automatically).

---

## 8. Playing Interactively Against the AI

```bash
# Play against your strongest trained model (auto-loads latest healthy V2 checkpoint)
python OceanGo.py --play --sims 1200 --play-symmetries 8

# Play against a specific board's Arena-promoted champion checkpoint as White
python OceanGo.py --play --model checkpoints/policy_value_net_best_19x19.pth --color white --sims 1200 --play-symmetries 8
```

### Interactive Controls & Features
- **Left Click**: Place a stone on any legal intersection.
- **`P` Key / Pass Button**: Pass your turn (two consecutive passes end the game).
- **`O` Key (Live Ownership Overlay)**: Toggles live neural territory ownership predictions from the V2 Auxiliary Ownership Head right on the board!
- **`E` Key / Count Button**: Score the board immediately under `7.5` Komi Chinese area rules.
- **Subtree Reuse & 8x Symmetry Ensemble**: During `--play`, the AI automatically reuses MCTS subtrees across both your moves and its own moves while ensembling all 8 dihedral symmetries (`--play-symmetries 8`) for maximum playing strength.
- **Post-Game Board Review**: Click **Review** (or `[X]`) after Game Over and click any stone group to toggle **Alive / Dead** with instant territory square updates; press **`R`** to play again or **`S`** to swap colors.

---

## 9. Inspecting Model Health, Pass Priors & Parameters

Run `policy_print.py` anytime to scan every `.pth` file in `checkpoints/` for finite weight/BatchNorm health (`HEALTHY (finite)` vs `CORRUPTED (NaN/Inf)`), inspect checkpoint metadata (`version`, `blocks`, `channels`, `board_size`, `iteration`, `elo_est`), and check opening Move-0 pass priors (`P(Pass|Move0)`) across `9x9`, `13x13`, and `19x19`:

```bash
# Scan all checkpoints in checkpoints/ and inspect the latest healthy V2 model
python policy_print.py

# Inspect a specific checkpoint file or custom directory
python policy_print.py --model checkpoints/policy_value_net_best_19x19.pth
python policy_print.py --save-dir checkpoints_run2
```
