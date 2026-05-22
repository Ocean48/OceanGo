# How to Run OceanGo

OceanGo contains a full AlphaZero-style training pipeline alongside an interactive graphical Go game engine.

## Prerequisites
- **Python 3.8+**
- **CUDA-enabled GPU** (Highly recommended for practical training speeds)

Install the required dependencies via the included `requirements.txt` file:
```bash
pip install -r requirements.txt
```
*Note: If you want hardware-accelerated GPU support for PyTorch (which is strongly recommended to avoid training bottlenecks), ensure you install PyTorch with the correct CUDA distribution for your system, e.g.:*
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Command Line Usage

OceanGo has been built flexibly so it can serve as an AI trainer or an interactive game. We trigger these behaviors using command-line arguments.

### 1. Training the AI
To start the training sequence from scratch (or continue from the latest `policy_value_net.pth` model on disk):
```bash
python GPU.py
```
This automatically initiates the loop:
1. Begins self-play across the `N_GAMES_PER_ITER` range utilizing a multi-core Centralized GPU Queue architecture.
2. Displays real-time progress for both self-play game completions and live epoch/batch loss updates via terminal print loops.
3. Evaluates all unique states by routing CPU worker requests to a single dedicated GPU batch-inference thread, preventing Out-Of-Memory (OOM) crashes.
4. Generates an entire data buffer alongside horizontal/rotating augmented data symmetries.
5. Saves the training progress into the `checkpoints/` folder.
6. Logs all deep interactions to a time-stamped log file (e.g., `logs_YYYYMMDD_HHMMSS.log`).

### 2. Playing Against the AI
If you wish to bypass training constraints and play immediately against whichever latest model exists in the base repository folder:
```bash
python GPU.py --play
```

### 3. Loading specific checkpoint snapshots
During your research, you may prefer an earlier generation model (e.g. Iteration 3) to test progression instead of the final iteration override. Use the `--model` argument path:
```bash
python GPU.py --model checkpoints/policy_value_net_iter_3.pth --play
```

### 4. Customizing Scale parameters
To alter the scale for bigger computational limits (e.g. larger board, massive batch nodes) modify the top components inside `GPU.py`.
- `BOARD_SIZE`: Adjusts Go board mapping (e.g., 9x9 by default, standard sizing runs at 13 or 19).
- `N_ITER`: Number of complete system evaluation iterations (default 50).
- `N_MCTS_SIMS`: Number of MCTS predictions tree depth loops (higher = smarter strategy but slower execution time bounds).
- `TRAIN_SAMPLE_SIZE`: Number of board states to sample for neural network backpropagation (default 250,000).
- `MAX_REPLAY_BUFFER_SIZE`: Total size of the historical buffer used to prevent overfitting (default 500,000).