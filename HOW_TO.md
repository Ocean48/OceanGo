# How to Run OceanGo

OceanGo contains a full AlphaZero-style training pipeline alongside an interactive graphical Go game engine.

## Prerequisites
- **Python 3.8+**
- **Pygame**: To render the visual board.
  ```bash
  pip install pygame
  ```
- **NumPy**: Matrix calculus logic handling.
  ```bash
  pip install numpy
  ```
- **PyTorch**: *Highly recommended* to install with CUDA support to enable fast GPU offloading during parallel batched logic processing.
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
  *(Example only. Adjust based on your preferred CUDA distribution version)*

---

## Command Line Usage

OceanGo has been built flexibly so it can serve as an AI trainer or an interactive game. We trigger these behaviors using command-line arguments.

### 1. Training the AI
To start the training sequence from scratch (or continue from the latest `policy_value_net.pth` model on disk):
```bash
python GPU.py
```
This automatically initiates the loop:
1. Begins self-play across the `N_GAMES_PER_ITER` range.
2. Displays real-time progress via terminal print loops.
3. Evaluates all unique states asynchronously utilizing Virtual Loss scaling into the ResNet arrays.
4. Generates an entire data buffer alongside horizontal/rotating augmented data symmetries.
5. Saves the training progress into the `checkpoints/` folder.
6. Logs all deep interactions to `logs.log`.

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
- `BOARD_SIZE`: Adjusts Go board mapping (e.g., standard sizing runs at 13 or 19).
- `N_ITER`: Number of complete system evaluation iterations.
- `N_MCTS_SIMS`: Number of MCTS predictions tree depth loops (higher = smarter strategy but slower execution time bounds).