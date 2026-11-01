# OceanGo

**OceanGo** is a research project designed to recreate the core architecture of AlphaGo and AlphaZero on a smaller, consumer-accessible scale. 

## Academic Context
This project was developed as a **Final Year Research Project** for the **Bachelor of Computer Science** degree at the **University of Windsor**. 

It was created during a pivotal time in technology—the rapid rise of Artificial Intelligence and Large Language Models (LLMs). The primary motivation for this research is to investigate and build a profound, hands-on understanding of foundational AI methodologies beyond language models, specifically **Monte Carlo Tree Search (MCTS)** and **Policy-Value Neural Networks**.

## Project Intent
The overarching goal of OceanGo is to implement an AlphaGo-like algorithm capable of playing the ancient board game Go, configured to run on accessible hardware. 

While DeepMind's Alpha systems required massive clusters of custom TPUs to train, OceanGo establishes a framework scaled to train and execute on a **single computer with a single CPU and GPU**. It proves that the mathematical principles and architectures of AlphaZero can be structured and understood within the scope of a bachelor's degree timeline using individual consumer hardware.

## Key Technical Features
The theoretical backbone of this engine replicates modern reinforcement learning techniques:
* **Deep Residual Network (ResNet)**: A deep convolutional neural network utilizing residual blocks to evaluate board states, outputting both move probabilities (Policy Head) and win/loss predictions (Value Head).
* **State History & Input Planes**: Feeds the neural network the last 8 board states along with player perspective channels, granting the AI an understanding of "momentum" and complex situational tactics like the Ko rule.
* **Monte Carlo Tree Search (MCTS)**: Advanced forecasting of possible move outcomes utilizing PUCT (Predictor Upper Confidence Bounds). By default, it runs exactly **800 simulations per move**, mirroring the DeepMind paper.
* **Batched/Parallel MCTS with Virtual Loss**: Enables concurrent evaluations of unique tree search paths.
* **Centralized GPU Inference Queue**: Uses an asynchronous client/server multiprocessing architecture. CPU MCTS workers send board states to a shared queue, while a single dedicated GPU thread evaluates massive batches concurrently. This bypasses the Python GIL and prevents GPU VRAM bottlenecks or OOM crashes while keeping hardware utilization at maximum.
* **Strict AlphaGo Zero Optimization**: Uses **Stochastic Gradient Descent (SGD) with momentum (0.9)** and **L2 weight decay**, aligning perfectly with the mathematical specifications of the original paper. The network uses exactly 19 residual blocks, 256 filters, and a 256-unit value head.
* **Self-Play Enhancements**: Utilizes **Dirichlet Noise** for divergent baseline exploration, **Temperature Decay** for shifting from exploration to exploitation, and **Symmetrical Data Augmentation** (rotations and reflections) to dynamically multiply the training experience.
* **Interactive Pygame UI**: Features a classic wooden Go board aesthetic, last-move highlight indicators, and manual turn passing (via the `P` key) for human vs. AI matches.
* **Rigorous Go Environment**: Fully implemented underlying rules engine dictating liberties, chained captures, positional superko, pass detection, and accurate Chinese area scoring incorporating Komi.

## Cross-Platform Hardware Scaling
OceanGo runs on both **macOS** and **Windows 11**:
* **macOS (Apple Silicon M1/M2/M3/M4)**: Accelerates neural network inference and SGD backpropagation using PyTorch's Metal Performance Shaders (MPS) backend.
* **Windows 11 (NVIDIA GPU)**: Supports NVIDIA CUDA acceleration (e.g. RTX 30/40/50 series) with FP32 Tensor Core optimizations.
* **CPU Mode**: Gracefully falls back to CPU if no discrete or integrated accelerator is detected.

By default, the environment is configured to train on a **9x9 board**, which shrinks the search space enough to allow a single consumer computer to achieve robust tactical play within just a few hours of self-play training. The architecture also natively supports 13x13 or standard 19x19 board dimensions.

## Quick Start / Setup

### 1. Environment Setup

#### On macOS
```bash
# 1. Create and activate venv (Python 3.8 – 3.14+ supported)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (PyTorch includes Apple Silicon MPS support by default)
pip install -r requirements.txt

# 3. Test hardware accelerator
python t.py
```

#### On Windows 11
```cmd
rem 1. Create and activate venv
python -m venv .venv
.venv\Scripts\activate

rem 2. Install PyTorch with CUDA 12.1 (for NVIDIA GPUs) and dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

rem 3. Test hardware accelerator
python t.py
```
*(Automated setup scripts are also included: run `./setup_venv.sh` on macOS or `setup_venv.bat` on Windows 11).*

### 2. Playing Against the AI
To immediately play a game on the interactive Pygame board against the latest model:
```bash
python OceanGo.py --play
```
*(Note: `python GPU.py --play` also works as a backwards-compatible alias).*
- Click on board intersections to place stones.
- Press **`P`** on your keyboard to pass your turn.
- To play against a specific historical snapshot:
  `python OceanGo.py --model checkpoints/policy_value_net_iter_3.pth --play`

### 3. Training the AI
To start the AlphaZero self-play and training loop from scratch (or continue from an existing `policy_value_net.pth` model on disk):
```bash
python OceanGo.py
```
This command spins up parallel processes to generate self-play data, evaluates positions, and begins training the ResNet.

---
*Developed for the University of Windsor Computer Science Program.*