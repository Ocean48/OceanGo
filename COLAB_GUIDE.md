# Google Colab Training Guide for OceanGo

This guide walks you through setting up, running, and monitoring **OceanGo** AlphaZero self-play training on **Google Colab** using your Google AI Pro monthly compute units.

---

## 1. Quick Overview

| Feature | Details |
|:---|:---|
| **Primary File** | [`OceanGo_Colab.ipynb`](OceanGo_Colab.ipynb) (included in the repository) |
| **Hardware** | Google Cloud **TPU `v5e-1`** (44 vCPUs) or NVIDIA **T4 GPU + High-RAM** (8 vCPUs) |
| **Recommended Runtime** | **TPU `v5e-1`** (~2.0 Compute Units/hr, **44 vCPUs** for 42 parallel MCTS workers) |
| **Checkpoints** | Automatically sync directly to your personal **Google Drive** (portable across TPU, Windows GPU, and Mac M5) |
| **Background Execution** | **Yes** — you can close the browser tab or shut down your Mac |

---

## 2. Step-by-Step Setup from Scratch

### Step 1: Open Google Colab
1. In your browser (Chrome or Safari), go to: **[colab.research.google.com](https://colab.research.google.com)**
2. Make sure you are signed in with the Google account that has **Google AI Pro / Colab Pro**.

---

### Step 2: Upload the Notebook
1. In the Colab welcome popup (or under **File** menu), click the **Upload** tab.
2. Drag and drop the file **`OceanGo_Colab.ipynb`** from your local `OceanGo` project directory (or click **Browse** and select it).

---

### Step 3: Configure the Accelerator (`TPU v5e-1` or `T4 GPU + High-RAM`)
1. In the top menu, click **Runtime** $\to$ **Change runtime type**.
2. Choose one of the two recommended hardware setups:
   - **Option A — Fastest Self-Play (Recommended)**: Select **TPU `v5e-1`**. *(Gives you **44 vCPUs** so 42 MCTS games run simultaneously alongside `bfloat16` TPU inference at ~2.0 Compute Units/hr!)*
   - **Option B — Best NVIDIA GPU Setup**: Select **T4 GPU** and toggle **High-RAM ON** *(upgrades your GPU VM from 2 vCPUs to 8 vCPUs at ~2.3 Compute Units/hr)*.
3. Click **Save**.

> [!TIP]
> **Why `TPU v5e-1` is so fast for AlphaZero**: Self-play MCTS spends most of its time traversing game trees and checking Go board rules on CPU cores. **`TPU v5e-1` provides 44 vCPUs** (5.5x more CPU cores than `T4 + High-RAM`), and `OceanGo.py` pads neural network batches into fixed buckets (`16, 32, 64, 128, 256, 512`) so the TPU evaluates all 42 CPU workers without recompiling!

---

### Step 4: Run the Notebook Cells

You can run the notebook by clicking **Runtime** $\to$ **Run all** (or press `Cmd + F9` / `Ctrl + F9`), or run each cell one-by-one by clicking the play button `[ ▶ ]` next to each section:

#### Cell 1: Hardware Check
Runs `!nvidia-smi` to confirm that the NVIDIA GPU driver is detected and active.

#### Cell 2: Connect Google Drive
Mounts your personal Google Drive and creates the checkpoint folder:
```python
from google.colab import drive
drive.mount('/content/drive')
```
*A Google popup will appear asking: **"Allow this notebook to access your Google Drive files?"** $\to$ Click **Connect to Google Drive**.*

#### Cell 3: Clone Repository & Install Dependencies
Clones the OceanGo repository and installs dependencies (`torch`, `pygame-ce`, `numpy`).

#### Cell 4: 10-Second Dry-Run Test (Zero Risk)
```bash
!python OceanGo.py --dry-run --save-dir /content/drive/MyDrive/OceanGo/checkpoints
```
* **Takes only 10 seconds** and costs virtually 0 compute units.
* Verifies that CUDA inference, multiprocessing queues, and Google Drive write permissions work properly before spending credits.
* Confirm that you see: `[Dry-Run] SUCCESS: Everything is functional and verified ready for full training!`

#### Cell 5: Launch Training (with Auto-Disconnect)
Starts the self-play reinforcement learning loop and automatically disconnects the GPU runtime as soon as training finishes:
```python
!python OceanGo.py --train \
    --save-dir /content/drive/MyDrive/OceanGo/checkpoints \
    --games 200 \
    --sims 400 \
    --iters 50

from google.colab import runtime
print("Training finished! Automatically disconnecting runtime to save Compute Units...")
runtime.unassign()
```

---

## 3. How to Monitor Training

You do not need to keep your laptop open or stare at the screen. You can monitor progress across several channels:

### 1. In the Colab Browser Tab
The training displays a live single-line progress bar:
```text
Self-Play: [██████████████░░░░░░░░░░░░░░] 100/200 ( 50.0%) | 0.8 games/s | Elapsed: 00:45:10 | ETA: 00:45:10
```

### 2. On Your Phone (Anywhere)
Open **Safari** or **Chrome** on your phone $\to$ go to [colab.research.google.com](https://colab.research.google.com) $\to$ tap `OceanGo_Colab.ipynb`. You can check the live progress bar and speed while on the move.

### 3. In Google Drive
Open **Google Drive** in your browser or phone app and look at:
```text
MyDrive/OceanGo/checkpoints/
```
As each iteration finishes, you will see new files appear automatically:
- `policy_value_net.pth` (the latest trained model)
- `policy_value_net_iter_1.pth` (Iteration 1 snapshot)
- `policy_value_net_iter_2.pth` (Iteration 2 snapshot)
- `logs_YYYYMMDD_HHMMSS.log` (detailed text log file)

### 4. Tracking Compute Unit Balance
Click the **RAM / Disk** meter in the top-right corner of Colab:
- Shows your remaining compute units (e.g. `194.2 / 200 units remaining`).
- Shows the current hourly burn rate.
- Shows active sessions so you can stop instances when not in use.

---

## 4. Closing Your Laptop / Leaving the Tab

Because you enabled **Background execution** on Colab Pro:
1. You can close the browser tab.
2. You can shut down your MacBook or close the lid.
3. The training will continue running on Google’s cloud hardware.
4. When you want to check back, just reopen the notebook link in your browser.

---

## 5. Playing Against Your Trained Model on Your Mac

Once Colab finishes an iteration and saves `policy_value_net.pth` to your Google Drive:

1. **Download the model**:
   - Go to Google Drive $\to$ `OceanGo/checkpoints/`.
   - Download `policy_value_net.pth` (or any `policy_value_net_iter_X.pth`) to your MacBook.
2. **Move it into your local project folder**:
   - Place `policy_value_net.pth` in `/Users/sichaochen/Documents/Projects/OceanGo/`.
3. **Launch the interactive Pygame window**:
   ```bash
   python OceanGo.py --play
   ```
   Or to play against a specific historical iteration:
   ```bash
   python OceanGo.py --model checkpoints/policy_value_net_iter_5.pth --play
   ```

---

## 6. Runtime & Credit Optimization Cheat Sheet (200 Monthly CUs)

| Runtime Setup (`Change runtime type`) | Command Preset | Cost (CUs/hr) | Time per Loop | 50 Loops Cost & Time | Max Loops per 200 CUs |
|:---|:---|:---:|:---:|:---:|:---:|
| 🚀 **TPU `v5e-1`** *(Fastest Self-Play — **44 vCPUs**)* | `--games 200 --sims 400 --iters 50` | **~2.0 CUs/hr** | **~3–5 mins** | **~7–8 CUs (~3.5 hrs)** | **~1,200+ Loops** |
| 🏆 **T4 GPU + High-RAM ON** *(Best GPU Value — 8 vCPUs)* | `--games 200 --sims 400 --iters 50` | **~2.3 CUs/hr** | **~8–10 mins** | **~18 CUs (~8 hrs)** | **~550+ Loops** |
| **TPU `v5e-1`** *(DeepMind Heavy Preset — **44 vCPUs**)* | `--games 500 --sims 800 --iters 50` | **~2.0 CUs/hr** | **~12–16 mins** | **~24 CUs (~12 hrs)** | **~400 Loops** |
| **T4 GPU + High-RAM ON** *(DeepMind Heavy Preset)* | `--games 500 --sims 800 --iters 50` | **~2.3 CUs/hr** | ~35–45 mins | ~75 CUs (~32 hrs) | ~130 Loops |
| **L4 GPU (High-RAM OFF)** *(8 vCPUs, 2x cost)* | `--games 200 --sims 400 --iters 50` | ~4.5 CUs/hr | ~6–8 mins | ~30 CUs (~6.5 hrs) | ~330 Loops |

> [!IMPORTANT]
> **Zero Idle Credit Drain**: Cell 5 automatically calls `runtime.unassign()` as soon as training finishes, immediately shutting down the Colab GPU session so no credits are wasted while you are away.
