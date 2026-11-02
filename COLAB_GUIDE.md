# Google Colab Training Guide for OceanGo

This guide walks you through setting up, running, and monitoring **OceanGo** AlphaZero self-play training on **Google Colab** using your Google AI Pro monthly compute units.

---

## 1. Quick Overview

| Feature | Details |
|:---|:---|
| **Primary File** | [`OceanGo_Colab.ipynb`](OceanGo_Colab.ipynb) (included in the repository) |
| **Hardware** | NVIDIA Cloud GPU with CUDA & Tensor Cores |
| **Recommended GPU** | **T4 GPU** (~1.5 to 1.9 Compute Units / hour — cheapest tier) |
| **Checkpoints** | Automatically sync directly to your personal **Google Drive** |
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

### Step 3: Configure the GPU & Background Execution
1. In the top menu, click **Runtime** $\to$ **Change runtime type**.
2. Set the following options:
   - **Hardware accelerator**: Choose **GPU**.
   - **GPU type**: Select **T4** (recommended for lowest credit burn).
   - **Runtime shape**: Select **Standard**.
   - **Background execution**: Check the box **`☑ Run in background`** *(allows Colab to keep training after you close the tab or turn off your computer)*.
3. Click **Save**.

> [!TIP]
> **Cheapest Credit Burn**: Choosing the **T4 GPU** consumes only **~1.5 to 1.9 Compute Units per hour**. Your 200 monthly credits give you **110 to 130 hours** of GPU training!

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

#### Cell 5: Launch Training
Starts the self-play reinforcement learning loop:
```bash
!python OceanGo.py --train \
    --save-dir /content/drive/MyDrive/OceanGo/checkpoints \
    --games 200 \
    --sims 400 \
    --iters 50
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

## 6. Credit Optimization Cheat Sheet

| Strategy | Recommended Settings | Cost (CUs/hr) | Hours per 200 CUs | Iterations Achieved |
|:---|:---|:---:|:---:|:---:|
| **Maximum Budget (Cheapest)** | **T4 GPU** + `--games 200 --sims 400` | ~1.8 CUs/hr | **~120 Hours** | **~30 to 40 Iterations** |
| **Balanced** | **V100 GPU** + `--games 200 --sims 400` | ~5.0 CUs/hr | **~40 Hours** | **~25 to 30 Iterations** |
| **Ultra Speed** | **A100 GPU** + `--games 500 --sims 800` | ~12.5 CUs/hr | **~16 Hours** | **~3 to 4 Deep Iterations** |

> [!IMPORTANT]
> **Avoid Idle Credit Drain**: When your training run is completely finished, click **Runtime** $\to$ **Disconnect and delete runtime** so that your session stops drawing compute units.
