import atexit
import logging
import math
import random
import copy
import time
import sys
import os
import ctypes
import argparse
import concurrent.futures
import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import queue
import platform
from collections import deque
import numpy as np

# Suppress Pygame startup banner across spawned multiprocessing workers
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import pygame
except ImportError:
    pygame = None

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------------------
# Global Config & Hardware Auto-Profiler
# ----------------------------------------
OS_NAME = platform.system().lower()

# Default Go board dimension (supports 9, 13, 19, or any N via Global-Pooling architecture)
BOARD_SIZE = 9
GRID_SIZE = 40
SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE

IS_MAIN_PROCESS = (mp.current_process().name == "MainProcess")
IS_TPU = False
xm = None


def detect_system_ram_gb() -> float:
    """Detects total physical system RAM in GiB using stdlib only (macOS, Linux, Windows)."""
    try:
        if OS_NAME == "windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3)
        elif hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return (pages * page_size) / (1024 ** 3)
    except Exception:
        pass
    return 16.0


SYSTEM_RAM_GB = detect_system_ram_gb()
GPU_VRAM_GB = 0.0
CUDA_COMPUTE_CAP = (0, 0)
AMP_DTYPE = torch.float32
AMP_ENABLED = False
USE_GRAD_SCALER = False
USE_CHANNELS_LAST = False

if not IS_MAIN_PROCESS:
    # Keep worker subprocesses lightweight and prevent OpenMP thread contention
    torch.set_num_threads(1)
    device = torch.device("cpu")
    device_name = f"CPU Worker ({platform.machine()})"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    props = torch.cuda.get_device_properties(0)
    GPU_VRAM_GB = props.total_memory / (1024 ** 3)
    CUDA_COMPUTE_CAP = (props.major, props.minor)
    # Consumer Blackwell (sm_120: RTX 5090 / 5080 / 5060 Ti) under cu128 has known cuDNN NHWC
    # bfloat16 stride/alignment kernel bugs on odd spatial dims (9x9, 13x13, 19x19) and dynamic
    # MCTS batch sizes (1..512) that can silently emit NaNs; use contiguous NCHW on sm_120+.
    is_blackwell_or_newer = (props.major >= 12)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = (not is_blackwell_or_newer)
    device_name = f"{torch.cuda.get_device_name(0)} ({GPU_VRAM_GB:.1f} GB VRAM, sm_{props.major}{props.minor})"
    AMP_ENABLED = True
    USE_CHANNELS_LAST = (not is_blackwell_or_newer)
    if torch.cuda.is_bf16_supported():
        # Native bfloat16 Tensor Cores on Ampere / Ada / Hopper / Blackwell (RTX 5060 Ti, RTX 5090)
        AMP_DTYPE = torch.bfloat16
        USE_GRAD_SCALER = False
    else:
        AMP_DTYPE = torch.float16
        USE_GRAD_SCALER = True
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    GPU_VRAM_GB = SYSTEM_RAM_GB  # Unified memory on Apple Silicon
    device_name = f"Apple Silicon Metal / MPS ({platform.machine()}, {SYSTEM_RAM_GB:.0f} GB Unified)"
    AMP_ENABLED = True
    AMP_DTYPE = torch.bfloat16
    USE_GRAD_SCALER = False
    USE_CHANNELS_LAST = True
else:
    tpu_env_present = any(
        k in os.environ for k in ("TPU_NAME", "COLAB_TPU_ADDR", "PJRT_DEVICE", "XRT_TPU_CONFIG")
    ) or os.path.exists("/dev/accel0") or os.path.exists("/dev/vfio")
    if tpu_env_present:
        try:
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            import torch_xla.core.xla_model as _xm
            xm = _xm
            device = xm.xla_device()
            IS_TPU = True
            AMP_ENABLED = True
            AMP_DTYPE = torch.bfloat16
            device_name = f"Google Cloud TPU ({xm.xla_real_devices([str(device)])[0]})"
        except Exception:
            device = torch.device("cpu")
            device_name = f"CPU ({platform.machine()})"
    else:
        device = torch.device("cpu")
        device_name = f"CPU ({platform.machine()})"

# Fixed batch size buckets for TPU (XLA) to prevent dynamic-shape graph recompilations
TPU_BATCH_BUCKETS = (16, 32, 64, 128, 256, 512)

# --- V2 Hyperparameters (Auto-scaled by Hardware Profile) ---
N_ITER = 50
N_GAMES_PER_ITER = 500
N_MCTS_SIMS = 600
MAX_REPLAY_BUFFER_SIZE = 250000 if SYSTEM_RAM_GB < 12.0 else (750000 if SYSTEM_RAM_GB >= 48.0 else 500000)
REPLAY_RATIO = 1.5  # Controlled training sample ratio relative to newly generated positions

# MCTS Setup
MCTS_BATCH_SIZE = 16
GPU_SERVER_BATCH_SIZE = 512 if (device.type != "cpu" and (GPU_VRAM_GB >= 10.0 or IS_TPU)) else 256
C_PUCT = 1.1
C_PUCT_BASE = 19652.0
C_FPU_INTERNAL = 0.20
C_FPU_ROOT = 0.0
DIRICHLET_EPSILON = 0.25

# Playout Cap Oscillations & Resignation
FULL_SEARCH_PROB = 0.25       # 25% full search (policy_weight=1.0), 75% fast search (policy_weight=0.0)
FAST_SEARCH_RATIO = 0.20      # Fast search runs ~20% of full sims (minimum 80 when sims >= 80)
FAST_SEARCH_MIN_SIMS = 80
FULL_PLAYOUT_GAME_PROB = 0.25 # 25% of games disable resignation to play cleanly to double-pass
RESIGN_THRESHOLD = -0.90      # Root & best-child value threshold for calibrated resignation
RESIGN_CONSECUTIVE_PLIES = 5  # Consecutive plies below threshold required to resign

# Board-Size-Agnostic Global-Pooling ResNet Setup
IN_PLANES = 18
RESNET_CHANNELS = 192
RESNET_BLOCKS = 15
TRAIN_BATCH_SIZE = 256 if (device.type == "cuda" and GPU_VRAM_GB >= 12.0) else 128
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
VALUE_LOSS_WEIGHT = 1.2
OWNERSHIP_LOSS_WEIGHT = 0.5
EVAL_INTERVAL = 5
EVAL_GAMES = 20
PLAY_SYMMETRIES = 8


def get_dirichlet_alpha(board_size: int) -> float:
    """Scales Dirichlet concentration parameter inversely with board intersections."""
    if board_size >= 19:
        return 0.03
    if board_size >= 13:
        return 0.06
    return 0.15


DIRICHLET_ALPHA = get_dirichlet_alpha(BOARD_SIZE)


def get_temp_threshold(board_size: int) -> int:
    """Move number threshold across which opening exploration temperature decays smoothly."""
    if board_size >= 19:
        return 30
    if board_size >= 13:
        return 16
    return 10


def get_scheduled_lr(stage_iter: int, base_lr: float = LEARNING_RATE, warmup_iters_left: int = 0, warmup_step: int = 0) -> float:
    """
    Step-decay learning rate schedule with automatic 2-iteration curriculum transition warmup
    when transferring weights to a larger board size (9x9 -> 13x13 -> 19x19).
    """
    if stage_iter < 25:
        lr = base_lr
    elif stage_iter < 45:
        lr = base_lr * 0.3
    else:
        lr = base_lr * 0.1

    if warmup_iters_left > 0:
        warmup_factor = 0.25 if warmup_step == 0 else 0.60
        return lr * warmup_factor
    return lr


# Multi-core CPU worker allocation (auto-scaled for Ryzen 5800XT / 7900, Apple Silicon, and Colab)
sys_cores = os.cpu_count() or 1
is_colab = (
    "COLAB_RELEASE_TAG" in os.environ
    or "COLAB_GPU" in os.environ
    or "COLAB_TPU_ADDR" in os.environ
    or os.path.exists("/content")
)
if is_colab:
    if sys_cores >= 16:
        suggested_cores = sys_cores - 2
    else:
        suggested_cores = max(4, int(sys_cores * 1.5))
    PARALLEL_THREADS = min(64, suggested_cores)
else:
    suggested_cores = max(1, sys_cores - 2) if sys_cores > 2 else sys_cores
    PARALLEL_THREADS = min(48, suggested_cores)


def init_environment():
    """Initializes logging and displays profiled hardware runtime details."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_start = time.strftime("%Y-%m-%d %H:%M:%S")
    log_filename_time = time.strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(script_dir, f"logs_{log_filename_time}.log")

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    amp_str = str(AMP_DTYPE).replace("torch.", "") if AMP_ENABLED else "float32"
    layout_str = "channels_last" if USE_CHANNELS_LAST else "contiguous"

    print("========================================")
    print("OceanGo V2 Initializing...")
    print(f"OS Platform: {platform.system()} ({platform.machine()})")
    print(f"Computing Device: {device.type.upper()} | {device_name}")
    print(f"Precision & Layout: {amp_str} | {layout_str}")
    print(f"System Memory: {SYSTEM_RAM_GB:.1f} GB RAM | Replay Cap: {MAX_REPLAY_BUFFER_SIZE:,} states")
    print(f"CPU Threads: {sys_cores} available | Using {PARALLEL_THREADS} self-play workers")
    print("========================================\n")

    logging.info(f"\n\n{'=' * 40}")
    logging.info(f"Starting OceanGo V2 log at {log_start}")
    logging.info(f"OS Platform: {platform.system()} ({platform.machine()})")
    logging.info(f"Computing Device: {device.type.upper()} | {device_name}")
    logging.info(f"Precision & Layout: {amp_str} | {layout_str}")
    logging.info(f"System RAM: {SYSTEM_RAM_GB:.1f} GB | Workers: {PARALLEL_THREADS}/{sys_cores}")
    logging.info(f"{'=' * 40}")


# ====================================================
#   Board-Size-Agnostic Global-Pooling Neural Network
# ====================================================
class KataGlobalPool(nn.Module):
    """
    KataGo-style board-size-aware global pooling layer.
    Computes channel-wise [mean, mean * (N - 14.0) / 10.0, max] -> (B, 3 * C),
    explicitly encoding board scale relative to 7.5 Komi across 9x9, 13x13, and 19x19.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        board_n = float(x.size(-1))
        scale = (board_n - 14.0) / 10.0
        m = x.mean(dim=(-2, -1))
        mx = x.amax(dim=(-2, -1))
        return torch.cat([m, m * scale, mx], dim=1)


class ResBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class GlobalPoolResBlock(nn.Module):
    """
    Residual block augmented with KataGlobalPool context bias so the trunk
    integrates whole-board territory, ladders, ko threats, and liberty counts regardless of N.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        g_channels = max(16, num_channels // 3)
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)

        self.conv_g = nn.Conv2d(num_channels, g_channels, kernel_size=3, padding=1, bias=False)
        self.bn_g = nn.BatchNorm2d(g_channels)
        self.global_pool = KataGlobalPool()
        self.fc_g = nn.Linear(3 * g_channels, num_channels, bias=False)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn1(self.conv1(x))
        g = F.relu(self.bn_g(self.conv_g(x)))
        g_pool = self.global_pool(g)
        g_bias = self.fc_g(g_pool).unsqueeze(-1).unsqueeze(-1)
        out = F.relu(h + g_bias)
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class PolicyValueNet(nn.Module):
    """
    Board-Size-Agnostic Global-Pooling Residual Network (V2).
    100% of learned parameters are independent of board dimension N, enabling
    zero-loss curriculum transfer across 9x9 -> 13x13 -> 19x19.

    Input:  (B, 18, N, N)
    Heads:
      1. Policy Head:    (B, N*N + 1) move + pass logits
      2. Value Head:     (B, 1) inference value v = tanh(0.5 * u) in [-1, +1] (or raw logit u when return_aux=True)
      3. Ownership Head: (B, N, N) per-intersection ownership logits (evaluated only when return_aux=True)
    """
    def __init__(self, board_size: int = BOARD_SIZE, num_channels: int = RESNET_CHANNELS, num_res_blocks: int = RESNET_BLOCKS):
        super().__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks

        # Stem
        self.conv_in = nn.Conv2d(IN_PLANES, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)

        # Residual Tower: insert GlobalPoolResBlock every 3rd residual block
        blocks = []
        for i in range(num_res_blocks):
            if (i + 1) % 3 == 0:
                blocks.append(GlobalPoolResBlock(num_channels))
            else:
                blocks.append(ResBlock(num_channels))
        self.res_blocks = nn.ModuleList(blocks)
        self.global_pool = KataGlobalPool()

        # 1. Board-Agnostic Policy Head (spatial 1x1 conv + global context bias + pooled Pass logit)
        head_c = 32
        self.policy_conv1 = nn.Conv2d(num_channels, head_c, kernel_size=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(head_c)
        self.policy_g_conv = nn.Conv2d(num_channels, head_c, kernel_size=1, bias=False)
        self.policy_g_bn = nn.BatchNorm2d(head_c)
        self.policy_g_fc = nn.Linear(3 * head_c, head_c, bias=False)
        self.policy_conv2 = nn.Conv2d(head_c, 1, kernel_size=1, bias=True)
        self.policy_pass_fc = nn.Linear(3 * head_c, 1, bias=True)

        # 2. Board-Agnostic Value Head (outputs raw win-rate logit u; v = tanh(0.5 * u))
        self.value_conv = nn.Conv2d(num_channels, head_c, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(head_c)
        self.value_fc1 = nn.Linear(3 * head_c, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # 3. Auxiliary Ownership Head (computed only during training or UI overlay)
        self.ownership_conv1 = nn.Conv2d(num_channels, head_c, kernel_size=1, bias=False)
        self.ownership_bn1 = nn.BatchNorm2d(head_c)
        self.ownership_conv2 = nn.Conv2d(head_c, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy Head
        p_spatial = self.policy_bn1(self.policy_conv1(x))
        p_g = F.relu(self.policy_g_bn(self.policy_g_conv(x)))
        p_pool = self.global_pool(p_g)
        p_act = F.relu(p_spatial + self.policy_g_fc(p_pool).unsqueeze(-1).unsqueeze(-1))
        p_board = self.policy_conv2(p_act).flatten(1)   # Safe under channels_last: (B, N*N)
        p_pass = self.policy_pass_fc(p_pool)            # (B, 1)
        policy_logits = torch.cat([p_board, p_pass], dim=1)

        # Value Head
        v_feat = F.relu(self.value_bn(self.value_conv(x)))
        v_pool = self.global_pool(v_feat)               # (B, 96)
        v_hid = F.relu(self.value_fc1(v_pool))
        value_logit = self.value_fc2(v_hid)             # Raw win-rate logit u: (B, 1)

        if return_aux:
            own_feat = F.relu(self.ownership_bn1(self.ownership_conv1(x)))
            ownership_logits = self.ownership_conv2(own_feat).squeeze(1)  # (B, N, N)
            return policy_logits, value_logit, ownership_logits

        # Inference value in [-1, +1]: 2 * sigmoid(u) - 1 == tanh(0.5 * u)
        value = torch.tanh(0.5 * value_logit)
        return policy_logits, value


# ====================================================
#   Fast Incremental GoGame Engine (Zobrist + Chains)
# ====================================================
_BOARD_TABLES = {}


def _get_board_tables(board_size: int) -> dict:
    """
    Precomputes deterministic 64-bit Zobrist tables, flat 1D neighbor indices,
    and bitset masks for O(1) incremental chain & liberty updates.
    """
    tbl = _BOARD_TABLES.get(board_size)
    if tbl is not None:
        return tbl

    n_pts = board_size * board_size
    rng = random.Random(0x0CEA4602 ^ (board_size * 0x9E3779B1))
    zobrist_p1 = tuple(rng.getrandbits(64) for _ in range(n_pts))
    zobrist_p2 = tuple(rng.getrandbits(64) for _ in range(n_pts))
    zobrist = (None, zobrist_p1, zobrist_p2)

    neighbors = []
    pt_bit = []
    adj_mask = []
    coord_of = []
    for r in range(board_size):
        for c in range(board_size):
            idx = r * board_size + c
            pt_bit.append(1 << idx)
            coord_of.append((r, c))
            nbs = []
            amask = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < board_size and 0 <= nc < board_size:
                    n_idx = nr * board_size + nc
                    nbs.append(n_idx)
                    amask |= (1 << n_idx)
            neighbors.append(tuple(nbs))
            adj_mask.append(amask)

    tbl = {
        "n_pts": n_pts,
        "full_mask": (1 << n_pts) - 1,
        "zobrist": zobrist,
        "neighbors": tuple(neighbors),
        "pt_bit": tuple(pt_bit),
        "adj_mask": tuple(adj_mask),
        "coord_of": tuple(coord_of),
        "powers_u16": (1 << np.arange(8, dtype=np.uint16)),
    }
    _BOARD_TABLES[board_size] = tbl
    return tbl


class GoGame:
    """
    Exact Chinese-rules Go engine accelerated with 64-bit Zobrist hashing,
    immutable stone-chain tuples (color, stones_mask, neighbor_mask, zobrist_hash),
    and Python arbitrary-precision int bitmasks for O(1) liberty & superko checks.
    """
    __slots__ = (
        "board_size",
        "_tbl",
        "board",
        "current_player",
        "board_hash",
        "history",
        "passes_in_a_row",
        "empty_mask",
        "chain_map",
        "history_states",
        "_legal_moves_cache",
        "_winner_cache",
    )

    def __init__(self, board_size: int = BOARD_SIZE):
        self.board_size = board_size
        tbl = _get_board_tables(board_size)
        self._tbl = tbl
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1
        self.board_hash = 0
        self.history = {0}
        self.passes_in_a_row = 0
        self.empty_mask = tbl["full_mask"]
        self.chain_map = [None] * tbl["n_pts"]
        zero_snap = np.zeros((board_size, board_size), dtype=np.int8)
        self.history_states = (zero_snap,) * 8
        self._legal_moves_cache = None
        self._winner_cache = None

    def copy(self) -> "GoGame":
        g = GoGame.__new__(GoGame)
        g.board_size = self.board_size
        g._tbl = self._tbl
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.board_hash = self.board_hash
        g.history = self.history.copy()
        g.passes_in_a_row = self.passes_in_a_row
        g.empty_mask = self.empty_mask
        # Chain entries are immutable 4-tuples; shallow list slice copies all chains in O(N^2) C-pointer copy
        g.chain_map = self.chain_map[:]
        g.history_states = self.history_states
        g._legal_moves_cache = self._legal_moves_cache
        g._winner_cache = self._winner_cache
        return g

    def get_state_np(self):
        return (self.board.copy(), self.current_player)

    def get_nn_input(self) -> np.ndarray:
        """Returns canonical 18 x N x N float32 input tensor."""
        b_size = self.board_size
        planes = np.zeros((IN_PLANES, b_size, b_size), dtype=np.float32)
        cp = self.current_player
        opp = 3 - cp
        for i, board_i in enumerate(self.history_states):
            planes[i] = (board_i == cp)
            planes[i + 8] = (board_i == opp)
        if cp == 1:
            planes[16] = 1.0
        planes[17] = 1.0  # Constant on-board area mask for zero-padded 3x3 edge detection
        return planes

    def write_nn_input_uint8(self, out_planes: np.ndarray) -> None:
        """Writes canonical 18 x N x N binary features directly into a uint8 buffer (zero extra allocation)."""
        cp = self.current_player
        opp = 3 - cp
        for i, board_i in enumerate(self.history_states):
            np.equal(board_i, cp, out=out_planes[i])
            np.equal(board_i, opp, out=out_planes[i + 8])
        out_planes[16].fill(1 if cp == 1 else 0)
        out_planes[17].fill(1)

    def get_packed_state(self):
        """
        Packs all 16 binary stone history planes into a single uint16[N, N] array (722 bytes on 19x19).
        Bits 0..7: current player stones (t..t-7); Bits 8..15: opponent stones (t..t-7).
        """
        cp = self.current_player
        opp = 3 - cp
        packed = np.zeros((self.board_size, self.board_size), dtype=np.uint16)
        for i, board_i in enumerate(self.history_states):
            if i == 0 or board_i is not self.history_states[i - 1]:
                cur_m = (board_i == cp).astype(np.uint16)
                opp_m = (board_i == opp).astype(np.uint16)
            packed |= (cur_m << i) | (opp_m << (i + 8))
        return packed, cp

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def _try_move_idx(self, idx: int) -> bool:
        """O(1) legality check at empty intersection `idx` using bitmask liberties and Zobrist superko."""
        tbl = self._tbl
        pt = tbl["pt_bit"][idx]
        new_empty = self.empty_mask ^ pt
        cp = self.current_player
        opp = 3 - cp

        new_hash = self.board_hash ^ tbl["zobrist"][cp][idx]
        has_lib = (tbl["adj_mask"][idx] & new_empty) != 0
        captured_stones = 0
        chain_map = self.chain_map

        for n_idx in tbl["neighbors"][idx]:
            nc = chain_map[n_idx]
            if nc is not None:
                if nc[0] == opp:
                    if (nc[2] & new_empty) == 0 and (captured_stones & nc[1]) == 0:
                        captured_stones |= nc[1]
                        new_hash ^= nc[3]
                else:
                    if not has_lib and (nc[2] & new_empty) != 0:
                        has_lib = True

        return (has_lib or captured_stones != 0) and (new_hash not in self.history)

    def is_valid_move(self, x: int, y: int) -> bool:
        if x == -1 and y == -1:
            return True
        b_size = self.board_size
        if x < 0 or x >= b_size or y < 0 or y >= b_size:
            return False
        if self.board[x, y] != 0:
            return False
        return self._try_move_idx(x * b_size + y)

    def make_move(self, x: int, y: int, check_valid: bool = True) -> bool:
        if x == -1 and y == -1:
            self.passes_in_a_row += 1
            self.current_player = 3 - self.current_player
            self.history.add(self.board_hash)
            self.history_states = (self.history_states[0],) + self.history_states[:7]
            self._legal_moves_cache = None
            self._winner_cache = None
            return True

        b_size = self.board_size
        if check_valid:
            if x < 0 or x >= b_size or y < 0 or y >= b_size or self.board[x, y] != 0:
                return False
            idx = x * b_size + y
            if not self._try_move_idx(idx):
                return False
        else:
            idx = x * b_size + y

        tbl = self._tbl
        pt = tbl["pt_bit"][idx]
        cp = self.current_player
        opp = 3 - cp
        zob_pt = tbl["zobrist"][cp][idx]

        self.passes_in_a_row = 0
        self.board[x, y] = cp
        pre_empty = self.empty_mask & ~pt
        new_hash = self.board_hash ^ zob_pt

        captured_stones = 0
        m_stones = pt
        m_neighbors = tbl["adj_mask"][idx]
        m_zob = zob_pt
        chain_map = self.chain_map

        for n_idx in tbl["neighbors"][idx]:
            nc = chain_map[n_idx]
            if nc is not None:
                if nc[0] == opp:
                    if (nc[2] & pre_empty) == 0 and (captured_stones & nc[1]) == 0:
                        captured_stones |= nc[1]
                        new_hash ^= nc[3]
                else:
                    if (m_stones & nc[1]) == 0:
                        m_stones |= nc[1]
                        m_neighbors |= nc[2]
                        m_zob ^= nc[3]

        if captured_stones:
            coord_of = tbl["coord_of"]
            board = self.board
            temp = captured_stones
            while temp:
                lsb = temp & -temp
                s_idx = lsb.bit_length() - 1
                chain_map[s_idx] = None
                rs, cs = coord_of[s_idx]
                board[rs, cs] = 0
                temp ^= lsb
            self.empty_mask = pre_empty | captured_stones
        else:
            self.empty_mask = pre_empty

        new_chain = (cp, m_stones, m_neighbors, m_zob)
        if m_stones == pt:
            chain_map[idx] = new_chain
        else:
            temp = m_stones
            while temp:
                lsb = temp & -temp
                chain_map[lsb.bit_length() - 1] = new_chain
                temp ^= lsb

        self.board_hash = new_hash
        self.history.add(new_hash)
        self.current_player = opp
        self.history_states = (self.board.copy(),) + self.history_states[:7]
        self._legal_moves_cache = None
        self._winner_cache = None
        return True

    def get_legal_moves(self):
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache

        moves = []
        tbl = self._tbl
        empty_mask = self.empty_mask
        board_hash = self.board_hash
        zob_cp = tbl["zobrist"][self.current_player]
        adj_mask = tbl["adj_mask"]
        neighbors = tbl["neighbors"]
        coord_of = tbl["coord_of"]
        chain_map = self.chain_map
        opp = 3 - self.current_player
        history = self.history

        temp_empty = empty_mask
        while temp_empty:
            lsb = temp_empty & -temp_empty
            idx = lsb.bit_length() - 1
            temp_empty ^= lsb

            new_empty = empty_mask ^ lsb
            new_hash = board_hash ^ zob_cp[idx]
            has_lib = (adj_mask[idx] & new_empty) != 0
            captured_stones = 0

            for n_idx in neighbors[idx]:
                nc = chain_map[n_idx]
                if nc is not None:
                    if nc[0] == opp:
                        if (nc[2] & new_empty) == 0 and (captured_stones & nc[1]) == 0:
                            captured_stones |= nc[1]
                            new_hash ^= nc[3]
                    else:
                        if not has_lib and (nc[2] & new_empty) != 0:
                            has_lib = True

            if (has_lib or captured_stones != 0) and (new_hash not in history):
                moves.append(coord_of[idx])

        # Place Pass (-1, -1) last (matching policy_logits[N*N]) so tie-breaks / zero-visit
        # fallbacks never default to passing when legal board intersections exist.
        moves.append((-1, -1))
        self._legal_moves_cache = moves
        return moves

    def is_game_over(self) -> bool:
        return self.passes_in_a_row >= 2

    def get_group(self, r: int, c: int):
        """Returns (color, set_of_coordinates) for the connected stone group at (r, c)."""
        cell = int(self.board[r, c])
        if cell == 0:
            return 0, set()
        idx = r * self.board_size + c
        nc = self.chain_map[idx]
        if nc is not None and nc[0] == cell:
            coord_of = self._tbl["coord_of"]
            grp = set()
            temp = nc[1]
            while temp:
                lsb = temp & -temp
                grp.add(coord_of[lsb.bit_length() - 1])
                temp ^= lsb
            return cell, grp

        # Fallback BFS if self.board was modified directly outside make_move
        q = deque([(r, c)])
        group = {(r, c)}
        while q:
            cr, cc = q.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc_c = cr + dr, cc + dc
                if 0 <= nr < self.board_size and 0 <= nc_c < self.board_size:
                    if self.board[nr, nc_c] == cell and (nr, nc_c) not in group:
                        group.add((nr, nc_c))
                        q.append((nr, nc_c))
        return cell, group

    def auto_detect_dead_stones(self):
        """
        Identifies dead / hopeless trapped stone groups lacking two eyes at game end,
        ensuring isolated dead stones inside opponent territory never zero out surrounding score.
        """
        b_size = self.board_size
        tbl = self._tbl
        coord_of = tbl["coord_of"]
        neighbors = tbl["neighbors"]
        board_flat = self.board.reshape(-1)
        n_pts = tbl["n_pts"]

        # Gather all unique chains and their liberty sets
        seen_chains = {}
        point_to_gid = [-1] * n_pts
        groups = []

        visited_pts = [False] * n_pts
        for idx in range(n_pts):
            col = int(board_flat[idx])
            if col == 0 or visited_pts[idx]:
                continue
            # Use chain_map when synchronized, else BFS
            nc = self.chain_map[idx]
            if nc is not None and nc[0] == col:
                sm = nc[1]
                if sm in seen_chains:
                    continue
                gid = len(groups)
                seen_chains[sm] = gid
                stones = []
                temp = sm
                while temp:
                    lsb = temp & -temp
                    s_idx = lsb.bit_length() - 1
                    visited_pts[s_idx] = True
                    point_to_gid[s_idx] = gid
                    stones.append(s_idx)
                    temp ^= lsb
                lib_mask = nc[2] & self.empty_mask
                libs = []
                temp = lib_mask
                while temp:
                    lsb = temp & -temp
                    libs.append(lsb.bit_length() - 1)
                    temp ^= lsb
            else:
                gid = len(groups)
                q = [idx]
                visited_pts[idx] = True
                point_to_gid[idx] = gid
                stones = []
                libs_set = set()
                while q:
                    cur = q.pop()
                    stones.append(cur)
                    for nb in neighbors[cur]:
                        nb_col = int(board_flat[nb])
                        if nb_col == 0:
                            libs_set.add(nb)
                        elif nb_col == col and not visited_pts[nb]:
                            visited_pts[nb] = True
                            point_to_gid[nb] = gid
                            q.append(nb)
                libs = list(libs_set)

            groups.append({
                "gid": gid,
                "color": col,
                "stones": stones,
                "libs": libs,
                "dead": False,
            })

        if not groups:
            return set()

        # Pass 1: Classic low-liberty surrounded small groups (ordered smallest/fewest-liberties first
        # so mutual-contact corners never mark both sides dead simultaneously)
        order = sorted(range(len(groups)), key=lambda i: (len(groups[i]["libs"]), len(groups[i]["stones"])))
        for gi in order:
            g = groups[gi]
            col = g["color"]
            opp = 3 - col
            libs = g["libs"]
            n_stones = len(g["stones"])
            if len(libs) == 0:
                g["dead"] = True
                continue
            if len(libs) <= 2 and n_stones <= 5:
                all_surrounded = True
                stronger_opp_present = False
                for l_idx in libs:
                    touches_live_opp = False
                    for nb in neighbors[l_idx]:
                        if int(board_flat[nb]) == opp:
                            opp_g = groups[point_to_gid[nb]]
                            if not opp_g["dead"]:
                                touches_live_opp = True
                                if len(opp_g["libs"]) > len(libs) or len(opp_g["stones"]) > n_stones or len(opp_g["libs"]) >= 3:
                                    stronger_opp_present = True
                    if not touches_live_opp:
                        all_surrounded = False
                        break
                if all_surrounded and stronger_opp_present:
                    g["dead"] = True

        # Pass 2: Enclosed regions bounded strictly by live opponent chains where total interior
        # eye-space cannot possibly form 2 eyes (<= 1 empty point inside opponent wall)
        for target_col in (1, 2):
            opp = 3 - target_col
            seen_reg = [False] * n_pts
            for start_idx in range(n_pts):
                if seen_reg[start_idx]:
                    continue
                c_val = int(board_flat[start_idx])
                if c_val == opp and not groups[point_to_gid[start_idx]]["dead"]:
                    continue
                # Flood-fill region treating live `opp` stones as walls
                q = [start_idx]
                seen_reg[start_idx] = True
                empty_cnt = 0
                target_gids = set()
                touches_live_opp_strong = False
                reg_size = 0
                while q:
                    cur = q.pop()
                    reg_size += 1
                    cv = int(board_flat[cur])
                    if cv == 0:
                        empty_cnt += 1
                    elif cv == target_col:
                        tg = point_to_gid[cur]
                        if not groups[tg]["dead"]:
                            target_gids.add(tg)
                    else:
                        # Dead opp stone counts as empty space inside region
                        empty_cnt += 1
                    for nb in neighbors[cur]:
                        nb_col = int(board_flat[nb])
                        if nb_col == opp and not groups[point_to_gid[nb]]["dead"]:
                            opp_g = groups[point_to_gid[nb]]
                            if len(opp_g["libs"]) >= 2 or len(opp_g["stones"]) >= 4:
                                touches_live_opp_strong = True
                        elif not seen_reg[nb]:
                            seen_reg[nb] = True
                            q.append(nb)

                if target_gids and touches_live_opp_strong and empty_cnt <= 1 and reg_size <= max(8, b_size):
                    for tg in target_gids:
                        groups[tg]["dead"] = True

        dead_coords = set()
        for g in groups:
            if g["dead"]:
                for s_idx in g["stones"]:
                    dead_coords.add(coord_of[s_idx])
        return dead_coords

    def get_score_breakdown(self, dead_stones=None):
        """Chinese area scoring with 7.5 Komi."""
        b_size = self.board_size
        dead_set = set() if dead_stones is None else set(dead_stones)

        scoring_board = self.board.copy()
        for r, c in dead_set:
            scoring_board[r, c] = 0

        b_alive_stones = {
            (r, c) for r in range(b_size) for c in range(b_size)
            if scoring_board[r, c] == 1
        }
        w_alive_stones = {
            (r, c) for r in range(b_size) for c in range(b_size)
            if scoring_board[r, c] == 2
        }

        b_terr_points = set()
        w_terr_points = set()
        visited = np.zeros((b_size, b_size), dtype=np.bool_)

        for r in range(b_size):
            for c in range(b_size):
                if scoring_board[r, c] == 0 and not visited[r, c]:
                    q = [(r, c)]
                    visited[r, c] = True
                    group = [(r, c)]
                    touches_black = False
                    touches_white = False

                    while q:
                        cx, cy = q.pop()
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < b_size and 0 <= ny < b_size:
                                val = int(scoring_board[nx, ny])
                                if val == 1:
                                    touches_black = True
                                elif val == 2:
                                    touches_white = True
                                elif not visited[nx, ny]:
                                    visited[nx, ny] = True
                                    group.append((nx, ny))
                                    q.append((nx, ny))

                    if touches_black and not touches_white:
                        b_terr_points.update(group)
                    elif touches_white and not touches_black:
                        w_terr_points.update(group)

        komi = 7.5
        b_score = len(b_alive_stones) + len(b_terr_points)
        w_score = len(w_alive_stones) + len(w_terr_points) + komi

        if b_score > w_score:
            winner = 1
        elif w_score > b_score:
            winner = 2
        else:
            winner = 0

        return {
            "b_stones": len(b_alive_stones),
            "w_stones": len(w_alive_stones),
            "b_alive_stones": b_alive_stones,
            "w_alive_stones": w_alive_stones,
            "b_terr": len(b_terr_points),
            "w_terr": len(w_terr_points),
            "b_terr_points": b_terr_points,
            "w_terr_points": w_terr_points,
            "dead_stones": dead_set,
            "komi": komi,
            "b_score": b_score,
            "w_score": w_score,
            "winner": winner,
            "margin": abs(b_score - w_score),
        }

    def get_ownership_target(self, player: int = 1, dead_stones=None) -> np.ndarray:
        """
        Returns an (N, N) float32 ownership map at game end:
          1.0 = owned by `player` (default Black=1)
          0.0 = owned by opponent (3 - `player`)
          0.5 = neutral dame
        """
        if dead_stones is None:
            dead_stones = self.auto_detect_dead_stones()
        info = self.get_score_breakdown(dead_stones=dead_stones)
        own = np.full((self.board_size, self.board_size), 0.5, dtype=np.float32)
        b_val = 1.0 if player == 1 else 0.0
        w_val = 0.0 if player == 1 else 1.0
        for r, c in info["b_alive_stones"]:
            own[r, c] = b_val
        for r, c in info["b_terr_points"]:
            own[r, c] = b_val
        for r, c in info["w_alive_stones"]:
            own[r, c] = w_val
        for r, c in info["w_terr_points"]:
            own[r, c] = w_val
        return own

    def get_winner(self) -> int:
        if self._winner_cache is not None:
            return self._winner_cache
        dead = self.auto_detect_dead_stones()
        w = self.get_score_breakdown(dead_stones=dead)["winner"]
        self._winner_cache = w
        return w


# ====================================================
#  Accelerated Inference, Dihedral Symmetries & TPU
# ====================================================
def boards_to_tensor(batch_of_inputs) -> torch.Tensor:
    arr_np = batch_of_inputs if isinstance(batch_of_inputs, np.ndarray) else np.stack(batch_of_inputs, axis=0)
    arr_np = np.ascontiguousarray(arr_np)
    t = torch.from_numpy(arr_np).to(device=device, dtype=torch.float32)
    if USE_CHANNELS_LAST and t.ndim == 4:
        t = t.to(memory_format=torch.channels_last)
    return t


def _get_tpu_bucket_size(n: int) -> int:
    for b in TPU_BATCH_BUCKETS:
        if n <= b:
            return b
    return ((n + 63) // 64) * 64


def forward_infer_batch(net: nn.Module, batch_of_inputs):
    """
    Batched neural network inference across CUDA (native bfloat16/float16),
    Apple Silicon MPS, Google Cloud TPU (XLA bfloat16 buckets), or CPU.
    Accepts either float32 or uint8 (B, 18, N, N) arrays/tensors.
    """
    arr_np = batch_of_inputs if isinstance(batch_of_inputs, np.ndarray) else np.stack(batch_of_inputs, axis=0)
    n = arr_np.shape[0]
    if n == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.float32)

    if IS_TPU:
        bucket_n = _get_tpu_bucket_size(n)
        if bucket_n > n:
            pad_shape = (bucket_n - n,) + arr_np.shape[1:]
            arr_np = np.concatenate([arr_np, np.zeros(pad_shape, dtype=arr_np.dtype)], axis=0)
        input_tensor = torch.from_numpy(np.ascontiguousarray(arr_np)).to(device=device, dtype=torch.float32)
        with torch.no_grad(), torch.amp.autocast("xla", dtype=torch.bfloat16):
            policy_logits, value_out = net(input_tensor)
        if xm is not None:
            xm.mark_step()
        policy_np = policy_logits.float().cpu().numpy()[:n]
        value_np = value_out.float().squeeze(1).cpu().numpy()[:n]
        return policy_np, value_np

    # Transfer compact uint8 or float32 synchronously, then cast on GPU
    input_tensor = torch.from_numpy(np.ascontiguousarray(arr_np)).to(device=device)
    if input_tensor.dtype != torch.float32:
        input_tensor = input_tensor.to(dtype=torch.float32)
    if USE_CHANNELS_LAST and input_tensor.ndim == 4:
        input_tensor = input_tensor.to(memory_format=torch.channels_last)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
        policy_logits, value_out = net(input_tensor)

    policy_np = policy_logits.float().cpu().numpy()
    value_np = value_out.float().squeeze(1).cpu().numpy()
    return policy_np, value_np


def apply_dihedral_symmetry_torch(
    states: torch.Tensor,
    policies: torch.Tensor = None,
    ownerships: torch.Tensor = None,
    sym_id: int = 0,
):
    """
    Applies exact D4 dihedral symmetry `sym_id` in 0..7 to:
      - states:     (..., C, N, N) or (..., N, N)
      - policies:   (B, N*N + 1)
      - ownerships: (B, N, N)
    """
    rot_k = sym_id & 3
    do_flip = sym_id >= 4

    s_out = states
    if rot_k:
        s_out = torch.rot90(s_out, k=rot_k, dims=(-2, -1))
    if do_flip:
        s_out = torch.flip(s_out, dims=(-1,))

    p_out = policies
    if policies is not None:
        bsz = policies.size(0)
        n = states.size(-1)
        p_board = policies[:, : n * n].reshape(bsz, n, n)
        p_pass = policies[:, n * n :]
        if rot_k:
            p_board = torch.rot90(p_board, k=rot_k, dims=(-2, -1))
        if do_flip:
            p_board = torch.flip(p_board, dims=(-1,))
        p_out = torch.cat([p_board.reshape(bsz, n * n), p_pass], dim=1)

    o_out = ownerships
    if ownerships is not None:
        if rot_k:
            o_out = torch.rot90(o_out, k=rot_k, dims=(-2, -1))
        if do_flip:
            o_out = torch.flip(o_out, dims=(-1,))

    return s_out, p_out, o_out


def ensemble_infer_symmetries(net: nn.Module, batch_of_inputs, num_symmetries: int = 8):
    """
    Ensembles `num_symmetries` (1..8) dihedral rotations/reflections in a single batched GPU pass
    and averages inverse-transformed post-softmax policy probabilities (returned as log-probs)
    and value predictions so rotation-invariant Pass is never artificially boosted over spatial moves.
    """
    if num_symmetries <= 1:
        return forward_infer_batch(net, batch_of_inputs)

    arr_np = batch_of_inputs if isinstance(batch_of_inputs, np.ndarray) else np.stack(batch_of_inputs, axis=0)
    m, c, n, _ = arr_np.shape
    k_syms = min(8, max(1, int(num_symmetries)))
    sym_ids = list(range(8)) if k_syms == 8 else random.sample(range(8), k_syms)

    base_t = torch.from_numpy(arr_np)
    aug_list = []
    for sid in sym_ids:
        s_aug, _, _ = apply_dihedral_symmetry_torch(base_t, sym_id=sid)
        aug_list.append(s_aug)
    aug_batch = torch.cat(aug_list, dim=0).numpy()  # (k_syms * m, 18, N, N)

    p_all, v_all = forward_infer_batch(net, aug_batch)
    p_all_t = torch.from_numpy(p_all).reshape(k_syms, m, n * n + 1)
    p_all_t = torch.nan_to_num(p_all_t, nan=0.0, posinf=20.0, neginf=-20.0)
    probs_all_t = torch.softmax(p_all_t, dim=-1)
    v_clean = np.nan_to_num(v_all, nan=0.0, posinf=1.0, neginf=-1.0)
    v_mean = v_clean.reshape(k_syms, m).mean(axis=0)

    prob_acc = torch.zeros((m, n * n + 1), dtype=torch.float32)
    for idx_s, sid in enumerate(sym_ids):
        rot_k = sid & 3
        do_flip = sid >= 4
        pb = probs_all_t[idx_s, :, : n * n].reshape(m, n, n)
        pp = probs_all_t[idx_s, :, n * n :]
        # Exact inverse of (rot90(k) -> optional flip(-1)): first optional flip(-1), then rot90(-k)
        if do_flip:
            pb = torch.flip(pb, dims=(-1,))
        if rot_k:
            pb = torch.rot90(pb, k=(4 - rot_k) & 3, dims=(-2, -1))
        prob_acc += torch.cat([pb.reshape(m, n * n), pp], dim=1)

    prob_mean = (prob_acc / float(k_syms)).clamp(min=1e-12)
    log_prob_mean = torch.log(prob_mean).numpy()
    return log_prob_mean, v_mean


def warmup_tpu_buckets(net: nn.Module):
    """Pre-compiles fixed XLA inference bucket shapes once on TPU."""
    if not IS_TPU:
        return
    print(f"[TPU XLA] Pre-compiling fixed MCTS inference buckets {TPU_BATCH_BUCKETS} (one-time warmup)...")
    b_size = net.board_size
    was_training = net.training
    net.eval()
    t0 = time.time()
    for b in TPU_BATCH_BUCKETS:
        dummy = np.zeros((b, IN_PLANES, b_size, b_size), dtype=np.float32)
        _ = forward_infer_batch(net, dummy)
    if was_training:
        net.train()
    print(f"[TPU XLA] Bucket compilation complete in {time.time() - t0:.1f}s.")


def unwrap_model(net: nn.Module) -> nn.Module:
    """Unwraps torch.compile OptimizedModule if compiled."""
    return net._orig_mod if hasattr(net, "_orig_mod") else net


def get_cpu_state_dict(net: nn.Module) -> dict:
    """Returns a clean CPU-resident state_dict portable across CUDA, MPS, TPU, and CPU."""
    if IS_TPU and xm is not None:
        xm.mark_step()
    raw_net = unwrap_model(net)
    return {k: v.detach().cpu().contiguous() for k, v in raw_net.state_dict().items()}


def get_cpu_optim_state_dict(optimizer: optim.Optimizer) -> dict:
    if IS_TPU and xm is not None:
        xm.mark_step()
    opt_sd = copy.deepcopy(optimizer.state_dict())
    for state in opt_sd.get("state", {}).values():
        for k, v in list(state.items()):
            if isinstance(v, torch.Tensor):
                state[k] = v.detach().cpu()
    return opt_sd


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# ====================================================
#    Progress Bar Helpers
# ====================================================
def print_progress_bar(completed, total, start_time, prefix="Self-Play", bar_length=28):
    pct = completed / total if total > 0 else 1.0
    filled = int(bar_length * pct)
    bar = "█" * filled + "░" * (bar_length - filled)

    elapsed = max(0.001, time.time() - start_time)
    speed = completed / elapsed
    remaining = max(0, total - completed)
    eta_sec = int(remaining / speed) if speed > 0 else 0

    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(int(elapsed)))
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

    output = f"\r  {prefix}: [{bar}] {completed}/{total} ({pct*100:5.1f}%) | {speed:.3f} games/s | Elapsed: {elapsed_str} | ETA: {eta_str}"
    sys.stdout.write(output)
    sys.stdout.flush()
    if completed >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def print_train_progress(step, total_steps, loss, p_loss, v_loss, o_loss, entropy, start_time, bar_length=20):
    pct = step / total_steps if total_steps > 0 else 1.0
    filled = int(bar_length * pct)
    bar = "█" * filled + "░" * (bar_length - filled)
    elapsed = max(0.001, time.time() - start_time)
    speed = step / elapsed
    output = (
        f"\r  Train [{bar}] {step}/{total_steps} ({pct*100:4.1f}%) | "
        f"Loss: {loss:.3f} (P:{p_loss:.3f} V:{v_loss:.3f} O:{o_loss:.3f} H:{entropy:.2f}) | {speed:4.1f} step/s"
    )
    sys.stdout.write(output)
    sys.stdout.flush()
    if step >= total_steps:
        sys.stdout.write("\n")
        sys.stdout.flush()


# ====================================================
#    MCTS Node, Parent-Relative FPU & Dynamic PUCT
# ====================================================
class NN_MCTS_Node:
    __slots__ = (
        "game_state",
        "parent",
        "move",
        "children",
        "is_expanded",
        "N",
        "W",
        "Q",
        "virtual_loss",
        "raw_P",
        "P",
        "value",
        "visited_prior_sum",
        "is_root_noised",
    )

    def __init__(self, game_state: GoGame = None, parent=None, move=None, prior: float = 0.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.is_expanded = False

        # MCTS statistics (W and Q are stored from parent's player perspective; -node.Q is current player's value)
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.virtual_loss = 0.0
        self.raw_P = float(prior)  # Clean neural network prior (preserved across subtree reuse)
        self.P = float(prior)      # Active prior (may include root Dirichlet noise)
        self.value = 0.0
        self.visited_prior_sum = 0.0
        self.is_root_noised = False

    def ensure_game_state(self) -> GoGame:
        """Lazily materializes the GoGame board state only when the node is visited."""
        if self.game_state is None and self.parent is not None:
            ng = self.parent.ensure_game_state().copy()
            ng.make_move(*self.move, check_valid=False)
            self.game_state = ng
        return self.game_state


def apply_root_dirichlet_noise(root: NN_MCTS_Node, dirichlet_alpha: float, epsilon: float = DIRICHLET_EPSILON):
    """
    Mixes Dirichlet exploration noise into root priors while excluding Pass (-1, -1)
    whenever at least one legal board move exists, preventing noise-induced double passes.
    """
    if not root.children:
        return
    # Reset priors to clean neural priors first (important when reusing subtrees)
    for ch in root.children.values():
        ch.P = ch.raw_P

    board_moves = [mv for mv in root.children.keys() if mv != (-1, -1)]
    if not board_moves:
        root.is_root_noised = False
        return

    pass_ch = root.children.get((-1, -1))
    pass_prior = pass_ch.raw_P if pass_ch is not None else 0.0
    board_mass = max(1e-8, 1.0 - pass_prior)

    noise = np.random.dirichlet([dirichlet_alpha] * len(board_moves))
    for i, mv in enumerate(board_moves):
        ch = root.children[mv]
        ch.P = (1.0 - epsilon) * ch.raw_P + epsilon * board_mass * float(noise[i])
    if pass_ch is not None:
        pass_ch.P = pass_prior
    root.is_root_noised = True


def advance_mcts_root(root: NN_MCTS_Node, move, new_game_state: GoGame) -> NN_MCTS_Node:
    """
    Advances the MCTS search tree to `root.children[move]` for subtree reuse,
    detaching the parent reference and restoring clean neural network priors.
    """
    if root is not None and root.children and move in root.children:
        next_root = root.children[move]
        next_root.parent = None
        next_root.game_state = new_game_state.copy()
        next_root.is_root_noised = False
        for ch in next_root.children.values():
            ch.P = ch.raw_P
        return next_root
    return NN_MCTS_Node(new_game_state.copy())


def mcts_select(root: NN_MCTS_Node):
    """
    Selects a leaf node using AlphaZero dynamic c_puct(s) and KataGo Parent-Relative FPU:
      fpu_val = clamp(-node.Q - c_fpu * sqrt(visited_prior_mass), -1.0, +1.0)
    """
    node = root
    search_path = [node]
    while node.is_expanded and node.children:
        best_score = -1e30
        best_child = None

        parent_tot_n = max(1.0, node.N + node.virtual_loss)
        parent_n_sqrt = math.sqrt(parent_tot_n)
        c_puct = C_PUCT + math.log((1.0 + parent_tot_n + C_PUCT_BASE) / C_PUCT_BASE)

        # Parent-relative First-Play Urgency (FPU), using C_FPU_ROOT at all root nodes
        # and clamping to [-1.0, +1.0] so unvisited board moves never score below -1.0 (certain loss).
        raw_parent_v = -node.Q if node.N > 0 else node.value
        parent_v_for_cur = raw_parent_v if math.isfinite(raw_parent_v) else 0.0
        c_fpu = C_FPU_ROOT if (node.parent is None or node.is_root_noised) else C_FPU_INTERNAL
        vis_mass = min(1.0, max(0.0, node.visited_prior_sum))
        fpu_val = max(-1.0, min(1.0, parent_v_for_cur - c_fpu * math.sqrt(vis_mass)))

        cpuct_sqrt = c_puct * parent_n_sqrt
        for ch in node.children.values():
            total_n = ch.N + ch.virtual_loss
            prior_p = ch.P if math.isfinite(ch.P) else 0.0
            u_val = cpuct_sqrt * prior_p / (1.0 + total_n)
            q_val = (ch.W - ch.virtual_loss) / total_n if total_n > 0.0 else fpu_val
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_child = ch

        if best_child is None:
            best_child = next(iter(node.children.values()))
        node = best_child
        search_path.append(node)

    node.ensure_game_state()
    return node, search_path


# Worker Shared-Memory Globals
_WORKER_REQ_QUEUE = None
_WORKER_RES_QUEUES = None
_WORKER_PROGRESS_QUEUE = None
_WORKER_SHM_REQ = None
_WORKER_SHM_RES = None
_WORKER_SHM_REQ_ARR = None
_WORKER_SHM_RES_ARR = None


def _open_shared_memory_untracked(name: str):
    """Opens an existing SharedMemory block without double-tracking in Python 3.13+ resource_tracker."""
    if sys.version_info >= (3, 13):
        return shared_memory.SharedMemory(name=name, create=False, track=False)
    return shared_memory.SharedMemory(name=name, create=False)


def _cleanup_worker_shm():
    global _WORKER_SHM_REQ, _WORKER_SHM_RES, _WORKER_SHM_REQ_ARR, _WORKER_SHM_RES_ARR
    _WORKER_SHM_REQ_ARR = None
    _WORKER_SHM_RES_ARR = None
    if _WORKER_SHM_REQ is not None:
        try:
            _WORKER_SHM_REQ.close()
        except Exception:
            pass
        _WORKER_SHM_REQ = None
    if _WORKER_SHM_RES is not None:
        try:
            _WORKER_SHM_RES.close()
        except Exception:
            pass
        _WORKER_SHM_RES = None


def _init_self_play_worker(
    req_queue,
    res_queues,
    progress_queue,
    shm_req_name=None,
    shm_res_name=None,
    num_workers=0,
    slot_batch=MCTS_BATCH_SIZE,
    board_size=BOARD_SIZE,
):
    """Initializes OS queues and attaches pre-allocated uint8/float32 shared memory slots in each worker."""
    global _WORKER_REQ_QUEUE, _WORKER_RES_QUEUES, _WORKER_PROGRESS_QUEUE
    global _WORKER_SHM_REQ, _WORKER_SHM_RES, _WORKER_SHM_REQ_ARR, _WORKER_SHM_RES_ARR
    _WORKER_REQ_QUEUE = req_queue
    _WORKER_RES_QUEUES = res_queues
    _WORKER_PROGRESS_QUEUE = progress_queue
    torch.set_num_threads(1)

    if shm_req_name and shm_res_name and num_workers > 0:
        _WORKER_SHM_REQ = _open_shared_memory_untracked(shm_req_name)
        _WORKER_SHM_RES = _open_shared_memory_untracked(shm_res_name)
        _WORKER_SHM_REQ_ARR = np.ndarray(
            (num_workers, slot_batch, IN_PLANES, board_size, board_size),
            dtype=np.uint8,
            buffer=_WORKER_SHM_REQ.buf,
        )
        _WORKER_SHM_RES_ARR = np.ndarray(
            (num_workers, slot_batch, board_size * board_size + 2),
            dtype=np.float32,
            buffer=_WORKER_SHM_RES.buf,
        )
        atexit.register(_cleanup_worker_shm)


def nn_batch_evaluate_and_expand(
    nodes,
    net: nn.Module = None,
    worker_id: int = None,
    req_queue=None,
    res_queue=None,
    num_symmetries: int = 1,
):
    """
    Evaluates leaf nodes with the neural network (via direct GPU call, symmetry ensemble,
    or zero-pickle shared-memory worker slots) and expands legal children.
    """
    eval_nodes = []
    for nd in nodes:
        if nd.is_expanded:
            continue
        nd.ensure_game_state()
        if nd.game_state.is_game_over():
            w = nd.game_state.get_winner()
            if w == 0:
                nd.value = 0.0
            elif w == nd.game_state.current_player:
                nd.value = 1.0
            else:
                nd.value = -1.0
            nd.is_expanded = True
        else:
            eval_nodes.append(nd)

    if not eval_nodes:
        return

    n_eval = len(eval_nodes)
    if net is not None:
        batch = np.empty((n_eval, IN_PLANES, eval_nodes[0].game_state.board_size, eval_nodes[0].game_state.board_size), dtype=np.uint8)
        for i, nd in enumerate(eval_nodes):
            nd.game_state.write_nn_input_uint8(batch[i])
        if num_symmetries > 1:
            policy_logits, value_out = ensemble_infer_symmetries(net, batch.astype(np.float32), num_symmetries=num_symmetries)
        else:
            policy_logits, value_out = forward_infer_batch(net, batch)
    elif _WORKER_SHM_REQ_ARR is not None and worker_id is not None and n_eval <= _WORKER_SHM_REQ_ARR.shape[1]:
        slot_in = _WORKER_SHM_REQ_ARR[worker_id]
        for i, nd in enumerate(eval_nodes):
            nd.game_state.write_nn_input_uint8(slot_in[i])
        req_queue.put(("SHM", worker_id, n_eval))
        _ = res_queue.get()
        slot_out = _WORKER_SHM_RES_ARR[worker_id, :n_eval]
        policy_logits = slot_out[:, :-1].copy()
        value_out = slot_out[:, -1].copy()
    else:
        batch = [nd.game_state.get_nn_input() for nd in eval_nodes]
        req_queue.put((worker_id, batch))
        policy_logits, value_out = res_queue.get()

    board_s = eval_nodes[0].game_state.board_size
    pass_idx = board_s * board_s
    for i, nd in enumerate(eval_nodes):
        v_raw = float(value_out[i])
        nd.value = max(-1.0, min(1.0, v_raw)) if math.isfinite(v_raw) else 0.0
        legal_moves = nd.game_state.get_legal_moves()

        move_indices = [(pass_idx if mv == (-1, -1) else mv[0] * board_s + mv[1]) for mv in legal_moves]
        if move_indices:
            selected_logits = policy_logits[i, move_indices]
            if np.all(np.isfinite(selected_logits)):
                probs = softmax(selected_logits)
                if not np.all(np.isfinite(probs)):
                    probs = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)
            else:
                probs = np.full(len(legal_moves), 1.0 / len(legal_moves), dtype=np.float32)
        else:
            probs = ()

        children_dict = nd.children
        for mv, p in zip(legal_moves, probs):
            children_dict[mv] = NN_MCTS_Node(game_state=None, parent=nd, move=mv, prior=float(p))
        nd.is_expanded = True


def apply_virtual_loss(search_path, loss: float = 1.0):
    for node in search_path:
        node.virtual_loss += loss


def remove_virtual_loss(search_path, loss: float = 1.0):
    for node in search_path:
        node.virtual_loss -= loss


def mcts_backup(search_path, value: float):
    """Backpropagates zero-sum inverted value up the search path and tracks visited prior mass."""
    curr_v = -value
    for node in reversed(search_path):
        was_unvisited = (node.N == 0)
        node.N += 1
        node.W += curr_v
        node.Q = node.W / node.N
        if was_unvisited and node.parent is not None:
            node.parent.visited_prior_sum += node.raw_P
        curr_v = -curr_v


def extract_mcts_policy_target(root: NN_MCTS_Node, prune_single_visit_noise: bool = True) -> dict:
    """
    Extracts normalized visit distribution `visits / sum(visits)` as the soft policy training target,
    decoupled from move-selection temperature, with lightweight single-visit noise pruning.
    """
    if not root.children:
        return {}
    moves = list(root.children.keys())
    visits = np.array([root.children[mv].N for mv in moves], dtype=np.float32)
    max_v = float(np.max(visits))
    if max_v <= 0.0:
        n_m = len(moves)
        return {mv: 1.0 / n_m for mv in moves}

    if prune_single_visit_noise and root.is_root_noised and max_v >= 16.0:
        best_idx = int(np.argmax(visits))
        for i, mv in enumerate(moves):
            if i != best_idx and visits[i] <= 1.0 and root.children[mv].raw_P < 0.05:
                visits[i] = 0.0

    v_sum = float(np.sum(visits))
    if v_sum > 0.0 and np.isfinite(v_sum):
        probs = visits / v_sum
    else:
        probs = np.zeros_like(visits)
        probs[int(np.argmax(visits))] = 1.0
    return dict(zip(moves, probs))


def sample_move_with_temperature(pi_target_dict: dict, move_count: int, board_size: int, greedy: bool = False):
    """
    Applies smooth opening temperature decay (1.0 -> 0.5 -> 0.15 -> greedy)
    independently of the stored soft policy training targets.
    """
    moves = list(pi_target_dict.keys())
    probs = np.array([pi_target_dict[mv] for mv in moves], dtype=np.float64)
    if greedy:
        return moves[int(np.argmax(probs))]

    t_thresh = get_temp_threshold(board_size)
    if move_count < t_thresh:
        temp = 1.0 - 0.5 * (move_count / max(1, t_thresh))
    elif move_count < int(1.5 * t_thresh):
        frac = (move_count - t_thresh) / max(1.0, 0.5 * t_thresh)
        temp = 0.5 - 0.35 * frac
    else:
        temp = 0.0

    if temp <= 0.08:
        max_p = np.max(probs)
        best_candidates = [moves[i] for i, p in enumerate(probs) if p >= max_p * 0.999]
        return random.choice(best_candidates)

    log_p = np.where(probs > 1e-12, np.log(np.maximum(probs, 1e-12)) / temp, -1e9)
    log_p -= np.max(log_p)
    scaled = np.exp(log_p)
    scaled[probs <= 1e-12] = 0.0
    s_sum = np.sum(scaled)
    if s_sum <= 0.0 or not np.isfinite(s_sum):
        return moves[int(np.argmax(probs))]
    scaled /= s_sum
    return random.choices(moves, weights=scaled.tolist(), k=1)[0]


def mcts_run(
    root: NN_MCTS_Node,
    net: nn.Module,
    n_simulations: int = 50,
    temp: float = None,
    batch_size: int = MCTS_BATCH_SIZE,
    worker_id: int = None,
    req_queue=None,
    res_queue=None,
    num_symmetries: int = 1,
) -> dict:
    """
    Runs batched PUCT MCTS simulations from `root`.
    When `temp is None` (default V2 behavior), returns clean normalized visit policy `pi_target`.
    If an explicit numeric `temp` is passed (legacy/greedy callers), returns temperature-shaped probabilities.
    """
    sims = 0
    pump_pygame = (pygame is not None and pygame.get_init())
    while sims < n_simulations:
        batch_paths = []
        batch_leaves = []
        seen_leaves = set()

        current_batch_size = min(batch_size, n_simulations - sims)
        for _ in range(current_batch_size):
            leaf, path = mcts_select(root)
            if leaf in seen_leaves:
                break
            seen_leaves.add(leaf)
            apply_virtual_loss(path)
            batch_paths.append(path)
            batch_leaves.append(leaf)

        if batch_leaves:
            nn_batch_evaluate_and_expand(
                batch_leaves,
                net=net,
                worker_id=worker_id,
                req_queue=req_queue,
                res_queue=res_queue,
                num_symmetries=num_symmetries,
            )

            for path, leaf in zip(batch_paths, batch_leaves):
                remove_virtual_loss(path)
                mcts_backup(path, leaf.value)

            sims += len(batch_leaves)
            if pump_pygame:
                pygame.event.pump()
        else:
            sims += 1

    pi_target = extract_mcts_policy_target(root, prune_single_visit_noise=True)
    if not pi_target or temp is None:
        return pi_target

    moves = list(pi_target.keys())
    visits = np.array([root.children[mv].N for mv in moves], dtype=np.float32)
    if temp <= 0.05:
        probs = np.zeros_like(visits)
        probs[int(np.argmax(visits))] = 1.0
    else:
        v = visits ** (1.0 / temp)
        v_sum = float(np.sum(v))
        if v_sum > 0 and np.isfinite(v_sum):
            probs = v / v_sum
        else:
            probs = np.zeros_like(visits)
            probs[int(np.argmax(visits))] = 1.0
    return dict(zip(moves, probs))


# ====================================================
#   Shared-Memory GPU Server & Self-Play Generation
# ====================================================
def gpu_server_thread(
    net: nn.Module,
    req_queue,
    res_queues,
    total_workers: int,
    shm_req_arr: np.ndarray = None,
    shm_res_arr: np.ndarray = None,
    board_size: int = BOARD_SIZE,
):
    """
    Central GPU inference server coalescing leaf requests across all CPU workers
    from zero-pickle SharedMemory uint8 slots into a single batched forward pass.
    """
    active_workers = total_workers
    coalesce_window_sec = 0.001 if IS_TPU else 0.00035
    max_coalesce = GPU_SERVER_BATCH_SIZE + MCTS_BATCH_SIZE * max(1, total_workers)
    staging_u8 = np.empty((max_coalesce, IN_PLANES, board_size, board_size), dtype=np.uint8)

    while active_workers > 0:
        requests = []
        total_states = 0
        try:
            req = req_queue.get(timeout=0.1)
            if req == "DONE":
                active_workers -= 1
                continue
            requests.append(req)
            total_states += req[2] if (isinstance(req, tuple) and len(req) == 3 and req[0] == "SHM") else len(req[1])
            waited_coalesce = False
            while total_states < GPU_SERVER_BATCH_SIZE:
                try:
                    req = req_queue.get_nowait()
                    if req == "DONE":
                        active_workers -= 1
                        continue
                    requests.append(req)
                    total_states += req[2] if (isinstance(req, tuple) and len(req) == 3 and req[0] == "SHM") else len(req[1])
                except queue.Empty:
                    if not waited_coalesce and active_workers > 1 and total_states < 64:
                        waited_coalesce = True
                        try:
                            req = req_queue.get(timeout=coalesce_window_sec)
                            if req == "DONE":
                                active_workers -= 1
                                continue
                            requests.append(req)
                            total_states += req[2] if (isinstance(req, tuple) and len(req) == 3 and req[0] == "SHM") else len(req[1])
                            continue
                        except queue.Empty:
                            break
                    break
        except queue.Empty:
            continue

        if not requests:
            continue

        # Check if all requests use the fast SharedMemory uint8 slot protocol
        if shm_req_arr is not None and all(isinstance(r, tuple) and len(r) == 3 and r[0] == "SHM" for r in requests):
            offset = 0
            for _, w_id, cnt in requests:
                staging_u8[offset : offset + cnt] = shm_req_arr[w_id, :cnt]
                offset += cnt

            policy_logits, value_out = forward_infer_batch(net, staging_u8[:offset])

            offset = 0
            for _, w_id, cnt in requests:
                shm_res_arr[w_id, :cnt, :-1] = policy_logits[offset : offset + cnt]
                shm_res_arr[w_id, :cnt, -1] = value_out[offset : offset + cnt]
                res_queues[w_id].put(cnt)
                offset += cnt
        else:
            all_inputs = []
            counts = []
            for r in requests:
                if isinstance(r, tuple) and len(r) == 3 and r[0] == "SHM":
                    _, w_id, cnt = r
                    all_inputs.extend(shm_req_arr[w_id, :cnt].astype(np.float32))
                    counts.append(("SHM", w_id, cnt))
                else:
                    w_id, inp_list = r
                    all_inputs.extend(inp_list)
                    counts.append(("QUEUE", w_id, len(inp_list)))

            policy_logits, value_out = forward_infer_batch(net, all_inputs)

            offset = 0
            for mode_tag, w_id, cnt in counts:
                p = policy_logits[offset : offset + cnt]
                v = value_out[offset : offset + cnt]
                if mode_tag == "SHM":
                    shm_res_arr[w_id, :cnt, :-1] = p
                    shm_res_arr[w_id, :cnt, -1] = v
                    res_queues[w_id].put(cnt)
                else:
                    res_queues[w_id].put((p, v))
                offset += cnt


def self_play_wrapper(board_size: int, n_mcts_sims: int, worker_id: int, num_games: int, req_queue=None, res_queue=None, progress_queue=None):
    if req_queue is None:
        req_queue = _WORKER_REQ_QUEUE
    if res_queue is None and _WORKER_RES_QUEUES is not None:
        res_queue = _WORKER_RES_QUEUES[worker_id]
    if progress_queue is None:
        progress_queue = _WORKER_PROGRESS_QUEUE

    all_states = []
    agg_stats = {
        "games": 0,
        "resigned_games": 0,
        "full_playout_games": 0,
        "false_resign_checks": 0,
        "false_resign_count": 0,
        "total_moves": 0,
        "policy_entropy_sum": 0.0,
        "policy_entropy_cnt": 0,
    }
    for g in range(num_games):
        game_states, g_stats = self_play_one_game(
            net=None,
            board_size=board_size,
            n_mcts_sims=n_mcts_sims,
            game_idx=g + 1,
            worker_id=worker_id,
            req_queue=req_queue,
            res_queue=res_queue,
            return_stats=True,
        )
        all_states.extend(game_states)
        for k in agg_stats:
            agg_stats[k] += g_stats[k]
        if progress_queue is not None:
            progress_queue.put(1)
    return all_states, agg_stats


def self_play_one_game(
    net: nn.Module = None,
    board_size: int = BOARD_SIZE,
    n_mcts_sims: int = N_MCTS_SIMS,
    temp: float = 1.0,
    game_idx: int = 1,
    worker_id: int = None,
    req_queue=None,
    res_queue=None,
    return_stats: bool = False,
):
    """
    Plays one self-play game with:
      - KataGo Playout Cap Oscillations (25% full search + Dirichlet + policy_weight=1.0; 75% fast search)
      - MCTS Subtree Reuse across turns
      - Smooth opening temperature decay decoupled from soft pi_target
      - Calibrated 5-ply resignation disabled in 25% of games for exact endgame ownership & false-resign tracking
    """
    game = GoGame(board_size)
    trajectory = []  # (packed_u16, current_player, pi_flat, policy_weight)
    move_count = 0
    max_moves = board_size * board_size * 2
    dirichlet_alpha = get_dirichlet_alpha(board_size)

    fast_sims = min(n_mcts_sims, max(FAST_SEARCH_MIN_SIMS, int(round(FAST_SEARCH_RATIO * n_mcts_sims))))
    if n_mcts_sims < 20:
        fast_sims = max(2, n_mcts_sims)

    playout_full_game = (random.random() < FULL_PLAYOUT_GAME_PROB)
    lose_streak = {1: 0, 2: 0}
    resigned_winner = None
    would_have_resigned_winner = None
    entropy_sum = 0.0
    entropy_cnt = 0

    root = NN_MCTS_Node(game.copy())
    min_pass_moves = max(16, (board_size * board_size) // 4)

    while not game.is_game_over() and move_count < max_moves:
        # Ensure at least move 0 or 25% of moves execute full search
        is_full_search = (move_count == 0) or (n_mcts_sims < 20) or (random.random() < FULL_SEARCH_PROB)
        target_sims = n_mcts_sims if is_full_search else fast_sims
        # Mask policy loss on post-resignation playout-calibration plies so hopeless losing states
        # never concentrate PUCT visits onto Pass and inflate policy_pass_fc.
        policy_weight = 1.0 if (is_full_search and would_have_resigned_winner is None) else 0.0

        # On fast-search turns, avoid letting an already huge reused child skip fresh evaluation;
        # on full-search turns, reuse subtree and run enough fresh Dirichlet-noised simulations.
        if not is_full_search and root.N >= target_sims:
            root = NN_MCTS_Node(game.copy())

        if not root.is_expanded:
            nn_batch_evaluate_and_expand([root], net=net, worker_id=worker_id, req_queue=req_queue, res_queue=res_queue)
            root.N = 1
            root.W = -root.value
            root.Q = -root.value

        if is_full_search:
            apply_root_dirichlet_noise(root, dirichlet_alpha=dirichlet_alpha, epsilon=DIRICHLET_EPSILON)
            sims_to_run = max(int(0.75 * target_sims), target_sims - root.N + 1) if target_sims >= 8 else target_sims
        else:
            root.is_root_noised = False
            for ch in root.children.values():
                ch.P = ch.raw_P
            sims_to_run = max(1, target_sims - root.N + 1)

        # Prevent premature opening Pass collapse before min_pass_moves when board moves exist
        suppress_opening_pass = (
            move_count < min_pass_moves
            and game.passes_in_a_row == 0
            and (-1, -1) in root.children
            and len(root.children) > 1
        )
        if suppress_opening_pass:
            root.children[(-1, -1)].P = 0.0
            b_sum = sum(ch.P for mv, ch in root.children.items() if mv != (-1, -1))
            if b_sum > 1e-8:
                inv_b = 1.0 / b_sum
                for mv, ch in root.children.items():
                    if mv != (-1, -1):
                        ch.P *= inv_b
            else:
                uni_b = 1.0 / float(len(root.children) - 1)
                for mv, ch in root.children.items():
                    if mv != (-1, -1):
                        ch.P = uni_b

        pi_target_dict = mcts_run(
            root,
            net=net,
            n_simulations=sims_to_run,
            temp=None,
            batch_size=MCTS_BATCH_SIZE,
            worker_id=worker_id,
            req_queue=req_queue,
            res_queue=res_queue,
        )
        if not pi_target_dict:
            break

        if suppress_opening_pass and (-1, -1) in pi_target_dict and len(pi_target_dict) > 1:
            pi_target_dict[(-1, -1)] = 0.0
            b_prob_sum = sum(p for mv, p in pi_target_dict.items() if mv != (-1, -1))
            if b_prob_sum > 1e-8:
                for mv in list(pi_target_dict.keys()):
                    if mv != (-1, -1):
                        pi_target_dict[mv] /= b_prob_sum
            else:
                uni_p = 1.0 / float(len(pi_target_dict) - 1)
                for mv in list(pi_target_dict.keys()):
                    pi_target_dict[mv] = 0.0 if mv == (-1, -1) else uni_p

        # Evaluate calibrated resignation condition (root value AND best-child value < -0.90 for 5 consecutive plies)
        cp = game.current_player
        opp = 3 - cp
        root_val_cp = -root.Q
        best_ch = max(root.children.values(), key=lambda c: c.N)
        best_ch_val_cp = best_ch.Q if best_ch.N > 0 else root_val_cp

        if root_val_cp < RESIGN_THRESHOLD and best_ch_val_cp < RESIGN_THRESHOLD:
            lose_streak[cp] += 1
        else:
            lose_streak[cp] = 0

        if root_val_cp > -RESIGN_THRESHOLD and best_ch_val_cp > -RESIGN_THRESHOLD:
            lose_streak[opp] += 1
        else:
            lose_streak[opp] = 0

        if move_count >= board_size and lose_streak[cp] >= RESIGN_CONSECUTIVE_PLIES:
            if would_have_resigned_winner is None:
                would_have_resigned_winner = opp
            if not playout_full_game:
                resigned_winner = opp
                break
            policy_weight = 0.0

        pi_flat = np.zeros(board_size * board_size + 1, dtype=np.float32)
        pass_idx = board_size * board_size
        for mv, p in pi_target_dict.items():
            idx = pass_idx if (mv[0] == -1 and mv[1] == -1) else (mv[0] * board_size + mv[1])
            pi_flat[idx] = p

        if is_full_search:
            nz = pi_flat[pi_flat > 1e-8]
            entropy_sum += float(-np.sum(nz * np.log(nz)))
            entropy_cnt += 1

        selected_move = sample_move_with_temperature(pi_target_dict, move_count=move_count, board_size=board_size)
        packed_u16, cplayer = game.get_packed_state()
        trajectory.append((packed_u16, cplayer, pi_flat, policy_weight))

        game.make_move(*selected_move, check_valid=False)
        move_count += 1

        # Advance MCTS subtree for next ply (used when continuing into full search or low-visit fast search)
        root = advance_mcts_root(root, selected_move, game)

    played_to_double_pass = (resigned_winner is None and game.is_game_over())
    if resigned_winner is not None:
        winner = resigned_winner
        own_black_u8 = np.ones((board_size, board_size), dtype=np.uint8)
        own_weight = 0.0
    else:
        dead = game.auto_detect_dead_stones()
        score_info = game.get_score_breakdown(dead_stones=dead)
        winner = score_info["winner"]
        if played_to_double_pass:
            own_black_f = game.get_ownership_target(player=1, dead_stones=dead)
            # Encode {0.0, 0.5, 1.0} losslessly into uint8 {0, 1, 2}
            own_black_u8 = np.rint(own_black_f * 2.0).astype(np.uint8)
            own_weight = 1.0
        else:
            own_black_u8 = np.ones((board_size, board_size), dtype=np.uint8)
            own_weight = 0.0

    own_white_u8 = (2 - own_black_u8).astype(np.uint8)

    final_data = []
    for packed_u16, player, pi_flat, p_weight in trajectory:
        if winner == 0:
            z = 0.0
        elif winner == player:
            z = 1.0
        else:
            z = -1.0
        own_p = own_black_u8 if player == 1 else own_white_u8
        final_data.append((packed_u16, player, pi_flat, z, p_weight, own_p, own_weight))

    false_resign_checks = 1 if (playout_full_game and would_have_resigned_winner is not None) else 0
    false_resign_count = 1 if (false_resign_checks and winner != would_have_resigned_winner) else 0

    stats = {
        "games": 1,
        "resigned_games": 1 if resigned_winner is not None else 0,
        "full_playout_games": 1 if playout_full_game else 0,
        "false_resign_checks": false_resign_checks,
        "false_resign_count": false_resign_count,
        "total_moves": move_count,
        "policy_entropy_sum": entropy_sum,
        "policy_entropy_cnt": entropy_cnt,
    }
    if return_stats:
        return final_data, stats
    return final_data


def unpack_packed_state_np(packed_u16: np.ndarray, cplayer: int) -> np.ndarray:
    """CPU helper to unpack a single uint16[N, N] state into canonical float32[18, N, N]."""
    n = packed_u16.shape[0]
    planes = np.zeros((IN_PLANES, n, n), dtype=np.float32)
    for b in range(16):
        planes[b] = ((packed_u16 >> b) & 1).astype(np.float32)
    if cplayer == 1:
        planes[16] = 1.0
    planes[17] = 1.0
    return planes


def get_equi_data(play_data):
    """
    Explicit 8-fold dihedral augmentation helper (retained for tests / CPU inspection).
    During training, OceanGo V2 stores unique bit-packed states and applies 8x symmetry on GPU.
    """
    extend_data = []
    for item in play_data:
        if len(item) == 7:
            packed_u16, cplayer, pi, z, p_w, own_u8, own_w = item
            state = unpack_packed_state_np(packed_u16, cplayer)
        else:
            state, cplayer, pi, z = item
            p_w = 1.0
            own_u8 = np.ones((state.shape[-1], state.shape[-1]), dtype=np.uint8)
            own_w = 0.0

        board_size = state.shape[1]
        pi_board = np.reshape(pi[:-1], (board_size, board_size))
        pi_pass = pi[-1]

        for k in (0, 1, 2, 3):
            equi_state = np.rot90(state, k=k, axes=(1, 2))
            equi_pi_board = np.rot90(pi_board, k=k)
            equi_own = np.rot90(own_u8, k=k)

            equi_pi = np.empty_like(pi)
            equi_pi[:-1] = equi_pi_board.flatten()
            equi_pi[-1] = pi_pass
            extend_data.append((equi_state.copy(), cplayer, equi_pi.copy(), z, p_w, equi_own.copy(), own_w))

            equi_state_flip = np.flip(equi_state, axis=2)
            equi_pi_board_flip = np.fliplr(equi_pi_board)
            equi_own_flip = np.fliplr(equi_own)

            equi_pi_flip = np.empty_like(pi)
            equi_pi_flip[:-1] = equi_pi_board_flip.flatten()
            equi_pi_flip[-1] = pi_pass
            extend_data.append((equi_state_flip.copy(), cplayer, equi_pi_flip.copy(), z, p_w, equi_own_flip.copy(), own_w))

    return extend_data


# ====================================================
#   Bit-Packed Compact Replay Buffer (34x RAM Savings)
# ====================================================
class CompactReplayBuffer:
    """
    Contiguous ring buffer storing unique bit-packed uint16[N, N] board histories,
    float16 policies, and uint8 ownership maps (~0.9 GB RAM for 500,000 19x19 states).
    """
    def __init__(self, max_size: int = MAX_REPLAY_BUFFER_SIZE, board_size: int = BOARD_SIZE):
        self.max_size = max_size
        self.board_size = board_size
        self.reset(board_size)

    def reset(self, board_size: int = None):
        if board_size is not None:
            self.board_size = board_size
        self.capacity = 0
        self.size = 0
        self.write_ptr = 0
        self.packed_planes = None
        self.cplayer = None
        self.pi = None
        self.z = None
        self.p_weight = None
        self.own_u8 = None
        self.own_weight = None

    def __len__(self) -> int:
        return self.size

    def _ensure_capacity(self, needed_extra: int):
        target = min(self.max_size, max(self.size + needed_extra, 8192))
        if self.capacity >= target:
            return
        new_cap = max(8192, self.capacity if self.capacity > 0 else 8192)
        while new_cap < target:
            new_cap = min(self.max_size, new_cap * 2)

        n = self.board_size
        new_packed = np.empty((new_cap, n, n), dtype=np.uint16)
        new_cplayer = np.empty((new_cap,), dtype=np.uint8)
        new_pi = np.empty((new_cap, n * n + 1), dtype=np.float16)
        new_z = np.empty((new_cap,), dtype=np.float16)
        new_pw = np.empty((new_cap,), dtype=np.float16)
        new_own = np.empty((new_cap, n, n), dtype=np.uint8)
        new_ow = np.empty((new_cap,), dtype=np.float16)

        if self.size > 0:
            new_packed[: self.size] = self.packed_planes[: self.size]
            new_cplayer[: self.size] = self.cplayer[: self.size]
            new_pi[: self.size] = self.pi[: self.size]
            new_z[: self.size] = self.z[: self.size]
            new_pw[: self.size] = self.p_weight[: self.size]
            new_own[: self.size] = self.own_u8[: self.size]
            new_ow[: self.size] = self.own_weight[: self.size]

        self.packed_planes = new_packed
        self.cplayer = new_cplayer
        self.pi = new_pi
        self.z = new_z
        self.p_weight = new_pw
        self.own_u8 = new_own
        self.own_weight = new_ow
        self.capacity = new_cap

    def extend(self, items):
        if not items:
            return
        first = items[0]
        item_board = first[0].shape[-1]
        if item_board != self.board_size:
            self.reset(item_board)

        m = len(items)
        self._ensure_capacity(m)

        # Batch-pack incoming list into contiguous arrays before ring-buffer insertion
        n = self.board_size
        b_packed = np.empty((m, n, n), dtype=np.uint16)
        b_cp = np.empty((m,), dtype=np.uint8)
        b_pi = np.empty((m, n * n + 1), dtype=np.float16)
        b_z = np.empty((m,), dtype=np.float16)
        b_pw = np.empty((m,), dtype=np.float16)
        b_own = np.empty((m, n, n), dtype=np.uint8)
        b_ow = np.empty((m,), dtype=np.float16)

        for i, it in enumerate(items):
            if len(it) == 7 and it[0].ndim == 2:
                pk, cp, pi, z, pw, own, ow = it
            elif len(it) == 7 and it[0].ndim == 3:
                planes, cp, pi, z, pw, own, ow = it
                pk = np.zeros((n, n), dtype=np.uint16)
                for bit in range(16):
                    pk |= (planes[bit] > 0.5).astype(np.uint16) << bit
            else:
                planes, cp, pi, z = it[:4]
                pk = np.zeros((n, n), dtype=np.uint16)
                for bit in range(16):
                    pk |= (planes[bit] > 0.5).astype(np.uint16) << bit
                pw = 1.0
                own = np.ones((n, n), dtype=np.uint8)
                ow = 0.0

            b_packed[i] = pk
            b_cp[i] = cp
            b_pi[i] = pi
            b_z[i] = z
            b_pw[i] = pw
            b_own[i] = own
            b_ow[i] = ow

        if m >= self.max_size:
            b_packed = b_packed[-self.max_size :]
            b_cp = b_cp[-self.max_size :]
            b_pi = b_pi[-self.max_size :]
            b_z = b_z[-self.max_size :]
            b_pw = b_pw[-self.max_size :]
            b_own = b_own[-self.max_size :]
            b_ow = b_ow[-self.max_size :]
            m = self.max_size

        end_ptr = self.write_ptr + m
        if end_ptr <= self.capacity:
            sl = slice(self.write_ptr, end_ptr)
            self.packed_planes[sl] = b_packed
            self.cplayer[sl] = b_cp
            self.pi[sl] = b_pi
            self.z[sl] = b_z
            self.p_weight[sl] = b_pw
            self.own_u8[sl] = b_own
            self.own_weight[sl] = b_ow
        else:
            first_part = self.capacity - self.write_ptr
            second_part = m - first_part
            self.packed_planes[self.write_ptr :] = b_packed[:first_part]
            self.cplayer[self.write_ptr :] = b_cp[:first_part]
            self.pi[self.write_ptr :] = b_pi[:first_part]
            self.z[self.write_ptr :] = b_z[:first_part]
            self.p_weight[self.write_ptr :] = b_pw[:first_part]
            self.own_u8[self.write_ptr :] = b_own[:first_part]
            self.own_weight[self.write_ptr :] = b_ow[:first_part]

            self.packed_planes[:second_part] = b_packed[first_part:]
            self.cplayer[:second_part] = b_cp[first_part:]
            self.pi[:second_part] = b_pi[first_part:]
            self.z[:second_part] = b_z[first_part:]
            self.p_weight[:second_part] = b_pw[first_part:]
            self.own_u8[:second_part] = b_own[first_part:]
            self.own_weight[:second_part] = b_ow[first_part:]

        self.size = min(self.capacity, self.size + m)
        self.write_ptr = end_ptr % self.capacity

    def sample_batch_arrays(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.packed_planes[idx],
            self.cplayer[idx],
            self.pi[idx],
            self.z[idx],
            self.p_weight[idx],
            self.own_u8[idx],
            self.own_weight[idx],
        )

    def state_dict(self) -> dict:
        if self.size == 0:
            return {"version": 2, "board_size": self.board_size, "size": 0}
        return {
            "version": 2,
            "board_size": self.board_size,
            "size": self.size,
            "write_ptr": self.write_ptr,
            "packed_planes": self.packed_planes[: self.size].copy(),
            "cplayer": self.cplayer[: self.size].copy(),
            "pi": self.pi[: self.size].copy(),
            "z": self.z[: self.size].copy(),
            "p_weight": self.p_weight[: self.size].copy(),
            "own_u8": self.own_u8[: self.size].copy(),
            "own_weight": self.own_weight[: self.size].copy(),
        }

    def load_state_dict(self, sd: dict):
        if not isinstance(sd, dict) or sd.get("version") != 2:
            return False
        if sd.get("board_size") != self.board_size:
            return False
        sz = int(sd.get("size", 0))
        self.reset(self.board_size)
        if sz <= 0:
            return True
        self._ensure_capacity(sz)
        sz = min(sz, self.capacity)
        self.packed_planes[:sz] = sd["packed_planes"][:sz]
        self.cplayer[:sz] = sd["cplayer"][:sz]
        self.pi[:sz] = sd["pi"][:sz]
        self.z[:sz] = sd["z"][:sz]
        self.p_weight[:sz] = sd["p_weight"][:sz]
        self.own_u8[:sz] = sd["own_u8"][:sz]
        self.own_weight[:sz] = sd["own_weight"][:sz]
        self.size = sz
        self.write_ptr = int(sd.get("write_ptr", sz)) % self.capacity
        return True


REPLAY_BUFFER = CompactReplayBuffer(max_size=MAX_REPLAY_BUFFER_SIZE, board_size=BOARD_SIZE)


def add_to_replay_buffer(new_data):
    REPLAY_BUFFER.extend(new_data)


# ====================================================
#   Multi-Task Non-Saturating Loss & GPU Symmetry Train
# ====================================================
def build_optimizer(net: nn.Module, lr: float = LEARNING_RATE) -> optim.Optimizer:
    """
    Constructs SGD with Momentum (0.9) applying L2 weight decay (1e-4) strictly to
    Conv/Linear weight matrices while excluding BatchNorm scales/biases and 1D biases.
    """
    raw_net = unwrap_model(net)
    decay_params = []
    no_decay_params = []
    for name, param in raw_net.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "bn" not in name:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return optim.SGD(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        momentum=0.9,
    )


def _unpack_and_augment_on_device(
    packed_u16_np: np.ndarray,
    cplayer_np: np.ndarray,
    pi_np: np.ndarray,
    z_np: np.ndarray,
    pw_np: np.ndarray,
    own_u8_np: np.ndarray,
    ow_np: np.ndarray,
    shifts_dev: torch.Tensor,
    augment_8x: bool = True,
):
    """
    Transfers compact bit-packed arrays to the accelerator, applies stratified 8-fold
    dihedral rotation/reflection directly on GPU, and unpacks into [B, 18, N, N].
    """
    bsz, n, _ = packed_u16_np.shape

    def _to_dev(arr, dtype):
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device=device)
        if t.dtype != dtype:
            t = t.to(dtype=dtype)
        return t

    pk_i32 = _to_dev(packed_u16_np.astype(np.int32, copy=False), torch.int32)
    cp_t = _to_dev(cplayer_np, torch.int32)
    tgt_p = _to_dev(pi_np, torch.float32)
    tgt_z = _to_dev(z_np, torch.float32)
    p_w = _to_dev(pw_np, torch.float32)
    tgt_own = _to_dev(own_u8_np, torch.float32) * 0.5
    o_w = _to_dev(ow_np, torch.float32)

    if augment_8x and bsz >= 8:
        # Stratified uniform D4 symmetry across 8 contiguous slices (zero dynamic-shape overhead)
        perm = np.random.permutation(8)
        chunk = bsz // 8
        pk_parts, p_parts, o_parts = [], [], []
        for s_idx in range(8):
            st = s_idx * chunk
            ed = bsz if s_idx == 7 else (s_idx + 1) * chunk
            sid = int(perm[s_idx])
            pk_s, p_s, o_s = apply_dihedral_symmetry_torch(
                pk_i32[st:ed], tgt_p[st:ed], tgt_own[st:ed], sym_id=sid
            )
            pk_parts.append(pk_s)
            p_parts.append(p_s)
            o_parts.append(o_s)
        pk_i32 = torch.cat(pk_parts, dim=0)
        tgt_p = torch.cat(p_parts, dim=0)
        tgt_own = torch.cat(o_parts, dim=0)
    elif augment_8x:
        sid = random.randrange(8)
        pk_i32, tgt_p, tgt_own = apply_dihedral_symmetry_torch(pk_i32, tgt_p, tgt_own, sym_id=sid)

    planes_0_15 = ((pk_i32.unsqueeze(1) >> shifts_dev) & 1).to(torch.float32)
    plane_16 = (cp_t == 1).view(bsz, 1, 1, 1).expand(bsz, 1, n, n).to(torch.float32)
    plane_17 = torch.ones((bsz, 1, n, n), device=device, dtype=torch.float32)
    inp = torch.cat([planes_0_15, plane_16, plane_17], dim=1)

    if USE_CHANNELS_LAST:
        inp = inp.to(memory_format=torch.channels_last)

    return inp, tgt_p, tgt_z, p_w, tgt_own, o_w


def train_policy_value_net(
    net: nn.Module,
    data=None,
    batch_size: int = TRAIN_BATCH_SIZE,
    epochs: int = 1,
    lr: float = LEARNING_RATE,
    optimizer: optim.Optimizer = None,
    num_steps: int = None,
    replay_buffer: CompactReplayBuffer = None,
) -> dict:
    """
    Trains PolicyValueNet with the non-saturating multi-task objective:
      L = L_policy(masked CE) + 1.2 * L_value(BCEWithLogits) + 0.5 * L_ownership(masked BCEWithLogits) + 1e-4 * ||theta||^2
    Supports both direct sampling from `CompactReplayBuffer` (with GPU on-the-fly 8x symmetry)
    and explicit list `data` (used in unit/dry-run tests).
    """
    if optimizer is None:
        optimizer = build_optimizer(net, lr=lr)
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and USE_GRAD_SCALER))
    net.train()

    # If a raw list was passed, load into a temporary CompactReplayBuffer for unified GPU execution
    if replay_buffer is None:
        if isinstance(data, CompactReplayBuffer):
            buf = data
        else:
            if not data:
                net.eval()
                return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "ownership_loss": 0.0, "entropy": 0.0}
            first_b = data[0][0].shape[-1]
            buf = CompactReplayBuffer(max_size=max(len(data), 64), board_size=first_b)
            buf.extend(data)
    else:
        buf = replay_buffer

    if len(buf) == 0:
        net.eval()
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "ownership_loss": 0.0, "entropy": 0.0}

    eff_batch = min(batch_size, len(buf)) if not IS_TPU else batch_size
    if num_steps is None:
        steps_per_ep = max(1, (len(buf) + eff_batch - 1) // eff_batch)
        total_steps = max(1, epochs * steps_per_ep)
    else:
        total_steps = max(1, int(num_steps))

    shifts_dev = torch.arange(16, device=device, dtype=torch.int32).view(1, 16, 1, 1)
    start_time = time.time()
    raw_net = unwrap_model(net)
    last_good_state = get_cpu_state_dict(raw_net)

    tot_loss_sum = 0.0
    p_loss_sum = 0.0
    v_loss_sum = 0.0
    o_loss_sum = 0.0
    ent_sum = 0.0
    valid_steps = 0
    nan_skips = 0

    for step_idx in range(1, total_steps + 1):
        pk_np, cp_np, pi_np, z_np, pw_np, own_np, ow_np = buf.sample_batch_arrays(eff_batch)
        inp, tgt_p, tgt_z, p_w, tgt_own, o_w = _unpack_and_augment_on_device(
            pk_np, cp_np, pi_np, z_np, pw_np, own_np, ow_np, shifts_dev=shifts_dev, augment_8x=True
        )
        # Convert outcome z in [-1, +1] to win probability target in [0, 1] for BCEWithLogitsLoss
        tgt_v_prob = 0.5 * (tgt_z + 1.0)

        optimizer.zero_grad(set_to_none=True)

        if IS_TPU:
            with torch.amp.autocast("xla", dtype=torch.bfloat16):
                out_p, out_v_logit, out_own_logit = net(inp, return_aux=True)
            logp = F.log_softmax(out_p.float(), dim=1)
            ce_terms = torch.where(tgt_p > 0.0, tgt_p * logp, torch.zeros_like(logp))
            ce_per = -torch.sum(ce_terms, dim=1)
            policy_loss = (ce_per * p_w).sum() / p_w.sum().clamp(min=1.0)

            value_loss = F.binary_cross_entropy_with_logits(out_v_logit.float().squeeze(-1), tgt_v_prob)

            own_bce = F.binary_cross_entropy_with_logits(out_own_logit.float(), tgt_own, reduction="none").mean(dim=(1, 2))
            ownership_loss = (own_bce * o_w).sum() / o_w.sum().clamp(min=1.0)

            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss + OWNERSHIP_LOSS_WEIGHT * ownership_loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            if torch.isfinite(loss) and torch.isfinite(grad_norm):
                if xm is not None:
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                raw_net.load_state_dict(last_good_state)
                nan_skips += 1
                continue
        else:
            # Keep loss computations strictly outside AMP autocast in pure float32
            with torch.amp.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                out_p, out_v_logit, out_own_logit = net(inp, return_aux=True)

            logp = F.log_softmax(out_p.float(), dim=1)
            ce_terms = torch.where(tgt_p > 0.0, tgt_p * logp, torch.zeros_like(logp))
            ce_per = -torch.sum(ce_terms, dim=1)
            policy_loss = (ce_per * p_w).sum() / p_w.sum().clamp(min=1.0)

            value_loss = F.binary_cross_entropy_with_logits(out_v_logit.float().squeeze(-1), tgt_v_prob)

            own_bce = F.binary_cross_entropy_with_logits(out_own_logit.float(), tgt_own, reduction="none").mean(dim=(1, 2))
            ownership_loss = (own_bce * o_w).sum() / o_w.sum().clamp(min=1.0)

            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss + OWNERSHIP_LOSS_WEIGHT * ownership_loss

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                raw_net.load_state_dict(last_good_state)
                nan_skips += 1
                logging.warning(f"Non-finite loss detected at step {step_idx}; rolled back BatchNorm/weights and skipped update.")
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    nan_skips += 1
                scaler.update()
                if not torch.isfinite(grad_norm):
                    raw_net.load_state_dict(last_good_state)
                    continue
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    raw_net.load_state_dict(last_good_state)
                    nan_skips += 1
                    logging.warning(f"Non-finite grad_norm detected at step {step_idx}; skipped optimizer step.")
                    continue
                optimizer.step()

        with torch.no_grad():
            probs_pred = torch.exp(logp.detach())
            ent_batch = float((-torch.sum(probs_pred * logp.detach(), dim=1).mean()).cpu().item())

        l_val = float(loss.detach().cpu().item())
        pl_val = float(policy_loss.detach().cpu().item())
        vl_val = float(value_loss.detach().cpu().item())
        ol_val = float(ownership_loss.detach().cpu().item())

        valid_steps += 1
        tot_loss_sum += l_val
        p_loss_sum += pl_val
        v_loss_sum += vl_val
        o_loss_sum += ol_val
        ent_sum += ent_batch

        if step_idx % 64 == 0:
            last_good_state = get_cpu_state_dict(raw_net)

        if step_idx % 10 == 0 or step_idx == total_steps:
            print_train_progress(step_idx, total_steps, l_val, pl_val, vl_val, ol_val, ent_batch, start_time)

    # Final integrity check across all parameters and BatchNorm running statistics
    if not all(torch.isfinite(t).all().item() for t in raw_net.state_dict().values()):
        logging.error("Non-finite weights/BatchNorm stats detected after training loop; restoring last finite snapshot!")
        print("\n  [Warning] Non-finite tensor detected after training; restored last clean model snapshot.")
        raw_net.load_state_dict(last_good_state)

    net.eval()
    denom = max(1, valid_steps)
    metrics = {
        "loss": tot_loss_sum / denom,
        "policy_loss": p_loss_sum / denom,
        "value_loss": v_loss_sum / denom,
        "ownership_loss": o_loss_sum / denom,
        "entropy": ent_sum / denom,
        "steps": valid_steps,
        "nan_skips": nan_skips,
    }
    logging.info(
        f"Train Metrics ({total_steps} steps): Total={metrics['loss']:.4f} | "
        f"Policy={metrics['policy_loss']:.4f} | Value(BCE)={metrics['value_loss']:.4f} | "
        f"Own(BCE)={metrics['ownership_loss']:.4f} | PolicyEntropy={metrics['entropy']:.3f}"
    )
    return metrics


# ====================================================
#   Lightweight Parallel Arena Evaluation & Elo
# ====================================================
def _generate_varied_2move_openings(board_size: int, num_pairs: int = 10):
    """Generates `num_pairs` balanced 2-move (Black, White) openings away from the 1st line."""
    lo = 2 if board_size >= 9 else 1
    hi = board_size - 1 - lo
    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    if len(candidates) < 2:
        candidates = [(r, c) for r in range(board_size) for c in range(board_size)]

    rng = random.Random(42 + board_size * 100 + num_pairs)
    openings = []
    attempts = 0
    while len(openings) < num_pairs and attempts < num_pairs * 20:
        attempts += 1
        m1, m2 = rng.sample(candidates, 2)
        if abs(m1[0] - m2[0]) + abs(m1[1] - m2[1]) >= 2:
            if (m1, m2) not in openings:
                openings.append((m1, m2))
    while len(openings) < num_pairs:
        m1, m2 = rng.sample(candidates, 2)
        openings.append((m1, m2))
    return openings


def evaluate_arena(
    candidate_net: nn.Module,
    ref_net: nn.Module,
    board_size: int = BOARD_SIZE,
    num_games: int = EVAL_GAMES,
    eval_sims: int = 100,
    intra_batch: int = 8,
) -> dict:
    """
    Runs `num_games` paired evaluation matches (half Candidate=Black, half Candidate=White
    from identical varied 2-move openings) using lockstep multi-game GPU batching.
    """
    candidate_net.eval()
    ref_net.eval()

    n_pairs = max(1, num_games // 2)
    total_games = n_pairs * 2
    openings = _generate_varied_2move_openings(board_size, num_pairs=n_pairs)

    games = []
    cand_colors = []
    roots = []
    active = []
    move_counts = []
    max_moves = board_size * board_size * 2

    for m1, m2 in openings:
        for cand_col in (1, 2):
            g = GoGame(board_size)
            g.make_move(*m1, check_valid=True)
            g.make_move(*m2, check_valid=True)
            games.append(g)
            cand_colors.append(cand_col)
            roots.append(NN_MCTS_Node(g.copy()))
            active.append(True)
            move_counts.append(2)

    def _batch_eval_for_model(model, node_list):
        if not node_list:
            return
        eval_nds = []
        for nd in node_list:
            if nd.is_expanded:
                continue
            nd.ensure_game_state()
            if nd.game_state.is_game_over():
                w = nd.game_state.get_winner()
                nd.value = 0.0 if w == 0 else (1.0 if w == nd.game_state.current_player else -1.0)
                nd.is_expanded = True
            else:
                eval_nds.append(nd)
        if not eval_nds:
            return
        b_arr = np.empty((len(eval_nds), IN_PLANES, board_size, board_size), dtype=np.uint8)
        for i, nd in enumerate(eval_nds):
            nd.game_state.write_nn_input_uint8(b_arr[i])
        p_logits, v_out = forward_infer_batch(model, b_arr)
        pass_idx = board_size * board_size
        for i, nd in enumerate(eval_nds):
            nd.value = float(v_out[i])
            legals = nd.game_state.get_legal_moves()
            m_idxs = [(pass_idx if mv == (-1, -1) else mv[0] * board_size + mv[1]) for mv in legals]
            probs = softmax(p_logits[i, m_idxs]) if m_idxs else ()
            for mv, pr in zip(legals, probs):
                nd.children[mv] = NN_MCTS_Node(game_state=None, parent=nd, move=mv, prior=float(pr))
            nd.is_expanded = True

    while any(active):
        active_indices = [idx for idx, is_act in enumerate(active) if is_act]

        # Expand any unexpanded roots in two coalesced batches (Candidate vs Reference)
        cand_unexp = []
        ref_unexp = []
        for idx in active_indices:
            rt = roots[idx]
            if not rt.is_expanded:
                if games[idx].current_player == cand_colors[idx]:
                    cand_unexp.append(rt)
                else:
                    ref_unexp.append(rt)
        _batch_eval_for_model(candidate_net, cand_unexp)
        _batch_eval_for_model(ref_net, ref_unexp)
        for rt in cand_unexp + ref_unexp:
            if rt.N == 0:
                rt.N = 1
                rt.W = -rt.value
                rt.Q = -rt.value

        # Run lockstep MCTS simulations across all active games
        sims_done = {idx: 0 for idx in active_indices}
        while True:
            need_sim = [idx for idx in active_indices if sims_done[idx] < eval_sims]
            if not need_sim:
                break

            cand_leaves, cand_paths = [], []
            ref_leaves, ref_paths = [], []
            game_step_counts = {idx: 0 for idx in need_sim}

            for idx in need_sim:
                rt = roots[idx]
                is_cand_turn = (games[idx].current_player == cand_colors[idx])
                seen_local = set()
                cur_b = min(intra_batch, eval_sims - sims_done[idx])
                for _ in range(cur_b):
                    leaf, path = mcts_select(rt)
                    if leaf in seen_local:
                        break
                    seen_local.add(leaf)
                    apply_virtual_loss(path)
                    if is_cand_turn:
                        cand_leaves.append(leaf)
                        cand_paths.append(path)
                    else:
                        ref_leaves.append(leaf)
                        ref_paths.append(path)
                    game_step_counts[idx] += 1

            _batch_eval_for_model(candidate_net, cand_leaves)
            _batch_eval_for_model(ref_net, ref_leaves)

            for path, leaf in zip(cand_paths, cand_leaves):
                remove_virtual_loss(path)
                mcts_backup(path, leaf.value)
            for path, leaf in zip(ref_paths, ref_leaves):
                remove_virtual_loss(path)
                mcts_backup(path, leaf.value)

            for idx in need_sim:
                sims_done[idx] += max(1, game_step_counts[idx])

        # Select moves (slight temperature in first 6 plies for variety, greedy thereafter)
        for idx in active_indices:
            rt = roots[idx]
            g = games[idx]
            if not rt.children:
                active[idx] = False
                continue
            if move_counts[idx] < 6:
                pi_d = extract_mcts_policy_target(rt, prune_single_visit_noise=False)
                mv = sample_move_with_temperature(pi_d, move_count=move_counts[idx], board_size=board_size, greedy=False)
            else:
                mv = max(rt.children.items(), key=lambda kv: kv[1].N)[0]
            g.make_move(*mv, check_valid=False)
            move_counts[idx] += 1
            if g.is_game_over() or move_counts[idx] >= max_moves:
                active[idx] = False
            else:
                roots[idx] = advance_mcts_root(rt, mv, g)

    wins = 0
    losses = 0
    draws = 0
    b_wins = 0
    w_wins = 0
    for idx, g in enumerate(games):
        w = g.get_winner()
        c_col = cand_colors[idx]
        if w == 0:
            draws += 1
        elif w == c_col:
            wins += 1
            if c_col == 1:
                b_wins += 1
            else:
                w_wins += 1
        else:
            losses += 1

    win_rate = (wins + 0.5 * draws) / float(total_games)
    clamped_wr = min(0.975, max(0.025, win_rate))
    elo_diff = 400.0 * math.log10(clamped_wr / (1.0 - clamped_wr))

    return {
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "black_wins": b_wins,
        "white_wins": w_wins,
        "win_rate": win_rate,
        "elo_diff": elo_diff,
    }


# ====================================================
#   Self-Describing V2 Checkpoints & Curriculum Loader
# ====================================================
def is_state_dict_finite(state_dict: dict) -> bool:
    """Verifies that every parameter and BatchNorm buffer in `state_dict` is strictly finite (no NaN/Inf)."""
    if not isinstance(state_dict, dict) or not state_dict:
        return False
    for v in state_dict.values():
        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all().item():
            return False
    return True


def save_v2_checkpoint(
    path: str,
    net: nn.Module,
    board_size: int,
    iteration: int = 0,
    stage_iter: int = 0,
    elo_est: float = 1000.0,
):
    raw_net = unwrap_model(net)
    sd = get_cpu_state_dict(raw_net)
    if not is_state_dict_finite(sd):
        logging.error(f"Refusing to save corrupted checkpoint '{path}' (state_dict contains NaN/Inf).")
        print(f"  [ERROR] Skipping save to '{path}': model contains non-finite (NaN/Inf) weights!")
        return False
    payload = {
        "version": 2,
        "arch": "GlobalPoolResNet",
        "blocks": int(raw_net.num_res_blocks),
        "channels": int(raw_net.num_channels),
        "board_size": int(board_size),
        "iteration": int(iteration),
        "stage_iter": int(stage_iter),
        "elo_est": float(elo_est),
        "state_dict": sd,
    }
    torch.save(payload, path)
    return True


def load_checkpoint_auto(path: str) -> dict:
    """
    Inspects a .pth file on CPU and returns normalized metadata:
      - version == 2: self-describing V2 GlobalPoolResNet checkpoint
      - version == 1: legacy V1 fixed-board checkpoint (superseded)
    """
    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        raw = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict) and raw.get("version") == 2 and "state_dict" in raw:
        return raw

    # Detect V2 raw state_dict without wrapper metadata (if saved directly via get_cpu_state_dict)
    if isinstance(raw, dict) and "policy_pass_fc.weight" in raw and "conv_in.weight" in raw:
        channels = int(raw["conv_in.weight"].shape[0])
        block_indices = {
            int(k.split(".")[1]) for k in raw.keys() if k.startswith("res_blocks.") and k.split(".")[1].isdigit()
        }
        blocks = (max(block_indices) + 1) if block_indices else RESNET_BLOCKS
        return {
            "version": 2,
            "arch": "GlobalPoolResNet",
            "blocks": blocks,
            "channels": channels,
            "board_size": BOARD_SIZE,
            "iteration": 0,
            "stage_iter": 0,
            "elo_est": 1000.0,
            "state_dict": raw,
        }

    return {"version": 1, "raw": raw}


def resolve_latest_v2_checkpoint(save_dir: str) -> str:
    """
    Resolves the most recent valid, finite V2 checkpoint across `save_dir/policy_value_net_iter_*.pth`,
    `save_dir/policy_value_net_best_*.pth`, `save_dir/policy_value_net.pth`, and `./policy_value_net.pth`,
    automatically skipping any corrupted (NaN/Inf) files.
    """
    candidates = []
    if os.path.isdir(save_dir):
        for fname in os.listdir(save_dir):
            if fname.startswith("policy_value_net_iter_") and fname.endswith(".pth"):
                num_part = fname[len("policy_value_net_iter_") : -len(".pth")]
                if num_part.isdigit():
                    fpath = os.path.join(save_dir, fname)
                    if os.path.isfile(fpath):
                        candidates.append((int(num_part), os.path.getmtime(fpath), fpath))
            elif fname.startswith("policy_value_net_best_") and fname.endswith(".pth"):
                fpath = os.path.join(save_dir, fname)
                if os.path.isfile(fpath):
                    candidates.append((-1, os.path.getmtime(fpath), fpath))
        sd_default = os.path.join(save_dir, "policy_value_net.pth")
        if os.path.isfile(sd_default):
            candidates.append((-1, os.path.getmtime(sd_default), sd_default))

    if os.path.isfile("policy_value_net.pth"):
        candidates.append((-1, os.path.getmtime("policy_value_net.pth"), "policy_value_net.pth"))

    best_tuple = None
    skipped_nan = []
    for iter_hint, mtime, fpath in candidates:
        try:
            meta = load_checkpoint_auto(fpath)
        except Exception:
            continue
        if meta.get("version") != 2:
            continue
        if not is_state_dict_finite(meta.get("state_dict")):
            skipped_nan.append(fpath)
            continue
        it_num = max(iter_hint, int(meta.get("iteration", 0)))
        cand_key = (it_num, mtime, fpath)
        if best_tuple is None or cand_key[:2] > best_tuple[:2]:
            best_tuple = cand_key

    if skipped_nan and IS_MAIN_PROCESS:
        print(
            f"[Warning] Skipped {len(skipped_nan)} corrupted checkpoint(s) containing NaN/Inf weights "
            f"(e.g., '{skipped_nan[0]}')."
        )

    return best_tuple[2] if best_tuple is not None else ""


# ====================================================
#   Interactive Pygame UI (Subtree Reuse + Symmetries)
# ====================================================
def run_interactive_game(net: nn.Module, human_color: str = "black", play_symmetries: int = PLAY_SYMMETRIES, mcts_sims: int = N_MCTS_SIMS):
    if pygame is None:
        raise RuntimeError("Pygame is required for --play interactive mode. Install with: pip install pygame")

    board_size = unwrap_model(net).board_size
    pygame.init()
    pygame.font.init()

    # Adapt grid size cleanly so 19x19 fits comfortably on standard 1080p laptop screens
    disp_info = pygame.display.Info()
    max_board_px = max(420, disp_info.current_h - 220) if disp_info.current_h > 0 else 820
    grid_px = min(GRID_SIZE, max(26, max_board_px // (board_size + 1)))

    board_pixel_w = (board_size + 1) * grid_px
    board_pixel_h = (board_size + 1) * grid_px
    panel_height = 140

    window_width = max(board_pixel_w, 500)
    window_height = board_pixel_h + panel_height
    offset_x = (window_width - board_pixel_w) // 2
    offset_y = 0

    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"OceanGo V2 ({board_size}x{board_size} GlobalPool ResNet | {play_symmetries}x Sym)")

    font_title = pygame.font.Font(None, 26)
    font_body = pygame.font.Font(None, 21)
    font_small = pygame.font.Font(None, 18)
    font_btn = pygame.font.Font(None, 18)

    human_player = 2 if str(human_color).lower() in ("white", "w", "2") else 1
    ai_player = 3 - human_player
    h_col_str = "Black" if human_player == 1 else "White"
    ai_col_str = "White" if human_player == 1 else "Black"

    game = GoGame(board_size)
    mcts_root = NN_MCTS_Node(game.copy())
    clock = pygame.time.Clock()
    last_move = None
    game_over = False
    show_modal = False
    show_ownership = False
    ownership_probs_black = None
    resigned_by = None
    score_info = None
    current_dead_stones = set()
    msg_str = (
        "Click an intersection to play. P = Pass | O = Ownership Overlay"
        if human_player == 1
        else f"AI moves first as {ai_col_str}. Thinking..."
    )

    btn_y = board_pixel_h + 82
    btn_h = 42
    margin = 16
    spacing = 10
    avail_w = window_width - 2 * margin - 2 * spacing
    btn_w = avail_w // 3

    btn_pass_rect = pygame.Rect(margin, btn_y, btn_w, btn_h)
    btn_end_rect = pygame.Rect(margin + btn_w + spacing, btn_y, btn_w, btn_h)
    btn_resign_rect = pygame.Rect(margin + (btn_w + spacing) * 2, btn_y, btn_w, btn_h)

    modal_w = min(460, window_width - 24)
    modal_h = 265
    modal_x = (window_width - modal_w) // 2
    modal_y = max(14, (board_pixel_h - modal_h) // 2)
    modal_rect = pygame.Rect(modal_x, modal_y, modal_w, modal_h)
    modal_close_rect = pygame.Rect(modal_rect.right - 32, modal_rect.y + 10, 22, 22)

    m_btn_spacing = 6
    m_avail_w = modal_w - 24 - 3 * m_btn_spacing
    m_btn_w = m_avail_w // 4
    m_btn_h = 36
    m_btn_y = modal_rect.bottom - 48

    modal_review_rect = pygame.Rect(modal_rect.x + 12, m_btn_y, m_btn_w, m_btn_h)
    modal_play_again_rect = pygame.Rect(modal_rect.x + 12 + m_btn_w + m_btn_spacing, m_btn_y, m_btn_w, m_btn_h)
    modal_swap_rect = pygame.Rect(modal_rect.x + 12 + (m_btn_w + m_btn_spacing) * 2, m_btn_y, m_btn_w, m_btn_h)
    modal_exit_rect = pygame.Rect(modal_rect.x + 12 + (m_btn_w + m_btn_spacing) * 3, m_btn_y, m_btn_w, m_btn_h)

    def refresh_ownership_overlay():
        nonlocal ownership_probs_black
        if not show_ownership:
            return
        inp = boards_to_tensor([game.get_nn_input()])
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=(AMP_ENABLED and not IS_TPU)):
            _, _, own_logits = net(inp, return_aux=True)
            probs_cp = torch.sigmoid(own_logits.float()).squeeze(0).cpu().numpy()
        ownership_probs_black = probs_cp if game.current_player == 1 else (1.0 - probs_cp)

    def reset_game(swap_sides=False):
        nonlocal game, mcts_root, last_move, game_over, show_modal, resigned_by, score_info
        nonlocal current_dead_stones, human_player, ai_player, msg_str, h_col_str, ai_col_str, ownership_probs_black
        if swap_sides:
            human_player = 3 - human_player
            ai_player = 3 - human_player
            h_col_str = "Black" if human_player == 1 else "White"
            ai_col_str = "White" if human_player == 1 else "Black"
        game = GoGame(board_size)
        mcts_root = NN_MCTS_Node(game.copy())
        last_move = None
        game_over = False
        show_modal = False
        resigned_by = None
        score_info = None
        ownership_probs_black = None
        current_dead_stones = set()
        if show_ownership:
            refresh_ownership_overlay()
        if human_player == 1:
            msg_str = f"New game! You play {h_col_str} (first move). P = Pass | O = Ownership"
        else:
            msg_str = f"New game! You play {h_col_str}. AI plays {ai_col_str} (thinking)..."

    def trigger_game_over(resigned=None):
        nonlocal game_over, show_modal, resigned_by, score_info, current_dead_stones
        game_over = True
        show_modal = True
        resigned_by = resigned
        if not resigned:
            current_dead_stones = game.auto_detect_dead_stones()
            score_info = game.get_score_breakdown(dead_stones=current_dead_stones)

    def draw_button(surf, rect, text, font, base_color, hover_color, text_color=(255, 255, 255)):
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = rect.collidepoint(mouse_pos)
        color = hover_color if is_hovered else base_color
        pygame.draw.rect(surf, color, rect, border_radius=7)
        pygame.draw.rect(surf, (255, 255, 255, 50), rect, width=1, border_radius=7)
        txt_surf = font.render(text, True, text_color)
        surf.blit(txt_surf, txt_surf.get_rect(center=rect.center))

    def render_frame():
        screen.fill((15, 23, 42))

        # 1. Wood Go Board
        pygame.draw.rect(screen, (220, 179, 92), (offset_x, offset_y, board_pixel_w, board_pixel_h))
        pygame.draw.rect(screen, (160, 120, 50), (offset_x, offset_y, board_pixel_w, board_pixel_h), 2)

        for i in range(1, board_size + 1):
            gx = offset_x + i * grid_px
            gy = offset_y + i * grid_px
            pygame.draw.line(screen, (0, 0, 0), (gx, offset_y + grid_px), (gx, offset_y + board_pixel_h - grid_px), 2)
            pygame.draw.line(screen, (0, 0, 0), (offset_x + grid_px, gy), (offset_x + board_pixel_w - grid_px, gy), 2)

        if board_size == 19:
            star_points = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
        elif board_size == 13:
            star_points = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif board_size == 9:
            star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        else:
            star_points = []
        for sx, sy in star_points:
            px = offset_x + (sy + 1) * grid_px
            py = offset_y + (sx + 1) * grid_px
            pygame.draw.circle(screen, (0, 0, 0), (px, py), 4)

        # Optional Live Auxiliary Ownership Head Overlay (toggle with 'O')
        if show_ownership and ownership_probs_black is not None and not game_over:
            for r in range(board_size):
                for c in range(board_size):
                    pb = float(ownership_probs_black[r, c])
                    strength = abs(pb - 0.5) * 2.0
                    if strength > 0.18:
                        sz = max(4, int(grid_px * 0.28 * strength))
                        cx = offset_x + (c + 1) * grid_px
                        cy = offset_y + (r + 1) * grid_px
                        col = (15, 15, 15) if pb > 0.5 else (250, 250, 250)
                        pygame.draw.rect(screen, col, pygame.Rect(cx - sz // 2, cy - sz // 2, sz, sz))

        # 2. Stones
        radius = max(8, grid_px // 2 - 3)
        for x in range(board_size):
            for y in range(board_size):
                cell = game.board[x, y]
                if cell == 0:
                    continue
                cx = offset_x + (y + 1) * grid_px
                cy = offset_y + (x + 1) * grid_px
                is_dead = game_over and score_info and ((x, y) in score_info["dead_stones"])

                if is_dead:
                    dim_surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
                    center = (radius + 2, radius + 2)
                    if cell == 1:
                        pygame.draw.circle(dim_surf, (15, 15, 15, 90), center, radius)
                    else:
                        pygame.draw.circle(dim_surf, (250, 250, 250, 90), center, radius)
                        pygame.draw.circle(dim_surf, (20, 20, 20, 90), center, radius, 1)
                    screen.blit(dim_surf, (cx - radius - 2, cy - radius - 2))
                else:
                    if cell == 1:
                        pygame.draw.circle(screen, (15, 15, 15), (cx, cy), radius)
                    elif cell == 2:
                        pygame.draw.circle(screen, (250, 250, 250), (cx, cy), radius)
                        pygame.draw.circle(screen, (20, 20, 20), (cx, cy), radius, 1)

        # 3. End-Game Scoring Squares
        if game_over and score_info:
            sq_size = max(6, min(10, grid_px // 3))
            half_sq = sq_size // 2
            for r, c in score_info["b_alive_stones"]:
                cx = offset_x + (c + 1) * grid_px
                cy = offset_y + (r + 1) * grid_px
                sq_rect = pygame.Rect(cx - half_sq, cy - half_sq, sq_size, sq_size)
                pygame.draw.rect(screen, (15, 15, 15), sq_rect)
                pygame.draw.rect(screen, (240, 240, 240), sq_rect, 1)
            for r, c in score_info["b_terr_points"]:
                cx = offset_x + (c + 1) * grid_px
                cy = offset_y + (r + 1) * grid_px
                sq_rect = pygame.Rect(cx - half_sq, cy - half_sq, sq_size, sq_size)
                pygame.draw.rect(screen, (15, 15, 15), sq_rect)
                if (r, c) in score_info["dead_stones"]:
                    pygame.draw.rect(screen, (240, 240, 240), sq_rect, 1)
            for r, c in score_info["w_alive_stones"]:
                cx = offset_x + (c + 1) * grid_px
                cy = offset_y + (r + 1) * grid_px
                sq_rect = pygame.Rect(cx - half_sq, cy - half_sq, sq_size, sq_size)
                pygame.draw.rect(screen, (250, 250, 250), sq_rect)
                pygame.draw.rect(screen, (15, 15, 15), sq_rect, 1)
            for r, c in score_info["w_terr_points"]:
                cx = offset_x + (c + 1) * grid_px
                cy = offset_y + (r + 1) * grid_px
                sq_rect = pygame.Rect(cx - half_sq, cy - half_sq, sq_size, sq_size)
                pygame.draw.rect(screen, (250, 250, 250), sq_rect)
                pygame.draw.rect(screen, (15, 15, 15), sq_rect, 1)

        if not game_over and last_move and last_move != (-1, -1):
            lx, ly = last_move
            cx = offset_x + (ly + 1) * grid_px
            cy = offset_y + (lx + 1) * grid_px
            pygame.draw.circle(screen, (239, 68, 68), (cx, cy), max(3, radius // 4))

        # 4. Control Panel
        panel_rect = pygame.Rect(0, board_pixel_h, window_width, panel_height)
        pygame.draw.rect(screen, (24, 32, 47), panel_rect)
        pygame.draw.line(screen, (51, 65, 85), (0, board_pixel_h), (window_width, board_pixel_h), 2)

        ind_cx = 24
        ind_cy = board_pixel_h + 24
        if game_over:
            pygame.draw.circle(screen, (245, 158, 11), (ind_cx, ind_cy), 8)
            if resigned_by == human_player:
                t_str = f"Game Over (AI Won [{ai_col_str}] - You Resigned)"
            elif resigned_by == ai_player:
                t_str = f"Game Over (You Won [{h_col_str}] - AI Resigned)"
            elif score_info:
                w = score_info["winner"]
                m = score_info["margin"]
                if w == human_player:
                    t_str = f"Game Over (You Won [{h_col_str}] +{m:.1f} pts)"
                elif w == ai_player:
                    t_str = f"Game Over (AI Won [{ai_col_str}] +{m:.1f} pts)"
                else:
                    t_str = "Game Over (Tie)"
            else:
                t_str = "Game Over"
            turn_txt = font_body.render(t_str, True, (245, 158, 11))
        elif game.current_player == human_player:
            if human_player == 1:
                pygame.draw.circle(screen, (15, 15, 15), (ind_cx, ind_cy), 8)
                pygame.draw.circle(screen, (240, 240, 240), (ind_cx, ind_cy), 8, 1)
            else:
                pygame.draw.circle(screen, (250, 250, 250), (ind_cx, ind_cy), 8)
                pygame.draw.circle(screen, (15, 15, 15), (ind_cx, ind_cy), 8, 1)
            turn_txt = font_body.render(f"Your Turn ({h_col_str})", True, (255, 255, 255))
        else:
            if ai_player == 1:
                pygame.draw.circle(screen, (15, 15, 15), (ind_cx, ind_cy), 8)
                pygame.draw.circle(screen, (240, 240, 240), (ind_cx, ind_cy), 8, 1)
            else:
                pygame.draw.circle(screen, (250, 250, 250), (ind_cx, ind_cy), 8)
                pygame.draw.circle(screen, (15, 15, 15), (ind_cx, ind_cy), 8, 1)
            turn_txt = font_body.render(f"AI Thinking... ({ai_col_str})", True, (255, 255, 255))
        screen.blit(turn_txt, (ind_cx + 14, ind_cy - turn_txt.get_height() // 2))

        if game_over and score_info and not resigned_by:
            counts_txt = font_small.render(
                f"Score: B {score_info['b_score']:.1f} | W {score_info['w_score']:.1f} (incl. 7.5 Komi)",
                True,
                (251, 191, 36),
            )
        else:
            b_count = int(np.count_nonzero(game.board == 1))
            w_count = int(np.count_nonzero(game.board == 2))
            counts_txt = font_small.render(f"Stones: B {b_count} | W {w_count} (+7.5 Komi)", True, (148, 163, 184))
        screen.blit(counts_txt, (window_width - counts_txt.get_width() - 16, ind_cy - counts_txt.get_height() // 2))

        is_alert = any(k in msg_str.lower() for k in ("passed", "game over", "resign", "alert", "caution", "dead"))
        msg_color = (251, 191, 36) if is_alert else (203, 213, 225)
        msg_surf = font_small.render(msg_str, True, msg_color)
        screen.blit(msg_surf, (16, board_pixel_h + 48))

        if game_over and not show_modal:
            draw_button(screen, btn_pass_rect, "Show Score", font_btn, (37, 99, 235), (59, 130, 246))
            draw_button(screen, btn_end_rect, "Play Again (R)", font_btn, (16, 185, 129), (52, 211, 153))
            draw_button(screen, btn_resign_rect, "Swap Side (S)", font_btn, (139, 92, 246), (124, 58, 237))
        else:
            draw_button(screen, btn_pass_rect, "Pass (P)", font_btn, (37, 99, 235), (59, 130, 246))
            draw_button(screen, btn_end_rect, "Count / End", font_btn, (16, 185, 129), (52, 211, 153))
            draw_button(screen, btn_resign_rect, "Resign", font_btn, (220, 38, 38), (239, 68, 68))

        if game_over and show_modal:
            dim_surf = pygame.Surface((window_width, board_pixel_h), pygame.SRCALPHA)
            dim_surf.fill((0, 0, 0, 160))
            screen.blit(dim_surf, (0, 0))

            pygame.draw.rect(screen, (24, 32, 47), modal_rect, border_radius=12)
            pygame.draw.rect(screen, (71, 85, 105), modal_rect, width=2, border_radius=12)

            mouse_pos = pygame.mouse.get_pos()
            is_close_hov = modal_close_rect.collidepoint(mouse_pos)
            close_col = (220, 38, 38) if is_close_hov else (51, 65, 85)
            pygame.draw.rect(screen, close_col, modal_close_rect, border_radius=4)
            x_txt = font_small.render("X", True, (255, 255, 255))
            screen.blit(x_txt, x_txt.get_rect(center=modal_close_rect.center))

            title_surf = font_title.render("GAME OVER", True, (251, 191, 36))
            screen.blit(title_surf, (modal_rect.centerx - title_surf.get_width() // 2, modal_rect.y + 16))

            if resigned_by == human_player:
                res_surf = font_body.render(f"AI Wins ({ai_col_str} - You Resigned)", True, (239, 68, 68))
            elif resigned_by == ai_player:
                res_surf = font_body.render(f"You Win ({h_col_str} - AI Resigned)", True, (34, 197, 94))
            elif score_info:
                w = score_info["winner"]
                m = score_info["margin"]
                if w == human_player:
                    res_surf = font_body.render(f"You Won ({h_col_str})! (+{m:.1f} pts)", True, (34, 197, 94))
                elif w == ai_player:
                    res_surf = font_body.render(f"AI Won ({ai_col_str})! (+{m:.1f} pts)", True, (96, 165, 250))
                else:
                    res_surf = font_body.render("Game Tied!", True, (255, 255, 255))
            else:
                res_surf = font_body.render("Game Over", True, (255, 255, 255))
            screen.blit(res_surf, (modal_rect.centerx - res_surf.get_width() // 2, modal_rect.y + 48))

            if score_info and not resigned_by:
                b_owner = "You" if human_player == 1 else "AI"
                w_owner = "You" if human_player == 2 else "AI"
                b_str = f"Black ({b_owner}): {score_info['b_stones']} stones + {score_info['b_terr']} terr = {score_info['b_score']:.1f} pts"
                w_str = f"White ({w_owner}): {score_info['w_stones']} stones + {score_info['w_terr']} terr + {score_info['komi']:.1f} komi = {score_info['w_score']:.1f} pts"
                b_surf = font_small.render(b_str, True, (226, 232, 240))
                w_surf = font_small.render(w_str, True, (226, 232, 240))
                screen.blit(b_surf, (modal_rect.centerx - b_surf.get_width() // 2, modal_rect.y + 82))
                screen.blit(w_surf, (modal_rect.centerx - w_surf.get_width() // 2, modal_rect.y + 104))

                terr_hint = font_small.render("Squares: Black = Black Score | White = White Score (incl. dead stones)", True, (148, 163, 184))
                screen.blit(terr_hint, (modal_rect.centerx - terr_hint.get_width() // 2, modal_rect.y + 132))
                terr_hint2 = font_small.render("(Click 'Review' or [X] to inspect; click any stone to toggle Alive/Dead)", True, (148, 163, 184))
                screen.blit(terr_hint2, (modal_rect.centerx - terr_hint2.get_width() // 2, modal_rect.y + 152))

            draw_button(screen, modal_review_rect, "Review", font_btn, (37, 99, 235), (59, 130, 246))
            draw_button(screen, modal_play_again_rect, "Play Again (R)", font_btn, (16, 185, 129), (52, 211, 153))
            draw_button(screen, modal_swap_rect, "Swap Side (S)", font_btn, (139, 92, 246), (124, 58, 237))
            draw_button(screen, modal_exit_rect, "Exit (Q)", font_btn, (71, 85, 105), (100, 116, 139))

        pygame.display.flip()

    while True:
        clock.tick(30)
        render_frame()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if game_over:
                    if event.key == pygame.K_r:
                        reset_game(swap_sides=False)
                    elif event.key == pygame.K_s:
                        reset_game(swap_sides=True)
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    elif event.key in (pygame.K_ESCAPE, pygame.K_SPACE, pygame.K_b):
                        if show_modal:
                            show_modal = False
                            msg_str = "Reviewing board: Click stones to toggle Dead/Alive. [S] = Swap Sides."
                        else:
                            pygame.quit()
                            sys.exit()
                else:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_o:
                        show_ownership = not show_ownership
                        if show_ownership:
                            refresh_ownership_overlay()
                            msg_str = "Live Ownership Head overlay: ON (press O to hide)."
                        else:
                            msg_str = "Live Ownership Head overlay: OFF."
                    elif event.key == pygame.K_p and game.current_player == human_player:
                        game.make_move(-1, -1)
                        mcts_root = advance_mcts_root(mcts_root, (-1, -1), game)
                        last_move = (-1, -1)
                        if show_ownership:
                            refresh_ownership_overlay()
                        logging.info(f"Interactive Game: Human ({h_col_str}) passed.")
                        if game.is_game_over():
                            trigger_game_over()
                            msg_str = "Both players passed. Game Over!"
                        else:
                            msg_str = "You passed. AI is thinking..."
                    elif event.key in (pygame.K_e, pygame.K_c) and not game_over:
                        trigger_game_over()
                        msg_str = "Game ended by player. Score tallied."

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if game_over:
                    if show_modal:
                        if modal_close_rect.collidepoint((mx, my)) or modal_review_rect.collidepoint((mx, my)):
                            show_modal = False
                            msg_str = "Reviewing board: Click stones to toggle Dead/Alive. [S] = Swap Sides."
                        elif modal_play_again_rect.collidepoint((mx, my)):
                            reset_game(swap_sides=False)
                        elif modal_swap_rect.collidepoint((mx, my)):
                            reset_game(swap_sides=True)
                        elif modal_exit_rect.collidepoint((mx, my)):
                            pygame.quit()
                            sys.exit()
                        elif not modal_rect.collidepoint((mx, my)):
                            show_modal = False
                            msg_str = "Reviewing board: Click stones to toggle Dead/Alive. [S] = Swap Sides."
                    else:
                        if btn_pass_rect.collidepoint((mx, my)):
                            show_modal = True
                            msg_str = "Game over dialog open."
                        elif btn_end_rect.collidepoint((mx, my)):
                            reset_game(swap_sides=False)
                        elif btn_resign_rect.collidepoint((mx, my)):
                            reset_game(swap_sides=True)
                        elif my < board_pixel_h:
                            grid_x = round((my - offset_y) / grid_px - 1)
                            grid_y = round((mx - offset_x) / grid_px - 1)
                            if 0 <= grid_x < board_size and 0 <= grid_y < board_size:
                                cx = offset_x + (grid_y + 1) * grid_px
                                cy = offset_y + (grid_x + 1) * grid_px
                                if (mx - cx) ** 2 + (my - cy) ** 2 <= (grid_px * 0.45) ** 2:
                                    if game.board[grid_x, grid_y] != 0:
                                        color, grp = game.get_group(grid_x, grid_y)
                                        if any(pt in current_dead_stones for pt in grp):
                                            current_dead_stones.difference_update(grp)
                                            status = "Alive"
                                        else:
                                            current_dead_stones.update(grp)
                                            status = "Dead"
                                        score_info = game.get_score_breakdown(dead_stones=current_dead_stones)
                                        col_name = "Black" if color == 1 else "White"
                                        msg_str = f"Toggled {col_name} group ({len(grp)} stone{'s' if len(grp) > 1 else ''}) as {status}. Score updated!"
                else:
                    if btn_pass_rect.collidepoint((mx, my)):
                        if game.current_player == human_player:
                            game.make_move(-1, -1)
                            mcts_root = advance_mcts_root(mcts_root, (-1, -1), game)
                            last_move = (-1, -1)
                            if show_ownership:
                                refresh_ownership_overlay()
                            logging.info(f"Interactive Game: Human ({h_col_str}) passed.")
                            if game.is_game_over():
                                trigger_game_over()
                                msg_str = "Both players passed. Game Over!"
                            else:
                                msg_str = "You passed. AI is thinking..."
                    elif btn_end_rect.collidepoint((mx, my)):
                        trigger_game_over()
                        logging.info("Interactive Game: Human triggered End & Score.")
                        msg_str = "Game ended by player. Score tallied."
                    elif btn_resign_rect.collidepoint((mx, my)):
                        trigger_game_over(resigned=human_player)
                        logging.info(f"Interactive Game: Human ({h_col_str}) resigned.")
                        msg_str = "You resigned. AI wins!"
                    elif my < board_pixel_h and game.current_player == human_player:
                        grid_x = round((my - offset_y) / grid_px - 1)
                        grid_y = round((mx - offset_x) / grid_px - 1)
                        if 0 <= grid_x < board_size and 0 <= grid_y < board_size:
                            cx = offset_x + (grid_y + 1) * grid_px
                            cy = offset_y + (grid_x + 1) * grid_px
                            if (mx - cx) ** 2 + (my - cy) ** 2 <= (grid_px * 0.45) ** 2:
                                if game.is_valid_move(grid_x, grid_y):
                                    game.make_move(grid_x, grid_y, check_valid=False)
                                    mcts_root = advance_mcts_root(mcts_root, (grid_x, grid_y), game)
                                    last_move = (grid_x, grid_y)
                                    if show_ownership:
                                        refresh_ownership_overlay()
                                    logging.info(f"Interactive Game: Human ({h_col_str}) played ({grid_x}, {grid_y})")
                                    msg_str = f"You played ({grid_x}, {grid_y}). AI is thinking..."
                                else:
                                    msg_str = f"Illegal move at ({grid_x}, {grid_y}). Try another intersection."

        # AI Turn Handling (with MCTS Subtree Reuse + Dihedral Symmetry Ensemble)
        if not game_over and game.current_player == ai_player:
            msg_str = f"AI is thinking ({ai_col_str}, reused N={mcts_root.N})..."
            render_frame()

            if not mcts_root.is_expanded:
                nn_batch_evaluate_and_expand([mcts_root], net=net, num_symmetries=play_symmetries)
                mcts_root.N = 1
                mcts_root.W = -mcts_root.value
                mcts_root.Q = -mcts_root.value

            stones_on_board = int(np.count_nonzero(game.board != 0))
            min_opening_stones = max(6, (board_size * board_size) // 6)
            forbid_premature_pass = (
                stones_on_board < min_opening_stones
                and game.passes_in_a_row == 0
                and (-1, -1) in mcts_root.children
                and len(mcts_root.children) > 1
            )
            if forbid_premature_pass:
                mcts_root.children[(-1, -1)].P = 0.0
                b_moves = [mv for mv in mcts_root.children.keys() if mv != (-1, -1)]
                b_sum = sum(mcts_root.children[mv].P for mv in b_moves)
                if b_sum > 1e-6:
                    inv_b = 1.0 / b_sum
                    for mv in b_moves:
                        mcts_root.children[mv].P *= inv_b
                else:
                    p_raw, _ = ensemble_infer_symmetries(
                        net, mcts_root.game_state.get_nn_input()[None], num_symmetries=play_symmetries
                    )
                    b_idxs = [mv[0] * board_size + mv[1] for mv in b_moves]
                    b_logits = p_raw[0, b_idxs]
                    b_probs = softmax(b_logits) if np.all(np.isfinite(b_logits)) else np.full(len(b_moves), 1.0 / len(b_moves))
                    for mv, pr in zip(b_moves, b_probs):
                        mcts_root.children[mv].P = float(pr)

            pi_dict = mcts_run(
                mcts_root,
                net=net,
                n_simulations=mcts_sims,
                temp=1e-3,
                batch_size=MCTS_BATCH_SIZE,
                num_symmetries=play_symmetries,
            )

            if not pi_dict:
                game.make_move(-1, -1)
                mcts_root = advance_mcts_root(mcts_root, (-1, -1), game)
                last_move = (-1, -1)
                if game.is_game_over():
                    trigger_game_over()
                    msg_str = "Both players passed. Game Over!"
                else:
                    msg_str = "AI passed! Click [Pass] to finish, or keep playing."
            else:
                if forbid_premature_pass and (-1, -1) in pi_dict and len(pi_dict) > 1:
                    board_candidates = [mv for mv in mcts_root.children.keys() if mv != (-1, -1)]
                    best_move = max(board_candidates, key=lambda mv: (mcts_root.children[mv].N, mcts_root.children[mv].P))
                else:
                    best_move = max(pi_dict.items(), key=lambda elem: elem[1])[0]
                win_pct = 50.0 * (-mcts_root.Q + 1.0)
                if not math.isfinite(win_pct):
                    win_pct = 50.0
                if best_move == (-1, -1):
                    game.make_move(-1, -1)
                    mcts_root = advance_mcts_root(mcts_root, (-1, -1), game)
                    last_move = (-1, -1)
                    if game.is_game_over():
                        trigger_game_over()
                        msg_str = "Both players passed. Game Over!"
                    else:
                        msg_str = f"AI passed (AI est. {win_pct:.0f}%). Click [Pass] to finish, or keep playing."
                else:
                    game.make_move(*best_move, check_valid=False)
                    mcts_root = advance_mcts_root(mcts_root, best_move, game)
                    last_move = best_move
                    msg_str = f"AI played {best_move} (AI win est. {win_pct:.0f}%). Your turn ({h_col_str})."

            if show_ownership and not game_over:
                refresh_ownership_overlay()


# ====================================================
#   Parallel Self-Play Runner Helper (Shared Memory)
# ====================================================
def run_parallel_self_play_batch(net: nn.Module, board_size: int, n_games: int, n_sims: int, num_workers: int, mp_ctx, show_progress: bool = True):
    """
    Executes `n_games` parallel self-play games across `num_workers` CPU processes
    using pre-allocated SharedMemory uint8/float32 slots and a central GPU batch server.
    """
    num_workers = max(1, min(num_workers, n_games))
    slot_batch = MCTS_BATCH_SIZE

    req_nbytes = num_workers * slot_batch * IN_PLANES * board_size * board_size * np.dtype(np.uint8).itemsize
    res_nbytes = num_workers * slot_batch * (board_size * board_size + 2) * np.dtype(np.float32).itemsize

    shm_req = shared_memory.SharedMemory(create=True, size=req_nbytes)
    shm_res = shared_memory.SharedMemory(create=True, size=res_nbytes)
    shm_req_arr = np.ndarray((num_workers, slot_batch, IN_PLANES, board_size, board_size), dtype=np.uint8, buffer=shm_req.buf)
    shm_res_arr = np.ndarray((num_workers, slot_batch, board_size * board_size + 2), dtype=np.float32, buffer=shm_res.buf)

    req_queue = mp_ctx.Queue()
    res_queues = [mp_ctx.Queue() for _ in range(num_workers)]
    progress_queue = mp_ctx.Queue()

    gpu_thread = threading.Thread(
        target=gpu_server_thread,
        args=(net, req_queue, res_queues, num_workers, shm_req_arr, shm_res_arr, board_size),
        daemon=True,
    )
    gpu_thread.start()

    iteration_states = []
    totals = {
        "games": 0,
        "resigned_games": 0,
        "full_playout_games": 0,
        "false_resign_checks": 0,
        "false_resign_count": 0,
        "total_moves": 0,
        "policy_entropy_sum": 0.0,
        "policy_entropy_cnt": 0,
    }

    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp_ctx,
            initializer=_init_self_play_worker,
            initargs=(req_queue, res_queues, progress_queue, shm_req.name, shm_res.name, num_workers, slot_batch, board_size),
        ) as executor:
            futures = []
            base_games, rem_games = divmod(n_games, num_workers)
            for w_id in range(num_workers):
                g_cnt = base_games + (1 if w_id < rem_games else 0)
                if g_cnt > 0:
                    futures.append(executor.submit(self_play_wrapper, board_size, n_sims, w_id, g_cnt))
                else:
                    req_queue.put("DONE")

            completed_games = 0
            start_t = time.time()
            while completed_games < n_games:
                try:
                    progress_queue.get(timeout=0.5)
                    completed_games += 1
                    if show_progress:
                        print_progress_bar(completed_games, n_games, start_t, prefix="Self-Play")
                    if completed_games % 25 == 0 or completed_games == n_games:
                        logging.info(f"  Self-play progress: {completed_games}/{n_games} games completed.")
                except queue.Empty:
                    if all(f.done() for f in futures) and progress_queue.empty():
                        break

            for future in futures:
                w_states, w_stats = future.result()
                iteration_states.extend(w_states)
                for k in totals:
                    totals[k] += w_stats[k]
                req_queue.put("DONE")

        gpu_thread.join(timeout=10.0)
    finally:
        # Release numpy views before closing & unlinking shared memory segments
        del shm_req_arr
        del shm_res_arr
        for shm_seg in (shm_req, shm_res):
            try:
                shm_seg.close()
            except Exception:
                pass
            try:
                shm_seg.unlink()
            except Exception:
                pass

    return iteration_states, totals


# ====================================================
#   Main Entrypoint & Curriculum Training Loop
# ====================================================
def main():
    global BOARD_SIZE, SCREEN_SIZE, DIRICHLET_ALPHA, PARALLEL_THREADS
    init_environment()

    parser = argparse.ArgumentParser(description="OceanGo V2 - Board-Agnostic Global-Pooling AlphaZero/KataGo Engine")
    parser.add_argument("--model", type=str, default="", help="Path to specific V2 checkpoint to load (supports cross-board transfer)")
    parser.add_argument("--play", action="store_true", help="Play interactively against the AI in the Pygame window")
    parser.add_argument("--train", action="store_true", help="Start self-play reinforcement learning loop")
    parser.add_argument("--color", type=str, default="black", choices=["black", "white"], help="Human color in --play ('black' or 'white')")
    parser.add_argument("--dry-run", action="store_true", help="Run end-to-end verification (SharedMem workers, GPU batch, Multi-Head loss, Arena, Checkpoint)")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory where checkpoints and training state are saved")
    parser.add_argument("--board-size", type=int, default=None, help=f"Go board dimension: 9, 13, or 19 (default: {BOARD_SIZE} or checkpoint size in --play)")
    parser.add_argument("--blocks", type=int, default=RESNET_BLOCKS, help=f"Number of residual blocks (default: {RESNET_BLOCKS})")
    parser.add_argument("--channels", type=int, default=RESNET_CHANNELS, help=f"Number of trunk channels (default: {RESNET_CHANNELS})")
    parser.add_argument("--games", type=int, default=N_GAMES_PER_ITER, help=f"Self-play games per iteration (default: {N_GAMES_PER_ITER})")
    parser.add_argument("--sims", type=int, default=N_MCTS_SIMS, help=f"Full-search MCTS simulations per move (default: {N_MCTS_SIMS})")
    parser.add_argument("--iters", type=int, default=N_ITER, help=f"Number of training iterations to run (default: {N_ITER})")
    parser.add_argument("--workers", type=int, default=0, help=f"Parallel CPU self-play workers (default: auto = {PARALLEL_THREADS})")
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE, help=f"Training minibatch size (default: {TRAIN_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Base SGD learning rate (default: {LEARNING_RATE})")
    parser.add_argument("--replay-ratio", type=float, default=REPLAY_RATIO, help=f"Controlled training samples per new position (default: {REPLAY_RATIO})")
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL, help=f"Arena evaluation interval in iterations, 0 to disable (default: {EVAL_INTERVAL})")
    parser.add_argument("--eval-games", type=int, default=EVAL_GAMES, help=f"Number of paired Arena evaluation games (default: {EVAL_GAMES})")
    parser.add_argument("--play-symmetries", type=int, default=PLAY_SYMMETRIES, help=f"Dihedral symmetries ensembled during --play (1..8, default: {PLAY_SYMMETRIES})")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile graph fusion where supported")
    args = parser.parse_args()

    if args.workers > 0:
        PARALLEL_THREADS = args.workers

    board_size_explicit = (args.board_size is not None)
    BOARD_SIZE = args.board_size if board_size_explicit else BOARD_SIZE
    SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE
    DIRICHLET_ALPHA = get_dirichlet_alpha(BOARD_SIZE)
    REPLAY_BUFFER.reset(BOARD_SIZE)

    # Resolve checkpoint path (fixing V1 save_dir/policy_value_net.pth precedence bug)
    load_path = args.model if args.model else resolve_latest_v2_checkpoint(args.save_dir)

    start_iter = 0
    stage_iter = 0
    current_elo = 1000.0
    curriculum_transition = False
    net_blocks = args.blocks
    net_channels = args.channels

    if load_path and os.path.isfile(load_path):
        ckpt = load_checkpoint_auto(load_path)
        if ckpt.get("version") == 2 and not is_state_dict_finite(ckpt.get("state_dict")):
            fallback_path = resolve_latest_v2_checkpoint(args.save_dir) if args.model else ""
            if fallback_path and os.path.abspath(fallback_path) != os.path.abspath(load_path):
                print(
                    f"[ERROR] Checkpoint '{load_path}' contains corrupted NaN/Inf weights! "
                    f"Falling back to latest healthy checkpoint '{fallback_path}'."
                )
                load_path = fallback_path
                ckpt = load_checkpoint_auto(load_path)
            else:
                print(
                    f"[ERROR] Checkpoint '{load_path}' contains corrupted NaN/Inf weights! "
                    f"Run `python policy_print.py` to inspect '{args.save_dir}' or retrain with the Blackwell stability fix."
                )
                ckpt = {"version": 0}
        if ckpt.get("version") == 2:
            net_blocks = ckpt["blocks"]
            net_channels = ckpt["channels"]
            ckpt_board = ckpt.get("board_size", BOARD_SIZE)

            if args.play and not board_size_explicit:
                BOARD_SIZE = ckpt_board
                SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE
                DIRICHLET_ALPHA = get_dirichlet_alpha(BOARD_SIZE)

            net = PolicyValueNet(board_size=BOARD_SIZE, num_channels=net_channels, num_res_blocks=net_blocks)
            net.load_state_dict(ckpt["state_dict"], strict=True)
            start_iter = int(ckpt.get("iteration", 0))
            stage_iter = int(ckpt.get("stage_iter", start_iter))
            current_elo = float(ckpt.get("elo_est", 1000.0))

            if ckpt_board != BOARD_SIZE:
                curriculum_transition = True
                stage_iter = 0
                REPLAY_BUFFER.reset(BOARD_SIZE)
                print(
                    f"[Curriculum Transfer] Loaded 100% of weights from {ckpt_board}x{ckpt_board} checkpoint '{load_path}' "
                    f"into {BOARD_SIZE}x{BOARD_SIZE} ({net_blocks} blocks x {net_channels} channels)!"
                )
                logging.info(f"Curriculum 100% weight transfer {ckpt_board}x{ckpt_board} -> {BOARD_SIZE}x{BOARD_SIZE} from {load_path}")
            else:
                print(
                    f"Loaded V2 model from '{load_path}' "
                    f"({BOARD_SIZE}x{BOARD_SIZE}, {net_blocks}x{net_channels}, iter={start_iter}, Elo={current_elo:.0f})."
                )
                logging.info(f"Loaded V2 model from {load_path} (iter={start_iter}, Elo={current_elo:.0f})")
        else:
            print(
                f"Note: Skipping legacy V1 fixed-board checkpoint '{load_path}'. "
                f"Initializing fresh V2 board-agnostic GlobalPoolResNet ({net_blocks}x{net_channels}) on {BOARD_SIZE}x{BOARD_SIZE}."
            )
            net = PolicyValueNet(board_size=BOARD_SIZE, num_channels=net_channels, num_res_blocks=net_blocks)
    else:
        print(
            f"No existing V2 checkpoint found. Initializing fresh GlobalPoolResNet "
            f"({net_blocks} blocks x {net_channels} channels) on {BOARD_SIZE}x{BOARD_SIZE}."
        )
        net = PolicyValueNet(board_size=BOARD_SIZE, num_channels=net_channels, num_res_blocks=net_blocks)

    net = net.to(device=device)
    if USE_CHANNELS_LAST:
        net = net.to(memory_format=torch.channels_last)

    if args.compile and not IS_TPU:
        can_compile = True
        if device.type == "mps":
            can_compile = False
            print("[Compiler] Note: Apple Silicon MPS lacks Triton; using faster eager MPS bfloat16 execution.")
        elif OS_NAME == "windows" and device.type == "cuda":
            try:
                import triton  # noqa: F401
            except ImportError:
                can_compile = False
                print("[Compiler] Note: 'triton' is not installed on Windows (install triton-windows + MSVC to enable); continuing with fast eager CUDA bfloat16.")
        if can_compile:
            try:
                if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
                    torch._dynamo.config.suppress_errors = True
                    if hasattr(torch._dynamo.config, "cache_size_limit"):
                        torch._dynamo.config.cache_size_limit = max(64, int(torch._dynamo.config.cache_size_limit))
                    if hasattr(torch._dynamo.config, "recompile_limit"):
                        torch._dynamo.config.recompile_limit = max(64, int(torch._dynamo.config.recompile_limit))
                compiled_net = torch.compile(net)
                dummy_x1 = boards_to_tensor(np.zeros((2, IN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32))
                dummy_x2 = boards_to_tensor(np.zeros((8, IN_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32))
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=AMP_ENABLED):
                    _ = compiled_net(dummy_x1)
                    _ = compiled_net(dummy_x2)
                net = compiled_net
                print("[Compiler] torch.compile graph fusion active (dynamic batch warmup complete).")
            except Exception as e:
                print(f"[Compiler] torch.compile unavailable on this setup ({e}); continuing with eager execution.")

    net.eval()
    # Always use 'spawn' so worker pools never fork() inside an active multithreaded CUDA / Triton / Queue process
    mp_ctx = mp.get_context("spawn")

    # Comprehensive End-to-End Dry-Run Verification
    if args.dry_run:
        print("\n[Dry-Run] Running comprehensive OceanGo V2 end-to-end verification...")
        print(f"[Dry-Run] Device: {device_name} | AMP: {AMP_DTYPE if AMP_ENABLED else torch.float32} | Layout: {'channels_last' if USE_CHANNELS_LAST else 'contiguous'}")
        print(f"[Dry-Run] Board: {BOARD_SIZE}x{BOARD_SIZE} | Backbone: {net_blocks}x{net_channels} GlobalPoolResNet | Workers: {PARALLEL_THREADS}")

        t0 = time.time()
        dry_workers = min(2, max(1, PARALLEL_THREADS))
        dry_states, dry_stats = run_parallel_self_play_batch(
            net=net,
            board_size=BOARD_SIZE,
            n_games=2,
            n_sims=6,
            num_workers=dry_workers,
            mp_ctx=mp_ctx,
            show_progress=True,
        )
        print(
            f"[Dry-Run] Shared-memory self-play verified: 2 games across {dry_workers} workers -> "
            f"{len(dry_states)} unique bit-packed states in {time.time() - t0:.2f}s."
        )

        add_to_replay_buffer(dry_states)
        print(f"[Dry-Run] CompactReplayBuffer verified: {len(REPLAY_BUFFER)} uint16 states stored.")

        print("[Dry-Run] Testing GPU 8x dihedral symmetry + multi-head (Policy + BCE Value + Ownership) backprop...")
        metrics = train_policy_value_net(
            net,
            replay_buffer=REPLAY_BUFFER,
            batch_size=min(16, max(8, len(REPLAY_BUFFER))),
            lr=args.lr,
            num_steps=2,
        )
        print(
            f"[Dry-Run] Backprop verified: Loss={metrics['loss']:.4f} "
            f"(P={metrics['policy_loss']:.3f}, V={metrics['value_loss']:.3f}, O={metrics['ownership_loss']:.3f})."
        )

        ref_copy = copy.deepcopy(unwrap_model(net)).to(device=device)
        if USE_CHANNELS_LAST:
            ref_copy = ref_copy.to(memory_format=torch.channels_last)
        arena_res = evaluate_arena(net, ref_copy, board_size=BOARD_SIZE, num_games=2, eval_sims=4, intra_batch=2)
        print(f"[Dry-Run] Arena evaluation verified: {arena_res['wins']}W-{arena_res['losses']}L-{arena_res['draws']}D (WR={arena_res['win_rate']*100:.0f}%).")

        os.makedirs(args.save_dir, exist_ok=True)
        test_save_path = os.path.join(args.save_dir, "dry_run_test.pth")
        save_v2_checkpoint(test_save_path, net, board_size=BOARD_SIZE, iteration=max(1, start_iter), stage_iter=stage_iter, elo_est=current_elo)
        reloaded = load_checkpoint_auto(test_save_path)
        assert reloaded["version"] == 2 and reloaded["board_size"] == BOARD_SIZE
        print(f"[Dry-Run] Self-describing V2 checkpoint verified: {test_save_path}")
        print("[Dry-Run] SUCCESS: All V2 subsystems verified and ready for full curriculum training!\n")
        return

    # Interactive Mode Selection Prompt when neither --play nor --train was passed
    should_play = args.play
    if not args.play and not args.train:
        print("\nSelect OceanGo V2 Mode:")
        print("  [1] Play Go as Black (You move first vs AI White) [Default]")
        print("  [2] Play Go as White (AI moves first as Black)")
        print("  [3] Start V2 Curriculum Self-Play Training")
        choice = ""
        try:
            if sys.stdin.isatty():
                choice = input("\nEnter choice (1, 2, or 3, default is 1): ").strip()
            else:
                choice = "1"
        except (EOFError, KeyboardInterrupt):
            choice = "1"

        if choice == "3":
            should_play = False
        elif choice == "2":
            should_play = True
            args.color = "white"
        else:
            should_play = True
            args.color = "black"

    if should_play:
        unwrap_model(net).board_size = BOARD_SIZE
        print(
            f"Starting interactive Pygame window ({BOARD_SIZE}x{BOARD_SIZE}, You play {args.color.capitalize()}, "
            f"{args.play_symmetries}x Symmetry Ensemble, {args.sims} sims/move)..."
        )
        run_interactive_game(net, human_color=args.color, play_symmetries=args.play_symmetries, mcts_sims=args.sims)
        return

    # --- Full Self-Play Reinforcement Learning Loop ---
    checkpoints_dir = args.save_dir
    os.makedirs(checkpoints_dir, exist_ok=True)
    n_games = args.games
    n_sims = args.sims
    n_iters = args.iters
    batch_size = args.batch_size

    optimizer = build_optimizer(net, lr=args.lr)
    train_state_path = os.path.join(checkpoints_dir, f"train_state_v2_{BOARD_SIZE}x{BOARD_SIZE}.pth")

    if start_iter > 0 and not curriculum_transition and os.path.isfile(train_state_path):
        try:
            train_state = torch.load(train_state_path, map_location="cpu", weights_only=False)
            if "optimizer" in train_state:
                optimizer.load_state_dict(train_state["optimizer"])
            if "replay_buffer" in train_state:
                if REPLAY_BUFFER.load_state_dict(train_state["replay_buffer"]):
                    print(f"Restored optimizer momentum and {len(REPLAY_BUFFER):,} compact uint16 states from {train_state_path}.")
        except Exception as e:
            logging.warning(f"Could not restore auxiliary train_state ({e}); continuing with fresh optimizer state.")
    elif curriculum_transition:
        print(
            f"[Curriculum Transition] Initialized fresh {BOARD_SIZE}x{BOARD_SIZE} replay buffer, reset SGD momentum, "
            f"and scheduled 2-iteration LR warmup."
        )

    # Reference model for periodic Arena evaluation & Elo tracking
    ref_net = copy.deepcopy(unwrap_model(net)).to(device=device)
    if USE_CHANNELS_LAST:
        ref_net = ref_net.to(memory_format=torch.channels_last)
    ref_net.eval()
    ref_elo = current_elo

    warmup_tpu_buckets(net)

    total_start_time = time.time()
    if curriculum_transition or stage_iter >= n_iters:
        end_iter = start_iter + n_iters
    else:
        end_iter = start_iter + (n_iters - stage_iter)

    warmup_total = 2 if curriculum_transition else 0

    for local_idx, iteration in enumerate(range(start_iter, end_iter)):
        iteration_start_time = time.time()
        warmup_left = max(0, warmup_total - local_idx)
        current_lr = get_scheduled_lr(stage_iter, base_lr=args.lr, warmup_iters_left=warmup_left, warmup_step=local_idx)

        print(
            f"\n==== Iteration {iteration + 1}/{end_iter} "
            f"(Board: {BOARD_SIZE}x{BOARD_SIZE} | Stage Iter: {stage_iter + 1} | LR: {current_lr:.5f} | Elo: {current_elo:.0f}) ===="
        )
        logging.info(
            f"==== Iteration {iteration + 1}/{end_iter} (Board: {BOARD_SIZE}x{BOARD_SIZE}, StageIter: {stage_iter + 1}, LR: {current_lr:.5f}) ===="
        )

        num_workers = max(1, min(PARALLEL_THREADS, n_games))
        print(f"  Running {n_games} self-play games across {num_workers} shared-memory workers (Full: {n_sims} sims, Fast: {min(n_sims, max(FAST_SEARCH_MIN_SIMS, int(round(FAST_SEARCH_RATIO * n_sims))))} sims)...")

        new_states, sp_stats = run_parallel_self_play_batch(
            net=net,
            board_size=BOARD_SIZE,
            n_games=n_games,
            n_sims=n_sims,
            num_workers=num_workers,
            mp_ctx=mp_ctx,
            show_progress=True,
        )

        add_to_replay_buffer(new_states)

        avg_len = sp_stats["total_moves"] / max(1, sp_stats["games"])
        sp_ent = sp_stats["policy_entropy_sum"] / max(1, sp_stats["policy_entropy_cnt"])
        false_res_rate = (
            100.0 * sp_stats["false_resign_count"] / sp_stats["false_resign_checks"]
            if sp_stats["false_resign_checks"] > 0
            else 0.0
        )
        print(
            f"  Self-Play Telemetry: +{len(new_states):,} unique states (Buffer: {len(REPLAY_BUFFER):,}) | "
            f"Avg Len: {avg_len:.1f} | Resigned: {sp_stats['resigned_games']}/{sp_stats['games']} | "
            f"False-Resign: {false_res_rate:.1f}% ({sp_stats['false_resign_count']}/{sp_stats['false_resign_checks']}) | "
            f"Search Entropy: {sp_ent:.2f}"
        )
        logging.info(
            f"  Self-Play Telemetry: +{len(new_states)} states (Buffer={len(REPLAY_BUFFER)}), "
            f"AvgLen={avg_len:.1f}, Resigned={sp_stats['resigned_games']}/{sp_stats['games']}, "
            f"FalseResign={false_res_rate:.1f}%, SearchEntropy={sp_ent:.3f}"
        )

        # Controlled Replay Ratio: scale gradient steps to ~1.5x newly generated positions (with sensible floor)
        if len(REPLAY_BUFFER) > 0:
            min_samples_floor = min(len(REPLAY_BUFFER) * 4, batch_size * 32)
            target_samples = max(min_samples_floor, int(round(args.replay_ratio * len(new_states))))
            target_steps = max(1, (target_samples + batch_size - 1) // batch_size)
            print(
                f"  Training on {target_steps * batch_size:,} GPU-augmented samples ({target_steps} steps, "
                f"ratio={target_steps * batch_size / max(1, len(new_states)):.2f}x new states)..."
            )
            t_train0 = time.time()
            train_policy_value_net(
                net,
                replay_buffer=REPLAY_BUFFER,
                batch_size=batch_size,
                lr=current_lr,
                optimizer=optimizer,
                num_steps=target_steps,
            )
            logging.info(f"  Training finished in {time.time() - t_train0:.2f}s.")

        stage_iter += 1

        # Periodic Lightweight Arena Evaluation
        if args.eval_interval > 0 and ((iteration + 1) % args.eval_interval == 0):
            eval_sims = min(160, max(60, n_sims // 4))
            print(f"  Running Arena Evaluation ({args.eval_games} paired games @ {eval_sims} sims vs reference model)...")
            t_eval0 = time.time()
            arena = evaluate_arena(
                net,
                ref_net,
                board_size=BOARD_SIZE,
                num_games=args.eval_games,
                eval_sims=eval_sims,
            )
            cand_elo = ref_elo + arena["elo_diff"]
            current_elo = cand_elo
            promoted = ""
            if arena["win_rate"] >= 0.52:
                ref_net.load_state_dict(unwrap_model(net).state_dict())
                ref_elo = cand_elo
                best_stage_path = os.path.join(checkpoints_dir, f"policy_value_net_best_{BOARD_SIZE}x{BOARD_SIZE}.pth")
                save_v2_checkpoint(best_stage_path, net, board_size=BOARD_SIZE, iteration=iteration + 1, stage_iter=stage_iter, elo_est=current_elo)
                promoted = " [PROMOTED NEW REFERENCE]"
            print(
                f"  Arena Result ({time.time() - t_eval0:.1f}s): {arena['wins']}W-{arena['losses']}L-{arena['draws']}D "
                f"(Win Rate: {arena['win_rate']*100:.1f}%, ΔElo: {arena['elo_diff']:+.0f} -> Est. Elo: {current_elo:.0f}){promoted}"
            )
            logging.info(
                f"  Arena: {arena['wins']}W-{arena['losses']}L-{arena['draws']}D "
                f"(WR={arena['win_rate']*100:.1f}%, Elo={current_elo:.1f}){promoted}"
            )

        # Save self-describing V2 checkpoints consistently (including checkpoints_dir/policy_value_net.pth!)
        iter_ckpt_path = os.path.join(checkpoints_dir, f"policy_value_net_iter_{iteration + 1}.pth")
        save_v2_checkpoint(iter_ckpt_path, net, board_size=BOARD_SIZE, iteration=iteration + 1, stage_iter=stage_iter, elo_est=current_elo)
        save_v2_checkpoint("policy_value_net.pth", net, board_size=BOARD_SIZE, iteration=iteration + 1, stage_iter=stage_iter, elo_est=current_elo)
        if os.path.abspath(checkpoints_dir) != os.path.abspath("."):
            save_v2_checkpoint(
                os.path.join(checkpoints_dir, "policy_value_net.pth"),
                net,
                board_size=BOARD_SIZE,
                iteration=iteration + 1,
                stage_iter=stage_iter,
                elo_est=current_elo,
            )

        torch.save(
            {
                "optimizer": get_cpu_optim_state_dict(optimizer),
                "replay_buffer": REPLAY_BUFFER.state_dict(),
            },
            train_state_path,
        )
        print(f"  Saved V2 checkpoint -> {iter_ckpt_path} (Duration: {time.time() - iteration_start_time:.1f}s)")
        logging.info(f"Iteration {iteration + 1} saved to {iter_ckpt_path} in {time.time() - iteration_start_time:.2f}s.")

    total_time = time.time() - total_start_time
    print(f"\nTraining complete in {total_time:.1f}s! Checkpoints saved to '{checkpoints_dir}'.")
    logging.info(f"Total training time: {total_time:.2f}s.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
