#!/usr/bin/env python3
# OceanGo - 9×9 (default) Go with GPU training & CPU self-play
# Features: progress bars, AMP, simple-Ko, suicide ban, GUI.
# --------------------------------------------------------------------------
import datetime
import platform
import time
import os
import random
import math
import warnings
import multiprocessing
import subprocess
import logging
import logging.handlers
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional   # ★ NEW Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm

try:
    import psutil            # for RAM read-out
except ImportError:
    psutil = None

# ────────────── silence pygame banner ──────────────
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", category=UserWarning,
                        module="pygame.pkgdata")
pygame = None        # lazy import inside GUI

# ───────────────────────── CONFIG ─────────────────────────
@dataclass
class Config:
    # ── 1. Board & GUI ────────────────────────────────────────
    board_size: int = 9        # Size of the Go board (NxN).
                               # ↑ : tensors grow ∝ N², MCTS gets slower,
                               #     GPU VRAM & CPU time up sharply.
                               # Resource  → CPU-time, GPU-VRAM.

    grid: int = 50             # Pixel width of one cell in the GUI.
                               # Only affects rendering - no perf impact.

    # ── 2. Self-play schedule (how much data you generate) ────
    n_iter: int = 60           # Outer training loops per run.
    games_per_iter: int = 40  # Parallel self-play games each loop.

    # ── 3. MCTS search depth ─────────────────────────────────
    mcts_sims: int = 5_000                # Used in GUI & training
    mcts_sims_worker: int = 1_200         # ★ NEW lighter CPU self-play

    # ── 4. Replay buffer (lives in **system RAM**) ────────────
    replay_cap: int = 40_000   # Maximum positions kept.

    # ── 5. SGD / GPU training parameters ─────────────────────
    train_batch: int = 512
    train_epochs: int = 5
    lr: float = 1e-4

    # ── 6. Hardware toggles ───────────────────────────────────
    amp: bool = True           # Automatic Mixed Precision on GPU.
    num_workers: int = 12      # CPU processes that run self-play.

    # ── 7. Robustness knobs (watch-dog & move cap) ────────────
    max_moves: Optional[int] = None              # Per-game hard cap; None → unlimited.
    worker_timeout: Optional[int] = 6 * 3600     # ★ MOD 6 h; None → disable.


# ───────────────────────── TQDM HELPERS ─────────────────────────
from functools import partial

tqdm_bar = partial(
    tqdm,
    dynamic_ncols=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} "
               "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
)

# Make tqdm output from multiple processes cooperate ★ NEW
tqdm.set_lock(multiprocessing.RLock())

CFG = Config()
if CFG.max_moves is None:                       # ★ NEW reasonable default
    CFG.max_moves = 2 * CFG.board_size * CFG.board_size   # 2 × board area

SCREEN = (CFG.board_size + 1) * CFG.grid
HEADER_WRITTEN = False
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable fast Tensor-Core paths on Ampere+ GPUs ★ NEW
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

root  = Path(__file__).resolve().parent
# ───────────────────────── LOGGING ─────────────────────────
def init_logger():
    fmt   = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datef = "%Y-%m-%d %H:%M:%S"
    logf  = root / "ocean_go.log"

    # Rotate at 10 MB, keep 5 archives  ocean_go.log.1 … .5
    handler = logging.handlers.RotatingFileHandler(
        logf, maxBytes=10*2**20, backupCount=5, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(fmt, datef))

    logging.basicConfig(
        level=logging.INFO, handlers=[handler], force=True
    )

    logging.getLogger("torch").setLevel(logging.WARNING)   # silence spam
    return logging.getLogger("OceanGo")

LOGGER = init_logger()

# --------------------------------------------------------------------------
#                              HEADERS
# --------------------------------------------------------------------------
def log_super_header():
    global HEADER_WRITTEN
    if HEADER_WRITTEN:          # already done in this interpreter
        return
    if multiprocessing.current_process().name != "MainProcess":
        return                  # child process → skip

    HEADER_WRITTEN = True       # flip the switch

    header = {
        "session_id":  os.urandom(4).hex(),
        "script"    :  os.path.basename(__file__),
        "git_rev"   :  subprocess.run(
                          ["git", "rev-parse", "--short", "HEAD"],
                          text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL).stdout.strip() or "N/A",
        "started_at":  datetime.datetime.now()
                           .isoformat(timespec="seconds"),
        "python"    :  platform.python_version(),
    }
    for k, v in header.items():
        LOGGER.info("%-11s : %s", k, v)

# run immediately once in the main interpreter
log_super_header()

def log_session_header():
    import torch, platform, subprocess, uuid, datetime, socket, psutil
    header = {
        "session_id" : uuid.uuid4().hex[:8],
        "script"     : Path(__file__).name,
        "git_rev"    : subprocess.run(
            ["git","rev-parse","--short","HEAD"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        ).stdout.strip() or "N/A",
        "started_at" : datetime.datetime.now().isoformat(timespec="seconds"),
        "host"       : socket.gethostname(),
        "cuda"       : torch.version.cuda,
        "gpu"        : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "torch"      : torch.__version__,
        "python"     : platform.python_version(),
        "ram_total"  : f"{psutil.virtual_memory().total/2**30:.1f} GB",
        "config"     : vars(CFG),
    }
    for k, v in header.items():
        LOGGER.info("%-11s : %s", k, v)

    LOGGER.info("")

log_session_header()

# ==========================================================
#                   POLICY-VALUE NETWORK
# ==========================================================
# ───────────────────── NN building blocks ────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)

# ───────────────────── Policy-value network ──────────────────
class PolicyValueNet(nn.Module):
    def __init__(self, bs: int = CFG.board_size, ch: int = 128, blocks: int = 6):
        super().__init__()
        self.bs = bs
        self.entry = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )
        self.torso = nn.Sequential(*(ResBlock(ch) for _ in range(blocks)))

        # policy head
        self.p_conv = nn.Conv2d(ch, 2, 1)
        self.p_fc   = nn.Linear(2 * bs * bs, bs * bs + 1)

        # value head
        self.v_conv = nn.Conv2d(ch, 1, 1)
        self.v_fc1  = nn.Linear(bs * bs, 256)
        self.v_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        x = self.torso(self.entry(x))
        p = F.relu(self.p_conv(x))
        p = self.p_fc(p.flatten(1))
        v = F.relu(self.v_conv(x))
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = torch.tanh(self.v_fc2(v))
        return p, v

# ==========================================================
#                     GO GAME LOGIC
# ==========================================================
def softmax_np(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def _chinese_area(board: np.ndarray, komi: float = 6.5) -> float:
    """Return (black_area - white_area) with komi already subtracted.
    Positive → black is ahead, negative → white is ahead."""
    bs = board.shape[0]
    seen = set()
    area_black = area_white = 0

    for x in range(bs):
        for y in range(bs):
            if (x, y) in seen:                 # already visited
                continue

            c = board[x, y]
            if c in (1, 2):                    # occupied
                if c == 1:
                    area_black += 1
                else:
                    area_white += 1
                continue

            # flood-fill an empty region
            queue = [(x, y)]
            empties = []
            border_colors = set()
            while queue:
                cx, cy = queue.pop()
                if (cx, cy) in seen:
                    continue
                seen.add((cx, cy))
                empties.append((cx, cy))
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < bs and 0 <= ny < bs:
                        col = board[nx, ny]
                        if col == 0:
                            queue.append((nx, ny))
                        else:
                            border_colors.add(col)

            if len(border_colors) == 1:        # territory
                if 1 in border_colors:
                    area_black += len(empties)
                else:
                    area_white += len(empties)
            # else: dame — nobody gets the points

    return area_black - (area_white + komi)


# ==========================================================
#  PASS move helper
# ==========================================================
PASS = (-1, -1)         # one extra action after the board cells


class GoGame:
    def __init__(self, bs: int = CFG.board_size):
        self.bs = bs
        self.board = np.zeros((bs, bs), np.int8)
        self.current_player = 1
        self.history: List[np.ndarray] = []   # snapshots for Ko
        self.ko_point: Tuple[int, int] | None = None

    # -------------- internal helpers ----------------
    def copy(self):
        g = GoGame(self.bs)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.history = [b.copy() for b in self.history]
        g.ko_point = self.ko_point
        return g

    def switch(self):
        self.current_player = 3 - self.current_player

    def is_valid(self, x, y):
        return 0 <= x < self.bs and 0 <= y < self.bs and self.board[x, y] == 0

    def _has_liberty(self, start, temp_board):
        from collections import deque
        color = temp_board[start]
        seen, dq = set(), deque([start])
        while dq:
            cx, cy = dq.pop()
            if (cx, cy) in seen:
                continue
            seen.add((cx, cy))
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.bs and 0 <= ny < self.bs:
                    if temp_board[nx, ny] == 0:
                        return True
                    if temp_board[nx, ny] == color and (nx, ny) not in seen:
                        dq.append((nx, ny))
        return False

    def is_suicide(self, x, y):
        if not self.is_valid(x, y): return False
        tmp = self.board.copy()
        tmp[x, y] = self.current_player
        self._remove_captured(3 - self.current_player, tmp)
        return not self._has_liberty((x, y), tmp)

    # -------------- captures & eyes ------------------
    def _find_group(self, start, board):
        from collections import deque
        color = board[start]
        group, dq, lib = set(), deque([start]), set()
        while dq:
            cx, cy = dq.pop()
            if (cx, cy) in group: continue
            group.add((cx, cy))
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.bs and 0 <= ny < self.bs:
                    if board[nx, ny] == 0:
                        lib.add((nx, ny))
                    elif board[nx, ny] == color and (nx, ny) not in group:
                        dq.append((nx, ny))
        return group, lib

    def _remove_captured(self, color, board=None):
        if board is None:
            board = self.board
        captured = []
        visited = set()
        for i in range(self.bs):
            for j in range(self.bs):
                if board[i, j] == color and (i, j) not in visited:
                    grp, lib = self._find_group((i, j), board)
                    visited |= grp
                    if not lib:
                        captured.extend(grp)
        for x, y in captured:
            board[x, y] = 0

    # -------------- move with simple-Ko check ----------
    def make_move(self, x, y):
        if (x, y) == PASS:           # pass → just toggle the player
            self.history.append(self.board.copy())
            self.switch()
            return True

        if not self.is_valid(x, y): return False
        if self.ko_point == (x, y): return False

        prev_board = self.board.copy()
        self.board[x, y] = self.current_player
        self._remove_captured(3 - self.current_player)

        # suicide?
        if not self._has_liberty((x, y), self.board):
            self.board[:] = prev_board
            return False

        # simple-Ko: board identical to position 2 plies ago?
        if self.history and np.array_equal(self.board, self.history[-1]):
            self.board[:] = prev_board
            return False

        # update Ko point (single-stone Ko)
        self.ko_point = None
        diff = np.where(prev_board != self.board)
        if diff[0].size == 1:  # one stone difference
            cx, cy = diff[0][0], diff[1][0]
            if prev_board[cx, cy] != 0 and self.board[cx, cy] == 0:
                self.ko_point = (cx, cy)

        self.history.append(prev_board)       # keep last state
        self.switch()
        return True
    
    def _last_two_passed(self):
        return (
            len(self.history) >= 2
            and np.array_equal(self.history[-1], self.history[-2])
        )

    # ---------- convenience for GUI / debug ------------
    def group_stats(self, x, y):
        """Return (#liberties, is_eye) for group containing (x,y)."""
        if self.board[x, y] == 0:
            return 0, False
        grp, lib = self._find_group((x, y), self.board)
        # eye heuristic: every liberty is fully surrounded by same color
        col = self.board[x, y]
        eye = all(all(self.board[nx + dx, ny + dy] == col
                       for dx, dy in ((1,0),(-1,0),(0,1),(0,-1))
                       if 0 <= nx + dx < self.bs and 0 <= ny + dy < self.bs)
                  for nx, ny in lib)
        return len(lib), eye

    # ---------- external interface ----------
    def get_legal_moves(self):
        moves = [(x, y) for x in range(self.bs) for y in range(self.bs)
                if self.board[x, y] == 0
                and not self.is_suicide(x, y)
                and (x, y) != self.ko_point]
        moves.append(PASS)        # <- brand-new line
        return moves

    def game_over(self):
        return self._last_two_passed()

    def winner(self, komi: float = 6.5):
        score = _chinese_area(self.board, komi)
        if abs(score) < 1e-3:
            return 0                 # draw
        return 1 if score > 0 else 2


    def state_tensor(self):
        p, o = self.current_player, 3 - self.current_player
        return torch.from_numpy(np.stack([(self.board == p).astype(np.float32),
                                          (self.board == o).astype(np.float32)], 0))

# ==========================================================
#                     MCTS  (NN-guided)
# ==========================================================
class Node:
    __slots__ = ("g", "p", "mv", "ch", "unexp", "N", "W", "Q", "P", "val")
    def __init__(self, g, parent=None, mv=None):
        self.g, self.p, self.mv = g, parent, mv
        self.ch: Dict[Tuple[int, int], "Node"] = {}
        self.unexp = g.get_legal_moves()
        self.N = self.W = self.Q = 0.0
        self.P = 0.0
        self.val = 0.0

def select(node: 'Node'):
    while True:
        if node.unexp or not node.ch:
            return node
        node = max(node.ch.values(),
                   key=lambda c: c.Q + 1.0 * c.P * math.sqrt(node.N) / (1 + c.N))

@torch.no_grad()
def nn_eval(batch: List[Node], net: PolicyValueNet):
    """Evaluate *many* leaves at once on the GPU (or CPU)."""
    if not batch:
        return
    dev = next(net.parameters()).device          # will now be cuda:0
    with torch.autocast(device_type=dev.type,
                        enabled=(CFG.amp and dev.type == "cuda"),
                        dtype=torch.bfloat16):
        x = torch.stack([n.g.state_tensor() for n in batch]).to(dev, non_blocking=True)
        logit, v = net(x)
    logit = logit.float().cpu().numpy()
    v = v.float().squeeze(1).cpu().numpy()
    for n, lvec, val in zip(batch, logit, v):
        n.val = float(val)
        if not n.unexp: continue
        idx = []
        for m in n.unexp:
            if m == PASS:
                idx.append(CFG.board_size ** 2)   # last slot
            else:
                idx.append(m[0] * CFG.board_size + m[1])

        pr = softmax_np(lvec[:CFG.board_size**2 + 1][idx])
        order = list(zip(n.unexp, pr))
        random.shuffle(order)
        for m, p in order:
            ch = Node(n.g.copy(), n, m); ch.P = float(p)
            n.ch[m] = ch
        n.unexp.clear()

def backup(n: Node, value: float):
    while n:
        n.N += 1
        n.W += value
        n.Q = n.W / n.N
        value = -value
        n = n.p

def mcts(game: GoGame, net: PolicyValueNet,
         sims: int = CFG.mcts_sims, temp: float = 1.0,
         batch_size: int = 128):
    
    root = Node(game.copy())
    nn_eval([root], net)

    if root is not None:
        e = 0.25
        a = 0.03
        dir_noise = np.random.dirichlet([a] * len(root.ch))
        for (mv, ch), n in zip(root.ch.items(), dir_noise):
            ch.P = (1 - e) * ch.P + e * n

    leaves: List[Node] = []
    for _ in range(sims):
        leaf = select(root)
        leaves.append(leaf)
        if len(leaves) >= batch_size:
            nn_eval(leaves, net)
            for lf in leaves:
                backup(lf, lf.val)
            leaves.clear()

    if leaves:
        nn_eval(leaves, net)
        for lf in leaves:
            backup(lf, lf.val)

    moves, visits = zip(*((m, ch.N) for m, ch in root.ch.items()))
    v = np.array(visits, np.float32)
    if temp < 1e-3:
        p = np.zeros_like(v); p[np.argmax(v)] = 1.0
    else:
        w = v ** (1 / temp); p = w / w.sum()
    return dict(zip(moves, p))

# ==========================================================
#              SELF-PLAY WORKER  (CPU tree, GPU NN)
# ==========================================================
def _strip_prefix(state_dict, prefix="_orig_mod."):
    """Remove the torch.compile() prefix from keys, if present."""
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    clean = {}
    for k, v in state_dict.items():            # keep the value!
        new_k = k[len(prefix):] if k.startswith(prefix) else k
        clean[new_k] = v
    return clean

def self_play_worker(args):
    rank, seed, model_path = args
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    net = PolicyValueNet(CFG.board_size).eval().to("cuda", non_blocking=True)
    state = torch.load(model_path, map_location="cpu")
    state = _strip_prefix(state)
    net.load_state_dict(state)

    show_bar = (rank == 0)
    if show_bar:
        moves_bar = tqdm_bar(total=None,
                             desc=f"Game {rank} moves",
                             unit="move",
                             position=1,
                             leave=False)

    LOGGER.debug("Worker %d  spawned (PID=%d, seed=%d)", rank, os.getpid(), seed)

    game, records = GoGame(CFG.board_size), []
    move_ctr = 0                                            # ★ NEW
    sims_per_move = CFG.mcts_sims_worker or CFG.mcts_sims   # ★ NEW

    while not game.game_over() and (
          CFG.max_moves is None or move_ctr < CFG.max_moves):
        if show_bar: moves_bar.update()

        pi = mcts(game, net, sims_per_move, 1.0)
        if not pi: break                                    # safety: no legal

        vec = np.zeros(CFG.board_size**2 + 1, np.float32)   # +1 for PASS
        for mv, p in pi.items():
            if mv == PASS:
                vec[-1] = p
            else:
                vec[mv[0] * CFG.board_size + mv[1]] = p

        records.append((game.board.copy(), game.current_player, vec))
        mv = random.choices(list(pi.keys()), weights=list(pi.values()))[0]
        game.make_move(*mv)
        move_ctr += 1                                       # ★ NEW

    if CFG.max_moves and move_ctr >= CFG.max_moves:
        LOGGER.warning("Worker %d hit max_moves → forcing end", rank)
    if show_bar: moves_bar.close()

    w = game.winner()
    LOGGER.info("Worker %d finished: moves=%d, winner=%s",
            rank, len(records), {0:"Draw",1:"Black",2:"White"}[w])
    return [(b, p, v, 0 if w == 0 else (1 if p == w else -1))
            for b, p, v in records]

# ==========================================================
#                     REPLAY BUFFER
# ==========================================================
class Replay:
    def __init__(self, cap): self.data, self.cap = [], cap
    def add(self, new): self.data.extend(new); self.data = self.data[-self.cap:]
    def sample(self, n): return self.data if n >= len(self.data) else random.sample(self.data, n)
RB = Replay(CFG.replay_cap)

# ==========================================================
#                       TRAINING
# ==========================================================
def planes(board, player):
    return torch.from_numpy(np.stack([(board == player).astype(np.float32),
                                      (board == 3 - player).astype(np.float32)], 0))

def batch_tensor(batch):
    bds, pl, pi, z = zip(*batch)
    X = torch.stack([planes(b, p) for b, p in zip(bds, pl)]).to(device_gpu)
    pi_np = np.stack(pi, axis=0).astype(np.float32)
    z_np  = np.asarray(z, dtype=np.float32)[:, None]
    return X, torch.from_numpy(pi_np).to(device_gpu), torch.from_numpy(z_np).to(device_gpu)

def train(net: PolicyValueNet, batch, it_idx):
    LOGGER.info(                        # ①  session-level headline
        "Iter %d | start training | batch=%d, epochs=%d, lr=%.1e",
        it_idx, len(batch), CFG.train_epochs, CFG.lr,
    )
    opt    = optim.Adam(net.parameters(), lr=CFG.lr)
    scaler = torch.amp.GradScaler(device='cuda', enabled=CFG.amp)
    net.train()

    X, pi_t, z_t = batch_tensor(batch)
    steps_per_ep = math.ceil(len(batch) / CFG.train_batch)
    total_steps  = CFG.train_epochs * steps_per_ep

    bar = tqdm_bar(
        total=CFG.train_epochs * steps_per_ep,
        desc=f"Train  |  Epoch 1/{CFG.train_epochs}",
        unit="step",
        position=0,
        leave=False,
    )

    running, step_global = 0.0, 0
    for ep in range(1, CFG.train_epochs + 1):
        idx = np.arange(len(batch)); np.random.shuffle(idx)
        for s in range(0, len(batch), CFG.train_batch):
            samp = idx[s:s+CFG.train_batch]
            inp, tgt_pi, tgt_z = X[samp], pi_t[samp], z_t[samp]

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=CFG.amp,
                                dtype=torch.bfloat16):
                out_p, out_v = net(inp)
                loss_p = -(tgt_pi * F.log_softmax(out_p, 1)).sum(1).mean()
                loss_v = F.mse_loss(out_v, tgt_z)
                loss   = loss_p + loss_v
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

            # running loss & UI
            step_global += 1
            running += (loss.item() - running) / step_global
            vram = torch.cuda.memory_allocated() / 2**20
            if step_global == 1 or step_global % 50 == 0:
                LOGGER.debug(
                    "Iter %d | Ep %d | step %4d/%d | loss=%.4f (avg=%.4f)",
                    it_idx, ep, step_global, total_steps, loss.item(), running,
                    extra={"vram": torch.cuda.memory_allocated() // 2**20},
                )
            bar.set_postfix(loss=f"{loss.item():.4f}",
                            avg=f"{running:.4f}",
                            vram=f"{vram:.0f} MB")
            bar.update()

        if ep < CFG.train_epochs:
            bar.set_description(f"Iter {it_idx} | Epoch {ep+1}/{CFG.train_epochs}")

    LOGGER.info("Iter %d finished - avg_loss=%.4f (batch=%d)",  it_idx, running, len(batch))

    bar.close()
    print(f"#  Training finished — avg loss {running:.4f}")
    logging.info("Iter %d | Final avg loss %.4f", it_idx, running)
    net.eval()

# ==========================================================
#                         GUI
# ==========================================================
def run_gui(net: PolicyValueNet):
    global pygame
    import pygame as _pg; pygame = _pg

    pygame.init()
    screen = pygame.display.set_mode((SCREEN, SCREEN))
    pygame.display.set_caption("OceanGo")

    def draw(g: GoGame):
        screen.fill((240, 207, 155))
        for i in range(1, CFG.board_size + 1):
            pygame.draw.line(screen, (0,0,0),
                             (i * CFG.grid, CFG.grid),
                             (i * CFG.grid, SCREEN - CFG.grid), 2)
            pygame.draw.line(screen, (0,0,0),
                             (CFG.grid, i * CFG.grid),
                             (SCREEN - CFG.grid, i * CFG.grid), 2)
        for x in range(CFG.board_size):
            for y in range(CFG.board_size):
                s = g.board[x, y]
                if s:
                    col = (0,0,0) if s == 1 else (240,240,240)
                    pygame.draw.circle(screen, col,
                                       ((y + 1) * CFG.grid, (x + 1) * CFG.grid),
                                       CFG.grid // 2 - 4)
        pygame.display.flip()

    game, clock = GoGame(CFG.board_size), pygame.time.Clock()
    while True:
        if game.game_over():
            print("Winner:", game.winner())
            pygame.time.wait(3000); return
        draw(game); clock.tick(30)

        if game.current_player == 1:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: return
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = ev.pos
                    gx, gy = round(my / CFG.grid - 1), round(mx / CFG.grid - 1)
                    game.make_move(gx, gy)
        else:
            probs = mcts(game, net, CFG.mcts_sims, temp=0.1)
            if not probs: print("AI passes."); return
            best = max(probs.items(), key=lambda kv: kv[1])[0]
            game.make_move(*best)

# ==========================================================
#                           MAIN
# ==========================================================
def main():
    print("OceanGo — CUDA:", torch.cuda.is_available(), "- AMP:", CFG.amp)
    torch.backends.cudnn.benchmark = True

    raw_net = PolicyValueNet().to("cuda")           # ← the parameters live here
    net      = torch.compile(raw_net, backend="inductor")  # fast wrapper
    print("Raw net device :", next(raw_net.parameters()).device)
    print("Comp net device:", next(net.parameters()).device)
    print("------- start self-play -------")


    # Add CFG info in file name for checkpoint
    cfg_info = (f"R6_B{CFG.board_size}_N{CFG.n_iter}"
            f"_GPI{CFG.games_per_iter}_MS{CFG.mcts_sims}"
            f"_RC{CFG.replay_cap}_TB{CFG.train_batch}"
            f"_TE{CFG.train_epochs}_LR{CFG.lr:.0e}")
    ckpt = root / f"policy_value_net_{cfg_info}.pth"
    print("Checkpoint:", ckpt)
    do_train = True
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cuda")
        cur = net.state_dict(); cur.update(state); net.load_state_dict(cur, strict=False)
        print("Found model, loaded.")
        try:
            if input("Train further (T) or play GUI (G)? [T/G]: ").strip().lower() == "g":
                do_train = False
        except EOFError:
            pass
    else:
        print("No checkpoint — starting fresh.")

    if not do_train:
        run_gui(net); return

    tmp = root / "_tmp_cpu.pth"
    torch.save(net.cpu().state_dict(), tmp)  # workers need CPU copy
    net.to("cuda")

    outer = tqdm_bar(
        range(1, CFG.n_iter + 1),
        desc="Meta-loop",
        unit="iter",
        position=0,
        leave=True,
    )

    for it in outer:
        seeds = [(rank, random.randrange(2**32))
                 for rank in range(CFG.games_per_iter)]
        print(f"\n->  Iteration {it}/{CFG.n_iter} — self-play")
        bar = tqdm_bar(total=CFG.games_per_iter,
                       desc="Self-play",
                       unit="game",
                       position=0,
                       leave=False)
        moves_total, t0 = 0, time.time()

        with multiprocessing.Pool(CFG.num_workers) as pool:
            async_results = [pool.apply_async(self_play_worker,
                              ((r, s, tmp.as_posix()),))
                              for r, s in seeds]

            for ar in async_results:
                try:
                    rec = ar.get(timeout=CFG.worker_timeout)
                except multiprocessing.TimeoutError:
                    LOGGER.error("A worker timed out - game dropped "
                                 "(consider raising Config.worker_timeout "
                                 "or lowering mcts_sims_worker).")
                    continue                                  # ★ MOD keep loop alive
                RB.add(rec); bar.update()
                moves_total += len(rec)
                bar.set_postfix_str(
                    f"buffer={len(RB.data):,}  "
                    f"moves={moves_total:,}  "
                    f"mps={moves_total/(time.time()-t0):.1f}"
                )
            bar.close()

        LOGGER.info(
            "Iter %d summary: games=%d  moves=%d  buffer=%d  wall=%.1fs",
            it, CFG.games_per_iter, moves_total, len(RB.data),
            time.time() - t0,
        )

        train(net, RB.sample(1024), it)
        torch.save(net.state_dict(), ckpt)            # GPU weights
        torch.save(net.cpu().state_dict(), tmp)       # worker copy
        net.to("cuda")

        if psutil:
            mem_gb = psutil.virtual_memory().used / 2**30
            outer.set_postfix_str(f"buffer={len(RB.data):,}  RAM={mem_gb:.1f} GB")
        else:
            outer.set_postfix_str(f"buffer={len(RB.data):,}")
        outer.update()

    run_gui(net)

# ───────────────────────── entry ─────────────────────────
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
