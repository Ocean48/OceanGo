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
from typing import List, Tuple, Dict, Optional

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
    # ── 1. Board & Game Rules ─────────────────────────────────
    board_size: int = 9        # Size of the Go board (NxN).
    komi: float = 6.5          # Points added to White's score.

    # ── 2. GUI Settings ───────────────────────────────────────
    grid: int = 50             # Pixel width of one cell in the GUI.

    # ── 3. Self-play schedule (how much data you generate) ────
    n_iter: int = 60           # Outer training loops per run.
    games_per_iter: int = 40   # Parallel self-play games each loop.

    # ── 4. MCTS search depth ─────────────────────────────────
    mcts_sims: int = 5_000                # Used in GUI & training
    mcts_sims_worker: int = 1_200         # Lighter CPU self-play

    # ── 5. Replay buffer (lives in **system RAM**) ────────────
    replay_cap: int = 40_000   # Maximum positions kept.
    train_sample_size: int = 4096 # Positions sampled for training.

    # ── 6. SGD / GPU training parameters ─────────────────────
    train_batch: int = 512
    train_epochs: int = 5
    lr: float = 1e-4

    # ── 7. Hardware toggles ───────────────────────────────────
    amp: bool = True           # Automatic Mixed Precision on GPU.
    num_workers: int = 12      # CPU processes that run self-play.

    # ── 8. Robustness knobs (watch-dog & move cap) ────────────
    max_moves: Optional[int] = None              # Per-game hard cap; None → unlimited.
    worker_timeout: Optional[int] = 6 * 3600     # 6 hours; None → disable.


# ───────────────────────── TQDM HELPERS ─────────────────────────
from functools import partial

tqdm_bar = partial(
    tqdm,
    dynamic_ncols=True,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} "
               "[{elapsed}<{remaining}, {rate_fmt}{postfix}]",
)

# Make tqdm output from multiple processes cooperate
tqdm.set_lock(multiprocessing.RLock())

CFG = Config()
if CFG.max_moves is None:
    CFG.max_moves = 2 * CFG.board_size * CFG.board_size   # 2 × board area

SCREEN = (CFG.board_size + 1) * CFG.grid
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable fast Tensor-Core paths on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

root  = Path(__file__).resolve().parent
# ───────────────────────── LOGGING ─────────────────────────
def init_logger():
    fmt   = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datef = "%Y-%m-%d %H:%M:%S"
    logf  = root / "    _go.log"

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
def log_session_header():
    """Logs a detailed header for the session. Runs only once in the main process."""
    if multiprocessing.current_process().name != "MainProcess":
        return
    if getattr(log_session_header, "has_run", False):
        return
    setattr(log_session_header, "has_run", True)

    import torch, platform, subprocess, uuid, datetime, socket
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
        "ram_total"  : f"{psutil.virtual_memory().total/2**30:.1f} GB" if psutil else "N/A",
        "config"     : vars(CFG),
    }
    for k, v in header.items():
        LOGGER.info("%-11s : %s", k, v)
    LOGGER.info("-" * 80)


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
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

# ───────────────── Improved Policy-value network ───────────────
class PolicyValueNet(nn.Module):
    def __init__(self, bs: int = CFG.board_size, ch: int = 128, blocks: int = 6):
        super().__init__()
        self.bs = bs
        self.entry = nn.Sequential(
            nn.Conv2d(2, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )
        self.torso = nn.Sequential(*(ResBlock(ch) for _ in range(blocks)))

        # Policy head (with BatchNorm)
        self.p_conv = nn.Conv2d(ch, 4, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(4)
        self.p_fc = nn.Linear(4 * bs * bs, bs * bs + 1)

        # Value head (with BatchNorm)
        self.v_conv = nn.Conv2d(ch, 2, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(2)
        self.v_fc1 = nn.Linear(2 * bs * bs, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.torso(self.entry(x))
        
        # Policy head
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = self.p_fc(p.flatten(1))

        # Value head
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = F.relu(self.v_fc1(v.flatten(1)))
        v = torch.tanh(self.v_fc2(v))
        return p, v


# ==========================================================
#                     GO GAME LOGIC
# ==========================================================
def softmax_np(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def _chinese_area(board: np.ndarray, komi: float = CFG.komi) -> float:
    """Return (black_area - white_area) with komi already subtracted.
    Positive → black is ahead, negative → white is ahead."""
    bs = board.shape[0]
    seen = np.zeros_like(board, dtype=bool)
    area_black = area_white = 0

    for x in range(bs):
        for y in range(bs):
            if seen[x, y]:
                continue

            c = board[x, y]
            if c in (1, 2):
                if c == 1: area_black += 1
                else: area_white += 1
                continue

            # Flood-fill an empty region
            q = [(x, y)]
            empties = []
            border_colors = set()
            while q:
                cx, cy = q.pop()
                if seen[cx, cy]: continue
                seen[cx, cy] = True
                empties.append((cx, cy))
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < bs and 0 <= ny < bs:
                        col = board[nx, ny]
                        if col == 0:
                            q.append((nx, ny))
                        else:
                            border_colors.add(col)

            if len(border_colors) == 1:
                if 1 in border_colors: area_black += len(empties)
                else: area_white += len(empties)

    return area_black - (area_white + komi)

PASS = (-1, -1)

class GoGame:
    def __init__(self, bs: int = CFG.board_size):
        self.bs = bs
        self.board = np.zeros((bs, bs), np.int8)
        self.current_player = 1
        self.history: List[np.ndarray] = []
        self.ko_point: Optional[Tuple[int, int]] = None

    @property
    def opponent_player(self) -> int:
        return 3 - self.current_player

    def copy(self):
        g = GoGame(self.bs)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.history = [b.copy() for b in self.history]
        g.ko_point = self.ko_point
        return g

    def switch(self):
        self.current_player = self.opponent_player

    def is_valid(self, x, y):
        return 0 <= x < self.bs and 0 <= y < self.bs and self.board[x, y] == 0

    def _has_liberty(self, start, temp_board):
        color = temp_board[start]
        q, seen = [start], {start}
        while q:
            cx, cy = q.pop()
            for dx, dy in ((1,0),(-1,0),(0,1),(-1,0)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.bs and 0 <= ny < self.bs:
                    if temp_board[nx, ny] == 0: return True
                    if temp_board[nx, ny] == color and (nx, ny) not in seen:
                        q.append((nx, ny)); seen.add((nx,ny))
        return False

    def is_suicide(self, x, y):
        tmp = self.board.copy()
        tmp[x, y] = self.current_player
        self._remove_captured(self.opponent_player, tmp)
        return not self._has_liberty((x, y), tmp)

    def _find_group(self, start, board):
        color = board[start]
        group, lib, q, seen = set(), set(), [start], {start}
        while q:
            cx, cy = q.pop()
            group.add((cx, cy))
            for dx, dy in ((1,0),(-1,0),(0,1),(-1,0)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.bs and 0 <= ny < self.bs:
                    if board[nx, ny] == 0: lib.add((nx, ny))
                    elif board[nx, ny] == color and (nx, ny) not in seen:
                        q.append((nx, ny)); seen.add((nx,ny))
        return group, lib

    def _remove_captured(self, color, board=None):
        if board is None: board = self.board
        captured_stones = []
        visited = set()
        for i in range(self.bs):
            for j in range(self.bs):
                if board[i, j] == color and (i, j) not in visited:
                    grp, lib = self._find_group((i, j), board)
                    visited.update(grp)
                    if not lib: captured_stones.extend(grp)
        for x, y in captured_stones: board[x, y] = 0
        return len(captured_stones)

    def make_move(self, x, y):
        if (x, y) == PASS:
            self.history.append(self.board.copy())
            self.switch()
            return True

        if not self.is_valid(x, y) or self.ko_point == (x, y):
            return False

        prev_board = self.board.copy()
        self.board[x, y] = self.current_player
        num_captured = self._remove_captured(self.opponent_player)

        if not self._has_liberty((x, y), self.board):
            self.board = prev_board
            return False # Suicide is illegal

        # Correct simple-Ko check: board cannot be identical to 2 plies ago
        if len(self.history) >= 2 and np.array_equal(self.board, self.history[-2]):
            self.board = prev_board
            return False

        # Update Ko point (only for single stone captures)
        self.ko_point = None
        if num_captured == 1:
            diff = np.where(prev_board != self.board)
            if diff[0].size == 2: # One stone placed, one removed
                # Find the removed stone's location
                removed_mask = (prev_board != 0) & (self.board == 0)
                if np.sum(removed_mask) == 1:
                    ko_x, ko_y = np.argwhere(removed_mask)[0]
                    self.ko_point = (ko_x, ko_y)

        self.history.append(prev_board)
        self.switch()
        return True

    def _last_two_passed(self):
        return (len(self.history) >= 2 and
                np.array_equal(self.history[-1], self.history[-2]))

    def get_legal_moves(self):
        moves = []
        for x in range(self.bs):
            for y in range(self.bs):
                if self.board[x, y] == 0 and (x, y) != self.ko_point:
                    # Optimized check: avoid full suicide check unless necessary
                    # A move is only suicidal if it has no liberties and captures nothing.
                    has_liberty = False
                    captures_something = False
                    for dx, dy in ((1,0),(-1,0),(0,1),(-1,0)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.bs and 0 <= ny < self.bs:
                            if self.board[nx, ny] == 0:
                                has_liberty = True; break
                            if self.board[nx, ny] == self.opponent_player:
                                _, libs = self._find_group((nx, ny), self.board)
                                if len(libs) == 1:
                                    captures_something = True; break
                    
                    if has_liberty or captures_something:
                        moves.append((x, y))
                    elif not self.is_suicide(x, y):
                        moves.append((x, y))

        moves.append(PASS)
        return moves

    def game_over(self): return self._last_two_passed()

    def winner(self, komi: float = CFG.komi):
        score = _chinese_area(self.board, komi)
        if abs(score) < 1e-3: return 0
        return 1 if score > 0 else 2

    def state_tensor(self):
        p, o = self.current_player, self.opponent_player
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
        self.N = self.W = self.Q = self.P = self.val = 0.0

def select(node: 'Node'):
    while True:
        if node.unexp or not node.ch: return node
        node = max(node.ch.values(),
                   key=lambda c: c.Q + 1.0 * c.P * math.sqrt(node.N) / (1 + c.N))

@torch.no_grad()
def nn_eval(batch: List[Node], net: PolicyValueNet):
    if not batch: return
    dev = next(net.parameters()).device
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
        
        move_to_idx = {m: (m[0] * CFG.board_size + m[1] if m != PASS else CFG.board_size**2) for m in n.unexp}
        valid_logits = np.array([lvec[idx] for idx in move_to_idx.values()])
        pr = softmax_np(valid_logits)

        for move, p in zip(n.unexp, pr):
            ch = Node(n.g.copy(), n, move); ch.P = float(p)
            n.ch[move] = ch
        n.unexp.clear()

def backup(n: Node, value: float):
    while n:
        n.N += 1; n.W += value; n.Q = n.W / n.N
        value = -value
        n = n.p

def mcts(game: GoGame, net: PolicyValueNet,
         sims: int = CFG.mcts_sims, temp: float = 1.0,
         batch_size: int = 128):
    root = Node(game.copy())
    nn_eval([root], net)

    if root.ch:
        dir_noise = np.random.dirichlet([0.03] * len(root.ch))
        for i, ch in enumerate(root.ch.values()):
            ch.P = 0.75 * ch.P + 0.25 * dir_noise[i]

    leaves: List[Node] = []
    for _ in range(sims):
        leaf = select(root)
        if leaf.g.game_over():
            winner = leaf.g.winner()
            value = 0 if winner == 0 else (1 if winner == leaf.g.current_player else -1)
            backup(leaf, value)
            continue

        leaves.append(leaf)
        if len(leaves) >= batch_size:
            nn_eval(leaves, net)
            for lf in leaves: backup(lf, lf.val)
            leaves.clear()

    if leaves:
        nn_eval(leaves, net)
        for lf in leaves: backup(lf, lf.val)

    if not root.ch: return {}
    moves, visits = zip(*((m, ch.N) for m, ch in root.ch.items()))
    v = np.array(visits, np.float32)
    if temp < 1e-3: p = np.zeros_like(v); p[np.argmax(v)] = 1.0
    else: w = v ** (1 / temp); p = w / w.sum()
    return dict(zip(moves, p))

# ==========================================================
#              SELF-PLAY WORKER  (CPU tree, GPU NN)
# ==========================================================
def _strip_prefix(state_dict, prefix="_orig_mod."):
    if not any(k.startswith(prefix) for k in state_dict): return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

def self_play_worker(args):
    rank, seed, model_path = args
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    net = PolicyValueNet(CFG.board_size).eval().to("cuda", non_blocking=True)
    state = _strip_prefix(torch.load(model_path, map_location="cpu"))
    net.load_state_dict(state)

    if rank == 0:
        moves_bar = tqdm_bar(total=CFG.max_moves, desc=f"Game {rank} moves", unit="move", position=1, leave=False)

    LOGGER.debug("Worker %d spawned (PID=%d, seed=%d)", rank, os.getpid(), seed)
    game, records = GoGame(CFG.board_size), []
    move_ctr = 0
    sims_per_move = CFG.mcts_sims_worker or CFG.mcts_sims

    while not game.game_over() and move_ctr < CFG.max_moves:
        if rank == 0: moves_bar.update()

        pi = mcts(game, net, sims_per_move, 1.0)
        if not pi: break

        vec = np.zeros(CFG.board_size**2 + 1, np.float32)
        for mv, p in pi.items():
            if mv == PASS: vec[-1] = p
            else: vec[mv[0] * CFG.board_size + mv[1]] = p

        records.append((game.board.copy(), game.current_player, vec))
        mv = random.choices(list(pi.keys()), weights=list(pi.values()))[0]
        game.make_move(*mv)
        move_ctr += 1

    if move_ctr >= CFG.max_moves:
        LOGGER.warning("Worker %d hit max_moves → forcing end", rank)
    if rank == 0: moves_bar.close()

    w = game.winner()
    LOGGER.info("Worker %d finished: moves=%d, winner=%s",
            rank, len(records), {0:"Draw",1:"Black",2:"White"}[w])
    return [(b, p, 0 if w == 0 else (1 if p == w else -1)) for b, p, _ in records]

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
    bds, pl, z = zip(*batch)
    X = torch.stack([planes(b, p) for b, p in zip(bds, pl)]).to(device_gpu)
    z_np  = np.asarray(z, dtype=np.float32)[:, None]
    return X, torch.from_numpy(z_np).to(device_gpu)

def train(net: PolicyValueNet, batch_data, it_idx):
    LOGGER.info(
        "Iter %d | start training | samples=%d, epochs=%d, lr=%.1e",
        it_idx, len(batch_data), CFG.train_epochs, CFG.lr,
    )
    opt = optim.Adam(net.parameters(), lr=CFG.lr)
    scaler = torch.amp.GradScaler(enabled=(CFG.amp and device_gpu.type == 'cuda'))
    net.train()

    # The policy vector is no longer part of the batch data, so we unpack accordingly
    bds, players, outcomes = zip(*batch_data)
    
    # We need to re-evaluate the policy for the current network state to get targets
    X = torch.stack([planes(b, p) for b, p in zip(bds, players)]).to(device_gpu, non_blocking=True)
    z_t = torch.tensor(outcomes, dtype=torch.float32).view(-1, 1).to(device_gpu, non_blocking=True)
    
    with torch.no_grad():
        pi_t, _ = net(X)
        pi_t = F.softmax(pi_t, dim=1)

    steps_per_ep = math.ceil(len(batch_data) / CFG.train_batch)
    bar = tqdm_bar(total=CFG.train_epochs * steps_per_ep, desc=f"Train Iter {it_idx}", unit="step", position=0, leave=False)

    running_loss, step_global = 0.0, 0
    for ep in range(1, CFG.train_epochs + 1):
        idx = torch.randperm(len(batch_data))
        for s in range(0, len(batch_data), CFG.train_batch):
            samp = idx[s:s+CFG.train_batch]
            inp, tgt_pi, tgt_z = X[samp], pi_t[samp], z_t[samp]

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=CFG.amp, dtype=torch.bfloat16):
                out_p, out_v = net(inp)
                loss_p = -(tgt_pi * F.log_softmax(out_p, 1)).sum(1).mean()
                loss_v = F.mse_loss(out_v, tgt_z)
                loss   = loss_p + loss_v
            
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            
            step_global += 1
            running_loss += (loss.item() - running_loss) / step_global
            vram = torch.cuda.memory_allocated() / 2**20 if device_gpu.type == 'cuda' else 0
            bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss:.4f}", vram=f"{vram:.0f} MB")
            bar.update()

    LOGGER.info("Iter %d finished - avg_loss=%.4f (samples=%d)",  it_idx, running_loss, len(batch_data))
    bar.close()
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
            pygame.draw.line(screen, (0,0,0), (i * CFG.grid, CFG.grid), (i * CFG.grid, SCREEN - CFG.grid), 2)
            pygame.draw.line(screen, (0,0,0), (CFG.grid, i * CFG.grid), (SCREEN - CFG.grid, i * CFG.grid), 2)
        for x in range(CFG.board_size):
            for y in range(CFG.board_size):
                if s := g.board[x, y]:
                    col = (10,10,10) if s == 1 else (245,245,245)
                    pygame.draw.circle(screen, col, ((y + 1) * CFG.grid, (x + 1) * CFG.grid), CFG.grid // 2 - 3)
        pygame.display.flip()

    game, clock = GoGame(CFG.board_size), pygame.time.Clock()
    while True:
        if game.game_over():
            winner_map = {0: "Draw", 1: "Black", 2: "White"}
            print(f"Game over! Winner: {winner_map[game.winner()]}")
            pygame.time.wait(4000); return
        draw(game); clock.tick(30)

        if game.current_player == 1: # Human player
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: return
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = ev.pos
                    gx, gy = round(my / CFG.grid - 1), round(mx / CFG.grid - 1)
                    if not game.make_move(gx, gy):
                        print("Illegal move!")
        else: # AI Player
            probs = mcts(game, net, CFG.mcts_sims, temp=0.01)
            if not probs: game.make_move(*PASS)
            else:
                best_move = max(probs.items(), key=lambda kv: kv[1])[0]
                game.make_move(*best_move)

# ==========================================================
#                           MAIN
# ==========================================================
def main():
    log_session_header()
    print(f"OceanGo — CUDA: {torch.cuda.is_available()} | AMP: {CFG.amp}")
    torch.backends.cudnn.benchmark = True

    raw_net = PolicyValueNet().to(device_gpu)
    net = torch.compile(raw_net, backend="inductor") if device_gpu.type == 'cuda' else raw_net
    print(f"Network is on device: {next(net.parameters()).device}")
    print("------- start self-play -------")

    ckpt_name = (f"B{CFG.board_size}_I{CFG.n_iter}_G{CFG.games_per_iter}"
                 f"_S{CFG.mcts_sims_worker}_R{CFG.replay_cap//1000}k_E{CFG.train_epochs}")
    ckpt_name = (
        f"B{CFG.board_size}"
        f"_Komi{CFG.komi}"
        f"_Sims{CFG.mcts_sims_worker}w{CFG.mcts_sims}g"
        f"_Replay{CFG.replay_cap}"
        f"_Batch{CFG.train_batch}"
        f"_Sample{CFG.train_sample_size}"
        f"_Epochs{CFG.train_epochs}"
        f"_LR{CFG.lr:.0e}"
        f"_C{CFG.mcts_c_puct}"
        f"_Alpha{CFG.dirichlet_alpha}"
        f"_Temp{CFG.start_temp}"
        f"_Decay{CFG.temp_decay_moves}"
        f"_AMP{int(CFG.amp)}"
        f"_Workers{CFG.num_workers}"
        f"_MaxMoves{CFG.max_moves}"
        f"_Timeout{CFG.worker_timeout}"
    )
    ckpt = root / f"oceango_{ckpt_name}.pth"
    print("Checkpoint path:", ckpt)
    
    do_train = True
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device_gpu)
        raw_net.load_state_dict(_strip_prefix(state))
        print("Found model checkpoint, loaded.")
        try:
            if input("Train further (T) or play GUI (G)? [T/G]: ").strip().lower() == "g":
                do_train = False
        except (EOFError, KeyboardInterrupt):
            print("\nProceeding with GUI.")
            do_train = False
    else:
        print("No checkpoint found — starting fresh.")

    if not do_train:
        run_gui(net); return

    tmp = root / "_tmp_cpu_model.pth"
    torch.save(raw_net.state_dict(), tmp)

    outer = tqdm_bar(range(1, CFG.n_iter + 1), desc="Meta-loop", unit="iter", position=0)

    for it in outer:
        print(f"\n->  Iteration {it}/{CFG.n_iter} — self-play")
        seeds = [(rank, random.randrange(2**32), tmp.as_posix()) for rank in range(CFG.games_per_iter)]
        bar = tqdm_bar(total=CFG.games_per_iter, desc="Self-play", unit="game", position=0, leave=False)
        moves_total, t0 = 0, time.time()

        with multiprocessing.Pool(CFG.num_workers) as pool:
            async_results = [pool.apply_async(self_play_worker, (s,)) for s in seeds]

            for ar in async_results:
                try:
                    rec = ar.get(timeout=CFG.worker_timeout)
                    RB.add(rec); bar.update()
                    moves_total += len(rec)
                    bar.set_postfix_str(f"buffer={len(RB.data):,} moves/s={moves_total/(time.time()-t0):.1f}")
                except multiprocessing.TimeoutError:
                    LOGGER.error("Worker timed out and was skipped.")
        bar.close()

        LOGGER.info(
            "Iter %d summary: games=%d, moves=%d, buffer=%d, wall=%.1fs",
            it, CFG.games_per_iter, moves_total, len(RB.data), time.time() - t0,
        )

        train(net, RB.sample(CFG.train_sample_size), it)
        torch.save(raw_net.state_dict(), ckpt)
        torch.save(raw_net.state_dict(), tmp)

        if psutil:
            mem_gb = psutil.virtual_memory().used / 2**30
            outer.set_postfix_str(f"buffer={len(RB.data):,} RAM={mem_gb:.1f}GB")
    
    run_gui(net)

# ───────────────────────── entry ─────────────────────────
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
