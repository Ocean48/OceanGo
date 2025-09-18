#!/usr/bin/env python3
# OceanGo - 9×9 (default) Go with GPU training & GPU self-play
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
import uuid
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
    n_iter: int = 100           # Outer training loops per run.
    games_per_iter: int = 48   # Parallel self-play games each loop.

    # ── 4. MCTS search depth & Hyperparameters ────────────────
    mcts_sims: int = 5_000                # Used in GUI for a very strong opponent.
    mcts_sims_worker: int = 5_000         # Sims per move in self-play.
    mcts_batch_size: int = 256            # Batch size for NN evals inside MCTS.
    mcts_c_puct: float = 1.25             # Exploration factor in PUCT.
    dirichlet_alpha: float = 0.1          # Noise for root node exploration.
    start_temp: float = 1.0               # Initial temperature for move selection.
    temp_decay_moves: int = 40            # Moves before temp -> 0 for exploitation.
    min_moves_for_pass: int = 0           # Will be set dynamically after Config init.
    cleanup_moves: int = 18               # Extra moves played to resolve dead stones.
    cleanup_policy_temp: float = 0.5      # Temperature for cleanup move selection.

    # ── 5. Neural Network Architecture ────────────────────────
    nn_blocks: int = 12                   # Number of residual blocks.
    nn_channels: int = 128                # Number of channels in conv layers.

    # ── 6. Replay buffer (lives in **system RAM**) ────────────
    replay_cap: int = 60_000   # Maximum positions kept.
    train_sample_size: int = 8192 # Positions sampled for training.

    # ── 7. SGD / GPU training parameters ─────────────────────
    train_batch: int = 1024
    train_epochs: int = 5
    lr: float = 1e-4
    ## CHANGELOG: Added weight decay to regularize the model and prevent pathological policies.
    weight_decay: float = 1e-4    # L2 regularization.

    # ── 8. Hardware toggles ───────────────────────────────────
    amp: bool = True           # Automatic Mixed Precision on GPU.
    num_workers: int = 20      # GPU processes that run self-play.

    # ── 9. Robustness knobs (watch-dog & move cap) ────────────
    max_moves: Optional[int] = None              # Per-game hard cap; None → unlimited.
    worker_timeout: Optional[int] = None         # 6 hours; None → disable.


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
CFG.min_moves_for_pass = CFG.board_size * CFG.board_size // 2

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
    LOGGER.info("\n\n" + "-" * 80)
    for k, v in header.items():
        LOGGER.info("%-11s : %s", k, v)
    LOGGER.info("-" * 80)


# ==========================================================
#                   POLICY-VALUE NETWORK
# ==========================================================
def _strip_prefix(state_dict, prefix="_orig_mod."):
    if not any(k.startswith(prefix) for k in state_dict): return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

# ───────────────────── NN building blocks ────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch=CFG.nn_channels):
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
    def __init__(self, bs: int = CFG.board_size, ch: int = CFG.nn_channels, blocks: int = CFG.nn_blocks):
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
def softmax_np(x, temp=1.0):
    if temp == 0:
        y = np.zeros_like(x)
        y[np.argmax(x)] = 1.0
        return y
    x = x / temp
    e = np.exp(x - np.max(x))
    return e / e.sum()

def _territory_score(game: "GoGame") -> float:
    board = game.board.copy()
    bs = board.shape[0]
    seen = np.zeros_like(board, dtype=bool)
    
    black_territory = 0
    white_territory = 0

    for x in range(bs):
        for y in range(bs):
            if board[x, y] == 0 and not seen[x, y]:
                q = [(x, y)]
                empties = []
                border_colors = set()
                visited_in_group = set()
                
                while q:
                    cx, cy = q.pop()
                    if (cx, cy) in visited_in_group: continue
                    visited_in_group.add((cx, cy))
                    
                    empties.append((cx, cy))
                    for dx, dy in ((1,0),(-1,0),(0,1),(-1,0)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < bs and 0 <= ny < bs:
                            col = board[nx, ny]
                            if col == 0:
                                q.append((nx, ny))
                            else:
                                border_colors.add(col)

                for ex, ey in empties:
                    seen[ex, ey] = True
                
                if len(border_colors) == 1:
                    if 1 in border_colors:
                        black_territory += len(empties)
                    else:
                        white_territory += len(empties)
                        
    black_score = black_territory + game.captured_by_black
    white_score = white_territory + game.captured_by_white + game.komi
    
    return black_score - white_score

def _chinese_area(board: np.ndarray, komi: float = CFG.komi) -> float:
    """Standard scoring for GUI/final result. Counts stones + territory."""
    bs = board.shape[0]
    seen = np.zeros_like(board, dtype=bool)
    area_black = np.sum(board == 1)
    area_white = np.sum(board == 2)

    for x in range(bs):
        for y in range(bs):
            if board[x, y] == 0 and not seen[x, y]:
                q = [(x, y)]
                empties = []
                border_colors = set()
                visited_in_group = set()
                
                while q:
                    cx, cy = q.pop()
                    if (cx, cy) in visited_in_group: continue
                    visited_in_group.add((cx, cy))
                    empties.append((cx, cy))

                    for dx, dy in ((1,0),(-1,0),(0,1),(-1,0)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < bs and 0 <= ny < bs:
                            col = board[nx, ny]
                            if col == 0:
                                q.append((nx, ny))
                            else:
                                border_colors.add(col)
                
                for ex, ey in empties:
                    seen[ex, ey] = True

                if len(border_colors) == 1:
                    if 1 in border_colors:
                        area_black += len(empties)
                    else:
                        area_white += len(empties)

    return area_black - (area_white + komi)

PASS = (-1, -1)

class GoGame:
    def __init__(self, bs: int = CFG.board_size):
        self.bs = bs
        self.board = np.zeros((bs, bs), np.int8)
        self.current_player = 1
        self.history: List[np.ndarray] = []
        self.ko_point: Optional[Tuple[int, int]] = None
        self.komi = CFG.komi
        self.captured_by_black = 0
        self.captured_by_white = 0


    @property
    def opponent_player(self) -> int:
        return 3 - self.current_player

    def copy(self):
        g = GoGame(self.bs)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.history = [b.copy() for b in self.history]
        g.ko_point = self.ko_point
        g.captured_by_black = self.captured_by_black
        g.captured_by_white = self.captured_by_white
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
        self._remove_captured(self.opponent_player, tmp, count_captures=False)
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

    def _remove_captured(self, color, board=None, count_captures=True):
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
        
        num_captured = len(captured_stones)
        if count_captures and num_captured > 0:
            if self.current_player == 1:
                self.captured_by_black += num_captured
            else:
                self.captured_by_white += num_captured
        return num_captured

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
            if num_captured > 0:
                if self.current_player == 1: self.captured_by_black -= num_captured
                else: self.captured_by_white -= num_captured
            return False

        if len(self.history) >= 2 and np.array_equal(self.board, self.history[-2]):
            self.board = prev_board
            if num_captured > 0:
                if self.current_player == 1: self.captured_by_black -= num_captured
                else: self.captured_by_white -= num_captured
            return False

        self.ko_point = None
        if num_captured == 1:
            diff = prev_board - self.board
            if np.sum(diff != 0) == 2:
                removed_mask = (prev_board != 0) & (self.board == 0)
                if np.sum(removed_mask) == 1:
                    ko_x, ko_y = np.argwhere(removed_mask)[0]
                    temp_board = self.board.copy()
                    temp_board[ko_x, ko_y] = self.opponent_player
                    if not self._has_liberty((ko_x, ko_y), temp_board):
                        self.ko_point = (ko_x, ko_y)

        self.history.append(prev_board)
        self.switch()
        return True

    def _last_two_passed(self):
        return (len(self.history) >= 2 and
                np.array_equal(self.history[-1], self.history[-2]))

    def get_legal_moves(self, include_pass=True):
        moves = []
        for x in range(self.bs):
            for y in range(self.bs):
                if self.board[x, y] == 0 and (x, y) != self.ko_point:
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
                                    captures_something = True
                    
                    if has_liberty or captures_something:
                        moves.append((x, y))
                    elif not self.is_suicide(x, y):
                        moves.append((x, y))

        if include_pass and (len(self.history) >= CFG.min_moves_for_pass or not moves):
            moves.append(PASS)

        return moves

    def game_over(self): return self._last_two_passed()

    def get_score(self, use_territory_scoring=False):
        """Returns the final score. Uses territory for self-play, Chinese for GUI."""
        if use_territory_scoring:
            return _territory_score(self)
        return _chinese_area(self.board, self.komi)

    def state_tensor(self):
        p, o = self.current_player, self.opponent_player
        return torch.from_numpy(np.stack([(self.board == p).astype(np.float32),
                                        (self.board == o).astype(np.float32)], 0))

@torch.no_grad()
def _policy_guided_playout(game: GoGame, net: PolicyValueNet):
    """
    After players pass, use the network's policy to play a set number of
    moves to resolve ambiguous life/death situations before final scoring.
    """
    temp_game = game.copy()
    
    for _ in range(CFG.cleanup_moves):
        legal_moves = temp_game.get_legal_moves(include_pass=False)
        if not legal_moves:
            temp_game.switch()
            legal_moves = temp_game.get_legal_moves(include_pass=False)
            if not legal_moves:
                break

        state_tensor = temp_game.state_tensor().unsqueeze(0).to(device_gpu)
        with torch.autocast(device_type="cuda", enabled=CFG.amp, dtype=torch.bfloat16):
            policy_logits, _ = net(state_tensor)
        
        policy_logits = policy_logits.squeeze(0).float().cpu().numpy()
        
        move_to_idx = {m: (m[0]*CFG.board_size + m[1]) for m in legal_moves}
        valid_logits = np.array([policy_logits[idx] for idx in move_to_idx.values()])
        
        probs = softmax_np(valid_logits, temp=CFG.cleanup_policy_temp)
        
        move_idx = np.random.choice(len(legal_moves), p=probs)
        move = legal_moves[move_idx]
        temp_game.make_move(*move)
        
    return temp_game

def render_board_to_string(board: np.ndarray) -> str:
    """Renders the board to a multi-line string for console/log output."""
    stone_map = {0: '.', 1: 'X', 2: 'O'} # . for empty, X for black, O for white
    bs = board.shape[0]
    header = "   " + " ".join([chr(ord('A') + i) for i in range(bs)])
    lines = [header]
    for r in range(bs):
        row_str = f"{r+1:<2d} " + " ".join([stone_map[board[r, c]] for c in range(bs)])
        lines.append(row_str)
    return "\n".join(lines)

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
                key=lambda c: c.Q + CFG.mcts_c_puct * c.P * math.sqrt(node.N) / (1 + c.N))

@torch.no_grad()
def nn_eval(batch: List[Node], net: PolicyValueNet):
    """Performs NN evaluation in the local process (e.g., for GUI or GPU workers)."""
    if not batch: return
    dev = next(net.parameters()).device
    with torch.autocast(device_type=dev.type, enabled=(CFG.amp and dev.type=="cuda"), dtype=torch.bfloat16):
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

def mcts(game: GoGame, net: PolicyValueNet, sims: int, temp: float = 1.0, add_noise: bool = False):
    
    root = Node(game.copy())
    nn_eval([root], net)

    if add_noise and root.ch:
        dir_noise = np.random.dirichlet([CFG.dirichlet_alpha] * len(root.ch))
        for i, ch in enumerate(root.ch.values()):
            ch.P = 0.75 * ch.P + 0.25 * dir_noise[i]

    leaves: List[Node] = []
    for _ in range(sims):
        leaf = select(root)
        if leaf.g.game_over():
            scoring_game = _policy_guided_playout(leaf.g, net)
            score = scoring_game.get_score(use_territory_scoring=True)
            winner = 0 if abs(score) < 1e-3 else (1 if score > 0 else 2)
            value = 0 if winner == 0 else (1 if winner == leaf.g.current_player else -1)
            backup(leaf, value)
            continue

        leaves.append(leaf)
        if len(leaves) >= CFG.mcts_batch_size:
            nn_eval(leaves, net)
            for lf in leaves: backup(lf, lf.val)
            leaves.clear()

    if leaves:
        nn_eval(leaves, net)
        for lf in leaves: backup(lf, lf.val)

    if not root.ch: return {}
    moves, visits = zip(*((m, ch.N) for m, ch in root.ch.items()))
    v = np.array(visits, np.float32)
    if temp < 1e-3:
        p = np.zeros_like(v)
        p[np.argmax(v)] = 1.0
    else:
        log_visits = np.log(np.maximum(v, 1e-9))
        log_weighted = log_visits / temp
        log_weighted -= np.max(log_weighted)
        w = np.exp(log_weighted)
        p = w / w.sum()
        if np.isnan(p).any() or p.sum() < 1e-6:
            p = np.ones(len(v), dtype=np.float32) / len(v)
            
    return dict(zip(moves, p))

# ==========================================================
#                 SELF-PLAY GPU WORKER
# ==========================================================
def self_play_worker(args):
    """A self-contained worker that loads the model onto the GPU and plays games."""
    rank, seed, model_path_str = args
    model_path = Path(model_path_str)
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    net = PolicyValueNet().eval().to(device_gpu)
    if model_path.exists():
        state = _strip_prefix(torch.load(model_path, map_location=device_gpu))
        net.load_state_dict(state)

    if rank == 0:
        moves_bar = tqdm_bar(total=CFG.max_moves, desc=f"Game {rank} moves", unit="move", position=1, leave=False)

    LOGGER.debug("GPU Worker %d spawned (PID=%d, seed=%d)", rank, os.getpid(), seed)
    game, records = GoGame(CFG.board_size), []
    move_ctr = 0
    sims_per_move = CFG.mcts_sims_worker or CFG.mcts_sims

    while not game.game_over() and move_ctr < CFG.max_moves:
        if rank == 0: moves_bar.update()

        temp = CFG.start_temp if move_ctr < CFG.temp_decay_moves else 0.01
        pi = mcts(game, net, sims_per_move, temp, add_noise=True)
        if not pi: break

        records.append((game.board.copy(), game.current_player, pi))
        mv = random.choices(list(pi.keys()), weights=list(pi.values()))[0]
        game.make_move(*mv)
        move_ctr += 1

    hit_max_moves = move_ctr >= CFG.max_moves
    if hit_max_moves:
        LOGGER.warning("Worker %d hit max_moves → forcing end as a DRAW", rank)
        w = 0
        final_game_state = game
    else:
        # CRITICAL: Resolve life and death dynamically before final scoring.
        final_game_state = _policy_guided_playout(game, net)
        final_score = final_game_state.get_score(use_territory_scoring=True)
        w = 0 if abs(final_score) < 1e-3 else (1 if final_score > 0 else 2)
        
    if rank == 0: moves_bar.close()

    player_map = {0:"Draw", 1:"Black", 2:"White"}
    LOGGER.info("Worker %d finished: moves=%d, winner=%s",
                rank, len(records), player_map[w])
    
    if rank < 3:
        LOGGER.info(f"Final board state for Worker {rank}:\n{render_board_to_string(final_game_state.board)}")

    training_data = []
    for board_state, player, policy_dict in records:
        outcome = 0 if w == 0 else (1 if player == w else -1)
        
        policy_vec = np.zeros(CFG.board_size**2 + 1, np.float32)
        for mv, p in policy_dict.items():
            if mv == PASS: policy_vec[-1] = p
            else: policy_vec[mv[0] * CFG.board_size + mv[1]] = p
        
        training_data.append((board_state, player, policy_vec, outcome))
        
    return training_data, w


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
    return np.stack([(board == player).astype(np.float32),
                    (board == 3 - player).astype(np.float32)], 0)

def train(net: PolicyValueNet, batch_data, it_idx):
    LOGGER.info(
        "Iter %d | start training | samples=%d, epochs=%d, lr=%.1e",
        it_idx, len(batch_data), CFG.train_epochs, CFG.lr,
    )
    ## CHANGELOG: Added weight_decay to the optimizer for regularization.
    opt = optim.Adam(net.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(CFG.amp and device_gpu.type == 'cuda'))
    net.train()

    boards, players, target_policies, target_values = zip(*batch_data)
    
    X_np = np.stack([planes(b, p) for b, p in zip(boards, players)])
    X = torch.from_numpy(X_np).to(device_gpu, non_blocking=True)
    pi_t = torch.tensor(np.array(target_policies), dtype=torch.float32).to(device_gpu, non_blocking=True)
    z_t = torch.tensor(target_values, dtype=torch.float32).view(-1, 1).to(device_gpu, non_blocking=True)

    steps_per_ep = math.ceil(len(batch_data) / CFG.train_batch)
    bar = tqdm_bar(total=CFG.train_epochs * steps_per_ep, desc=f"Train Iter {it_idx}", unit="step", position=0, leave=False)

    avg_loss, avg_p_loss, avg_v_loss = 0.0, 0.0, 0.0
    step_global = 0
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
            avg_loss += (loss.item() - avg_loss) / step_global
            avg_p_loss += (loss_p.item() - avg_p_loss) / step_global
            avg_v_loss += (loss_v.item() - avg_v_loss) / step_global

            vram = torch.cuda.memory_allocated() / 2**20 if device_gpu.type == 'cuda' else 0
            bar.set_postfix(
                loss=f"{avg_loss:.4f}", p_loss=f"{avg_p_loss:.4f}",
                v_loss=f"{avg_v_loss:.4f}", vram=f"{vram:.0f}MB"
            )
            bar.update()

    LOGGER.info("Iter %d finished - avg_loss=%.4f (p=%.4f, v=%.4f)",
                it_idx, avg_loss, avg_p_loss, avg_v_loss)
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
    font = pygame.font.Font(None, 28)

    def draw(g: GoGame, win_rate: float):
        screen.fill((240, 207, 155)) # Board color
        
        for i in range(1, CFG.board_size + 1):
            pygame.draw.line(screen, (0,0,0), (i*CFG.grid, CFG.grid), (i*CFG.grid, SCREEN-CFG.grid), 2)
            pygame.draw.line(screen, (0,0,0), (CFG.grid, i*CFG.grid), (SCREEN-CFG.grid, i*CFG.grid), 2)
        
        for x in range(CFG.board_size):
            for y in range(CFG.board_size):
                if s := g.board[x, y]:
                    col = (10,10,10) if s == 1 else (245,245,245)
                    pygame.draw.circle(screen, col, ((y+1)*CFG.grid, (x+1)*CFG.grid), CFG.grid//2-3)

        player_str = "Black" if g.current_player == 1 else "White"
        player_col = (10,10,10) if g.current_player == 1 else (245,245,245)
        text_surf = font.render(f"Turn: {player_str}", True, player_col, (200, 167, 115))
        screen.blit(text_surf, (10, 10))

        rate_str = f"Win Rate: {win_rate:.1f}%"
        text_surf = font.render(rate_str, True, player_col, (200, 167, 115))
        screen.blit(text_surf, (200, 10))

        score = _chinese_area(g.board, CFG.komi)
        score_str = f"Score: B+{-score:.1f}" if score < 0 else f"Score: B+{score:.1f}"
        text_surf = font.render(score_str, True, (10,10,10), (200, 167, 115))
        screen.blit(text_surf, (400, 10))

        pygame.display.flip()

    game, clock = GoGame(CFG.board_size), pygame.time.Clock()
    current_win_rate = 50.0

    while True:
        if game.game_over():
            winner_map = {0: "Draw", 1: "Black", 2: "White"}
            final_score = game.get_score(use_territory_scoring=False)
            winner_val = 0 if abs(final_score) < 1e-3 else (1 if final_score > 0 else 2)
            print(f"Game over! Winner: {winner_map[winner_val]}, Score: B+{final_score:.1f}")
            pygame.time.wait(4000); return
        
        root_node = Node(game.copy())
        nn_eval([root_node], net)
        current_win_rate = (root_node.val + 1) / 2 * 100

        draw(game, current_win_rate); clock.tick(30)

        if game.current_player == 1: # Human player (Black)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: return
                if ev.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = ev.pos
                    gx, gy = round(my / CFG.grid - 1), round(mx / CFG.grid - 1)
                    if not game.make_move(gx, gy):
                        print("Illegal move!")
        else: # AI Player (White)
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

    ckpt_name = (f"B{CFG.board_size}_S{CFG.mcts_sims_worker//1000}k_R{CFG.replay_cap//1000}k_E{CFG.train_epochs}")
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

    print("------- start self-play -------")
    # Save initial model for workers to load
    torch.save(raw_net.state_dict(), ckpt)
    
    outer = tqdm_bar(range(1, CFG.n_iter + 1), desc="Meta-loop", unit="iter", position=0)

    for it in outer:
        print(f"\n->  Iteration {it}/{CFG.n_iter} — self-play")
        # Each worker gets the path to the current model checkpoint.
        worker_args = [(rank, random.randrange(2**32), ckpt.as_posix()) for rank in range(CFG.games_per_iter)]
        
        bar = tqdm_bar(total=CFG.games_per_iter, desc="Self-play", unit="game", position=0, leave=False)
        t0 = time.time()
        black_wins, white_wins, draws, total_game_moves = 0, 0, 0, 0

        with multiprocessing.Pool(CFG.num_workers) as pool:
            async_results = [pool.apply_async(self_play_worker, (args,)) for args in worker_args]

            for ar in async_results:
                try:
                    records, winner = ar.get(timeout=CFG.worker_timeout)
                    if winner == 1: black_wins += 1
                    elif winner == 2: white_wins += 1
                    else: draws += 1
                    
                    RB.add(records); bar.update()
                    total_game_moves += len(records)
                    bar.set_postfix_str(f"buffer={len(RB.data):,} moves/s={total_game_moves/(time.time()-t0):.1f}")
                except multiprocessing.TimeoutError:
                    LOGGER.error("Worker timed out and was skipped.")
        bar.close()
        
        if CFG.games_per_iter > 0:
            avg_moves = total_game_moves / CFG.games_per_iter
            b_win_rate = black_wins / CFG.games_per_iter * 100
            w_win_rate = white_wins / CFG.games_per_iter * 100
            LOGGER.info(
                f"Iter {it} self-play stats: "
                f"B wins: {b_win_rate:.1f}% ({black_wins}) | "
                f"W wins: {w_win_rate:.1f}% ({white_wins}) | "
                f"Draws: {draws} | Avg game len: {avg_moves:.1f}"
            )

        if len(RB.data) > CFG.train_batch:
            train(net, RB.sample(CFG.train_sample_size), it)
            # Save the newly trained model for the next iteration of workers
            torch.save(raw_net.state_dict(), ckpt)
        else:
            LOGGER.warning(f"Skipping training for iter {it}, not enough data in buffer ({len(RB.data)} samples).")

        if psutil:
            mem_gb = psutil.virtual_memory().used / 2**30
            outer.set_postfix_str(f"buffer={len(RB.data):,} RAM={mem_gb:.1f}GB")
    
    run_gui(net)

# ───────────────────────── entry ─────────────────────────
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

