import logging
import math
import random
import copy
import time
import sys
import os
import argparse
import concurrent.futures
import multiprocessing as mp
import threading
import queue
import platform
from collections import deque
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------------------
# Global config
# ----------------------------------------
# Detect operating system dynamically: "windows", "darwin" (macOS), or "linux"
OS_NAME = platform.system().lower()

# The dimensions of the Go board (e.g., 9 for 9x9, 13 for 13x13, 19 for standard 19x19).
BOARD_SIZE = 9
# Pixel size of each grid square in the Pygame interactive UI.
GRID_SIZE = 40
# Total pixel dimension of the Pygame window.
SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE

# Cross-platform hardware acceleration:
# 1. CUDA (NVIDIA GPU on Windows / Linux)
# 2. MPS (Metal Performance Shaders on Apple Silicon macOS)
# 3. CPU fallback
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision('high')
    device_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = f"Apple Silicon Metal / MPS ({platform.machine()})"
else:
    device = torch.device("cpu")
    device_name = f"CPU ({platform.machine()})"

# --- Hyperparameters ---
# Self-Play / Training
# Number of outer loop iterations (Self-play -> Train -> Save). Search convergence improves with higher iterations.
N_ITER = 50
# Number of parallel self-play games to play per iteration. Adjust based on your available CPU cores.
N_GAMES_PER_ITER = 500
# Number of MCTS simulations per move. Higher = stronger AI strategy, but slower execution (e.g. 500-1600+).
N_MCTS_SIMS = 800
# Number of board states to sample randomly from the replay buffer for each neural network training phase.
TRAIN_SAMPLE_SIZE = 250000
# Maximum number of historical moves to keep in memory to prevent model from overfitting to recent games.
MAX_REPLAY_BUFFER_SIZE = 500000

# MCTS Setup
# Intra-tree virtual-loss batch size per game worker (16 keeps PUCT tree exploration sharp while still batching).
MCTS_BATCH_SIZE = 16
# Maximum number of inference requests coalesced simultaneously by the central GPU server across workers.
GPU_SERVER_BATCH_SIZE = 512
# Exploration constant for PUCT formulation. Higher = explores more unvisited tree nodes prior to exploiting.
C_PUCT = 1.0
# Shape of Dirichlet noise. Scales inversely with board area (0.03 for 19x19, 0.06 for 13x13, 0.15 for 9x9).
DIRICHLET_ALPHA = 0.03 if BOARD_SIZE >= 19 else (0.06 if BOARD_SIZE >= 13 else 0.15)
# Ratio of the Dirichlet noise added to the root node (0.25 means 25% noise / 75% real model policy).
DIRICHLET_EPSILON = 0.25


def get_dirichlet_alpha(board_size: int) -> float:
    """Scales Dirichlet concentration parameter inversely with board intersections (AlphaZero scaling)."""
    if board_size >= 19:
        return 0.03
    if board_size >= 13:
        return 0.06
    return 0.15


def get_temp_threshold(board_size: int) -> int:
    """
    Returns the move number after which self-play temperature drops from 1.0 (exploratory)
    to 1e-3 (greedy). Scales dynamically so exploration covers the opening (~15% of game length)
    on 9x9 (10 moves), 13x13 (16 moves), and 19x19 (30 moves, matching DeepMind AlphaGo Zero).
    """
    if board_size >= 19:
        return 30
    if board_size >= 13:
        return 16
    return 10


def get_scheduled_lr(iteration: int, base_lr: float = 0.01) -> float:
    """AlphaGo Zero style step-decay learning rate schedule across training iterations."""
    if iteration < 20:
        return base_lr
    if iteration < 40:
        return base_lr * 0.1
    return base_lr * 0.01

# Neural Network Setup
# Number of convolutional filters per ResNet block (AlphaZero used 256; use 64-128 for consumer GPUs).
RESNET_CHANNELS = 256
# Number of Residual blocks in the network (e.g., 5 for quick training, 19 to 39 for immense strategic depth).
RESNET_BLOCKS = 19
# Minibatch size fed to the GPU during neural network backpropagation.
TRAIN_BATCH_SIZE = 128
# Number of passes over the training sample batch per iteration.
TRAIN_EPOCHS = 5
# Gradient descent learning rate (SGD step size).
LEARNING_RATE = 0.01

# Multi-core allocation (safe across low or high CPU core counts)
sys_cores = os.cpu_count() or 1
is_colab = ("COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ or os.path.exists("/content"))
if is_colab:
    # On headless Google Colab VMs, use all vCPUs (minimum 4 workers to overlap CPU MCTS with GPU inference)
    suggested_cores = max(4, sys_cores)
else:
    # On local macOS (M-series) or Windows (RTX GPU), reserve 2 cores so the desktop OS stays responsive
    suggested_cores = max(1, sys_cores - 2) if sys_cores > 2 else sys_cores
PARALLEL_THREADS = min(14, suggested_cores)

def init_environment():
    """Initializes logging and displays system runtime details (called once from main)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_start = time.strftime("%Y-%m-%d %H:%M:%S")
    log_filename_time = time.strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(script_dir, f"logs_{log_filename_time}.log")

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    print("========================================")
    print("OceanGo Initializing...")
    print(f"OS Platform: {platform.system()} ({platform.machine()})")
    print(f"Processing Computing Device: {device.type.upper()}")
    print(f"Device Details: {device_name}")
    print(f"System CPU Cores Available: {sys_cores}")
    print(f"Algorithm Using: {PARALLEL_THREADS} Cores")
    print(f"Suggested Optimal Cores: {suggested_cores}")
    print("========================================\n")

    logging.info(f"\n\n{'='*40}")
    logging.info(f"Starting OceanGo log at {log_start}")
    logging.info(f"OS Platform: {platform.system()} ({platform.machine()})")
    logging.info(f"Processing Computing Device: {device.type.upper()}")
    logging.info(f"Device Details: {device_name}")
    logging.info(f"System CPU Cores Available: {sys_cores}")
    logging.info(f"Algorithm Using: {PARALLEL_THREADS} Cores")
    logging.info(f"Suggested Optimal Cores: {suggested_cores}")
    logging.info(f"{'='*40}")

# ====================================================
#             ResNet Blocks
# ====================================================
class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        x = F.relu(x)
        return x

# ====================================================
#             Policy-Value Network (ResNet)
# ====================================================
class PolicyValueNet(nn.Module):
    def __init__(self, board_size=BOARD_SIZE, num_channels=RESNET_CHANNELS, num_res_blocks=RESNET_BLOCKS):
        super().__init__()
        self.board_size = board_size

        # Initial Convolutional Block
        self.conv_in = nn.Conv2d(17, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)

        # Residual Blocks
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size + 1)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch_size, 17, board_size, board_size)
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        for block in self.res_blocks:
            x = block(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # in [-1, 1]

        return p, v

# ====================================================
#             GoGame class (CPU logic)
# ====================================================
class GoGame:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        # 0=empty, 1=player1, 2=player2
        self.board = np.zeros((board_size, board_size), dtype=np.int32)
        self.current_player = 1
        self.history = set()
        self.history.add(self.board.tobytes())
        self.passes_in_a_row = 0
        self.history_states = [np.zeros((board_size, board_size), dtype=np.int32) for _ in range(8)]
        self.history_states[0] = self.board.copy()
        self._legal_moves_cache = None
        self._is_game_over_cache = None

    def copy(self):
        g = GoGame(self.board_size)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.history = self.history.copy()
        g.passes_in_a_row = self.passes_in_a_row
        g.history_states = [b.copy() for b in self.history_states]
        g._legal_moves_cache = None
        g._is_game_over_cache = None
        return g

    def get_state_np(self):
        return (self.board.copy(), self.current_player)

    def get_nn_input(self):
        """ Returns 17 x board_size x board_size numpy array """
        planes = np.zeros((17, self.board_size, self.board_size), dtype=np.float32)
        for i in range(8):
            board_i = self.history_states[i]
            planes[i] = (board_i == self.current_player).astype(np.float32)
            planes[i+8] = (board_i == (3 - self.current_player)).astype(np.float32)
        planes[16] = 1.0 if self.current_player == 1 else 0.0
        return planes

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def is_valid_move(self, x, y):
        # Pass move
        if x == -1 and y == -1:
            return True
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False
        if self.board[x, y] != 0:
            return False
        return self._try_move(x, y)

    def _try_move(self, x, y):
        opponent = 3 - self.current_player
        b_size = self.board_size

        # Temporarily place stone on board to test captures and liberties without copying
        self.board[x, y] = self.current_player

        captured_neighbors = []
        has_direct_empty = False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < b_size and 0 <= ny < b_size:
                cell = self.board[nx, ny]
                if cell == 0:
                    has_direct_empty = True
                elif cell == opponent:
                    if not self._has_liberty((nx, ny), self.board):
                        captured_neighbors.append((nx, ny))

        # Fast path (>95% of moves): no opponent groups are captured
        if not captured_neighbors:
            has_lib = has_direct_empty or self._has_liberty((x, y), self.board)
            if not has_lib:
                self.board[x, y] = 0
                return False
            board_hash = self.board.tobytes()
            self.board[x, y] = 0
            return board_hash not in self.history

        # Capture path: copy board, remove captured opponent groups, and check positional superko
        # (Suicide is impossible when adjacent opponent stones are captured, as their removal creates liberties)
        temp_board = self.board.copy()
        self.board[x, y] = 0
        for c_pos in captured_neighbors:
            if temp_board[c_pos[0], c_pos[1]] == opponent:
                self._remove_group(c_pos, temp_board)

        return temp_board.tobytes() not in self.history

    def make_move(self, x, y, check_valid=True):
        if check_valid and not self.is_valid_move(x, y):
            return False
            
        if x == -1 and y == -1:
            self.passes_in_a_row += 1
        else:
            self.passes_in_a_row = 0
            self.board[x, y] = self.current_player
            
            # Remove captures
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board[nx, ny] == 3 - self.current_player:
                        if not self._has_liberty((nx, ny), self.board):
                            self._remove_group((nx, ny), self.board)
            
        self.switch_player()
        self.history.add(self.board.tobytes())
        self.history_states.insert(0, self.board.copy())
        self.history_states.pop()
        self._legal_moves_cache = None
        self._is_game_over_cache = None
        return True

    def _has_liberty(self, start, board_state):
        b_size = self.board_size
        sx, sy = start
        # Immediate O(1) neighbor check before allocating BFS structures
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = sx + dx, sy + dy
            if 0 <= nx < b_size and 0 <= ny < b_size and board_state[nx, ny] == 0:
                return True

        color = board_state[sx, sy]
        visited = {(sx, sy)}
        q = deque([start])
        while q:
            cx, cy = q.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < b_size and 0 <= ny < b_size:
                    cell = board_state[nx, ny]
                    if cell == 0:
                        return True
                    if cell == color and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return False

    def _remove_group(self, start, board_state):
        b_size = self.board_size
        sx, sy = start
        color = board_state[sx, sy]
        visited = {(sx, sy)}
        q = deque([start])
        while q:
            cx, cy = q.pop()
            board_state[cx, cy] = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < b_size and 0 <= ny < b_size:
                    if board_state[nx, ny] == color and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))

    def get_legal_moves(self):
        if self._legal_moves_cache is not None:
            return self._legal_moves_cache
        moves = [(-1, -1)]  # Pass is always a legal move
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == 0 and self._try_move(x, y):
                    moves.append((x, y))
        self._legal_moves_cache = moves
        return moves

    def is_game_over(self):
        return self.passes_in_a_row >= 2

    def get_group(self, r, c):
        """Returns (color, set_of_coordinates) for the connected stone group at (r, c)."""
        if self.board[r, c] == 0:
            return 0, set()
        color = int(self.board[r, c])
        q = deque([(r, c)])
        group = {(r, c)}
        while q:
            cr, cc = q.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if self.board[nr, nc] == color and (nr, nc) not in group:
                        group.add((nr, nc))
                        q.append((nr, nc))
        return color, group

    def auto_detect_dead_stones(self):
        """Automatically identifies small or completely trapped dead stone groups without 2 eyes."""
        b_size = self.board_size
        dead_stones = set()
        visited = set()
        groups = []

        for r in range(b_size):
            for c in range(b_size):
                if self.board[r, c] != 0 and (r, c) not in visited:
                    color, grp = self.get_group(r, c)
                    visited.update(grp)
                    libs = set()
                    for gr, gc in grp:
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = gr + dr, gc + dc
                            if 0 <= nr < b_size and 0 <= nc < b_size and self.board[nr, nc] == 0:
                                libs.add((nr, nc))
                    groups.append({"color": color, "stones": grp, "liberties": libs})

        for g in groups:
            color = g["color"]
            opp = 3 - color
            libs = g["liberties"]
            if len(libs) == 0:
                dead_stones.update(g["stones"])
            elif len(libs) <= 2 and len(g["stones"]) <= 4:
                # Check if all liberties are directly bounded by opponent stones
                all_surrounded = True
                for lr, lc in libs:
                    touches_opp = any(
                        0 <= lr + dr < b_size and 0 <= lc + dc < b_size and self.board[lr + dr, lc + dc] == opp
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    )
                    if not touches_opp:
                        all_surrounded = False
                        break
                if all_surrounded:
                    dead_stones.update(g["stones"])

        return dead_stones

    def get_score_breakdown(self, dead_stones=None):
        # Chinese area scoring with 7.5 Komi
        if dead_stones is None:
            dead_set = set()
        else:
            dead_set = set(dead_stones)

        scoring_board = self.board.copy()
        for r, c in dead_set:
            scoring_board[r, c] = 0

        b_alive_stones = {(r, c) for r in range(self.board_size) for c in range(self.board_size)
                          if self.board[r, c] == 1 and (r, c) not in dead_set}
        w_alive_stones = {(r, c) for r in range(self.board_size) for c in range(self.board_size)
                          if self.board[r, c] == 2 and (r, c) not in dead_set}

        b_terr_points = set()
        w_terr_points = set()
        visited = set()

        for r in range(self.board_size):
            for c in range(self.board_size):
                if scoring_board[r, c] == 0 and (r, c) not in visited:
                    q = deque([(r, c)])
                    group = set([(r, c)])
                    touches_black = False
                    touches_white = False

                    while q:
                        cx, cy = q.pop()
                        visited.add((cx, cy))
                        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                                val = scoring_board[nx, ny]
                                if val == 1:
                                    touches_black = True
                                elif val == 2:
                                    touches_white = True
                                elif val == 0 and (nx, ny) not in group:
                                    group.add((nx, ny))
                                    q.append((nx, ny))

                    visited.update(group)
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
            "margin": abs(b_score - w_score)
        }

    def get_winner(self):
        return self.get_score_breakdown()["winner"]

# ====================================================
#  Helper: Convert inputs to PyTorch Tensor
# ====================================================
def boards_to_tensor(batch_of_inputs):
    # batch_of_inputs is a list of 17-channel planes
    arr_np = np.stack(batch_of_inputs, axis=0)
    t = torch.from_numpy(arr_np).to(device)
    return t

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

# ====================================================
#    Progress Bar Helpers
# ====================================================
def print_progress_bar(completed, total, start_time, prefix="Self-Play", bar_length=28):
    """
    Renders a live single-line progress bar:
      Self-Play: [██████████░░░░░░░░░░] 250/500 (50.0%) | 1.8 games/s | Elapsed: 00:02:15 | ETA: 00:02:15
    """
    pct = completed / total if total > 0 else 1.0
    filled = int(bar_length * pct)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    elapsed = max(0.001, time.time() - start_time)
    speed = completed / elapsed
    
    remaining = max(0, total - completed)
    eta_sec = int(remaining / speed) if speed > 0 else 0
    
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(int(elapsed)))
    eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))
    
    output = f"\r  {prefix}: [{bar}] {completed}/{total} ({pct*100:5.1f}%) | {speed:.4f} games/s | Elapsed: {elapsed_str} | ETA: {eta_str}"
    sys.stdout.write(output)
    sys.stdout.flush()
    if completed >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()

def print_train_progress(epoch, total_epochs, batch, total_batches, loss, p_loss, v_loss, start_time, bar_length=20):
    pct = batch / total_batches if total_batches > 0 else 1.0
    filled = int(bar_length * pct)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    elapsed = max(0.001, time.time() - start_time)
    speed = batch / elapsed
    
    output = f"\r  Epoch {epoch}/{total_epochs} [{bar}] {batch}/{total_batches} ({pct*100:4.1f}%) | Loss: {loss:.4f} (P: {p_loss:.3f}, V: {v_loss:.3f}) | {speed:4.1f} batch/s"
    sys.stdout.write(output)
    sys.stdout.flush()
    if batch >= total_batches:
        sys.stdout.write("\n")
        sys.stdout.flush()

# ====================================================
#    MCTS Node & AlphaZero PUCT MCTS
# ====================================================
class NN_MCTS_Node:
    def __init__(self, game_state: GoGame = None, parent=None, move=None, prior=0.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.is_expanded = False

        # MCTS statistics
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.virtual_loss = 0.0  # For parallel batched MCTS exploration
        self.P = float(prior)    # Policy prior probability from neural network
        self.value = 0.0         # Position value evaluation

    def ensure_game_state(self):
        """Lazily materializes the GoGame board state only when the node is actually visited."""
        if self.game_state is None and self.parent is not None:
            ng = self.parent.ensure_game_state().copy()
            ng.make_move(*self.move, check_valid=False)
            self.game_state = ng
        return self.game_state

def mcts_select(root: "NN_MCTS_Node"):
    """
    Selects a leaf node to evaluate. Traverses expanded internal nodes
    using the PUCT formula: Q(s, a) + U(s, a) until reaching an unexpanded
    node or a terminal game state (where node.children is empty).
    """
    node = root
    search_path = [node]
    while node.is_expanded and node.children:
        best_score = -float('inf')
        best_child = None
        
        # Parent visit count for PUCT exploration term
        parent_n_sqrt = math.sqrt(max(1.0, node.N + node.virtual_loss))
        
        for mv, ch in node.children.items():
            total_n = ch.N + ch.virtual_loss
            U = C_PUCT * ch.P * parent_n_sqrt / (1.0 + total_n)
            Q_val = (ch.W - ch.virtual_loss) / total_n if total_n > 0 else 0.0
            score = Q_val + U
            if score > best_score:
                best_score = score
                best_child = ch
        
        if best_child is None:
            break
        node = best_child
        search_path.append(node)

    node.ensure_game_state()
    return node, search_path

def nn_batch_evaluate_and_expand(nodes, net=None, worker_id=None, req_queue=None, res_queue=None):
    """
    Evaluates leaf nodes with the neural network (or exact game rules if terminal),
    assigns value predictions, and expands legal children with their prior probabilities.
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

    batch = [nd.game_state.get_nn_input() for nd in eval_nodes]

    if net is not None:
        input_tensor = boards_to_tensor(batch)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
            policy_logits, value_out = net(input_tensor)
        policy_logits = policy_logits.float().cpu().numpy()
        value_out = value_out.float().squeeze(1).cpu().numpy()
    else:
        req_queue.put((worker_id, batch))
        policy_logits, value_out = res_queue.get()

    board_s = eval_nodes[0].game_state.board_size
    for i, nd in enumerate(eval_nodes):
        nd.value = float(value_out[i])
        legal_moves = nd.game_state.get_legal_moves()

        move_indices = [(board_s * board_s if mv == (-1, -1) else mv[0] * board_s + mv[1]) for mv in legal_moves]
        selected_logits = [policy_logits[i][idx] for idx in move_indices]
        if len(selected_logits) > 0:
            probs = softmax(np.array(selected_logits))
        else:
            probs = []

        for mv, p in zip(legal_moves, probs):
            nd.children[mv] = NN_MCTS_Node(game_state=None, parent=nd, move=mv, prior=p)
        nd.is_expanded = True

def apply_virtual_loss(search_path, loss=1.0):
    for node in search_path:
        node.virtual_loss += loss

def remove_virtual_loss(search_path, loss=1.0):
    for node in search_path:
        node.virtual_loss -= loss

def mcts_backup(search_path, value):
    """
    Backpropagates simulation value up the search tree.
    Because Go is a two-player zero-sum game, the value of the state for the
    player whose turn it is translates into -value for the parent who took
    the action leading into that state.
    """
    curr_v = -value
    for node in reversed(search_path):
        node.N += 1
        node.W += curr_v
        node.Q = node.W / node.N
        curr_v = -curr_v

def mcts_run(root: "NN_MCTS_Node", net, n_simulations=50, temp=1.0, batch_size=8, worker_id=None, req_queue=None, res_queue=None):
    sims = 0
    while sims < n_simulations:
        # Collect a batch of unique leaves concurrently and apply virtual loss so paths diverge
        batch_paths = []
        batch_leaves = []
        
        current_batch_size = min(batch_size, n_simulations - sims)
        for _ in range(current_batch_size):
            leaf, path = mcts_select(root)
            if leaf in batch_leaves:
                break # Prevents evaluating the exact same leaf redundantly in one batch
            apply_virtual_loss(path)
            batch_paths.append(path)
            batch_leaves.append(leaf)

        if batch_leaves:
            nn_batch_evaluate_and_expand(batch_leaves, net=net, worker_id=worker_id, req_queue=req_queue, res_queue=res_queue)

            for path, leaf in zip(batch_paths, batch_leaves):
                remove_virtual_loss(path)
                mcts_backup(path, leaf.value)
                
            sims += len(batch_leaves)
            if pygame.get_init():
                pygame.event.pump()
        else:
            sims += 1

    move_N = [(mv, ch.N) for mv, ch in root.children.items()]
    if not move_N:
        return {}
    moves, visits = zip(*move_N)
    visits = np.array(visits, dtype=np.float32)
    if temp <= 0.05:
        best_idx = np.argmax(visits)
        probs = np.zeros_like(visits)
        probs[best_idx] = 1.0
    else:
        v = visits ** (1.0 / temp)
        v_sum = np.sum(v)
        if v_sum > 0 and np.isfinite(v_sum):
            probs = v / v_sum
        else:
            best_idx = np.argmax(visits)
            probs = np.zeros_like(visits)
            probs[best_idx] = 1.0
    policy_dict = dict(zip(moves, probs))
    return policy_dict

# ====================================================
#        Self-Play: Generating Training Data
# ====================================================

def gpu_server_thread(net, req_queue, res_queues, total_workers):
    active_workers = total_workers
    while active_workers > 0:
        requests = []
        try:
            req = req_queue.get(timeout=0.1)
            if req == "DONE":
                active_workers -= 1
                continue
            requests.append(req)
            while len(requests) < GPU_SERVER_BATCH_SIZE:
                try:
                    req = req_queue.get_nowait()
                    if req == "DONE":
                        active_workers -= 1
                        continue
                    requests.append(req)
                except queue.Empty:
                    break
        except queue.Empty:
            continue
            
        if requests:
            all_inputs = []
            for w_id, inputs in requests:
                all_inputs.extend(inputs)
                
            input_tensor = boards_to_tensor(all_inputs)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                policy_logits, value_out = net(input_tensor)
            policy_logits = policy_logits.float().cpu().numpy()
            value_out = value_out.float().squeeze(1).cpu().numpy()
            
            offset = 0
            for w_id, inputs in requests:
                n = len(inputs)
                p = policy_logits[offset:offset+n]
                v = value_out[offset:offset+n]
                res_queues[w_id].put((p, v))
                offset += n

def self_play_wrapper(board_size, n_mcts_sims, temp, worker_id, num_games, req_queue, res_queue, progress_queue):
    data = []
    for g in range(num_games):
        game_data = self_play_one_game(net=None, board_size=board_size, n_mcts_sims=n_mcts_sims, temp=temp, game_idx=g+1, worker_id=worker_id, req_queue=req_queue, res_queue=res_queue)
        data.extend(game_data)
        progress_queue.put(1)
    return data

def self_play_one_game(net=None, board_size=BOARD_SIZE, n_mcts_sims=50, temp=1.0, game_idx=1, worker_id=None, req_queue=None, res_queue=None):
    game = GoGame(board_size)
    data = []  # store (nn_input, player, pi, z)
    move_count = 0
    max_moves = board_size * board_size * 2  # Safety cutoff against infinite game loops
    temp_threshold = get_temp_threshold(board_size)
    dirichlet_alpha = get_dirichlet_alpha(board_size)

    logging.info(f"  Starting new self-play game {game_idx} on worker {worker_id} (board={board_size}x{board_size}).")
    
    while not game.is_game_over() and move_count < max_moves:
        current_temp = 1.0 if move_count < temp_threshold else 1e-3

        root = NN_MCTS_Node(game.copy())
        nn_batch_evaluate_and_expand([root], net=net, worker_id=worker_id, req_queue=req_queue, res_queue=res_queue)
        root.N = 1
        root.W = -root.value
        root.Q = -root.value

        # Add Dirichlet noise to the root node for exploration
        moves = list(root.children.keys())
        if len(moves) > 0:
            noise = np.random.dirichlet([dirichlet_alpha] * len(moves))
            for i, mv in enumerate(moves):
                root.children[mv].P = (1.0 - DIRICHLET_EPSILON) * root.children[mv].P + DIRICHLET_EPSILON * noise[i]

        pi_dict = mcts_run(root, net, n_mcts_sims, temp=current_temp, batch_size=MCTS_BATCH_SIZE, worker_id=worker_id, req_queue=req_queue, res_queue=res_queue)
        pi_flat = np.zeros(board_size * board_size + 1, dtype=np.float32)
        for mv, p in pi_dict.items():
            if mv[0] == -1 and mv[1] == -1:
                idx = board_size * board_size
            else:
                idx = mv[0]*board_size + mv[1]
            pi_flat[idx] = p

        if len(pi_dict) > 0:
            moves, probs = zip(*pi_dict.items())
            selected_move = random.choices(moves, weights=probs, k=1)[0]
        else:
            break

        nn_input = game.get_nn_input()
        current_player = game.current_player
        data.append((nn_input, current_player, pi_flat))

        game.make_move(*selected_move)
        move_count += 1
        
        logging.info(f"Game {game_idx} Move {move_count}: Player {current_player} -> {selected_move}")

    winner = game.get_winner()
    logging.info(f"Game {game_idx} over. Winner: Player {winner} after {move_count} moves.")
    
    for i, (nn_input_idx, player, pi_flat) in enumerate(data):
        if winner == 0:
            z = 0
        elif winner == player:
            z = 1
        else:
            z = -1
        data[i] = (nn_input_idx, player, pi_flat, z)
    return data

def get_equi_data(play_data):
    """
    Augment data using all 8 symmetries of the board (rotations and reflections).
    play_data: list of (nn_input, current_player, pi_flat, z)
    """
    extend_data = []
    for (state, cplayer, pi, z) in play_data:
        board_size = state.shape[1]
        # Reshape the policy vector (excluding the pass move) back to the board dimensions
        pi_board = np.reshape(pi[:-1], (board_size, board_size))
        pi_pass = pi[-1]
        
        for i in [1, 2, 3, 4]:
            # Rotate counter-clockwise (over H and W axes for 3D state)
            equi_state = np.rot90(state, k=i, axes=(1, 2))
            equi_pi_board = np.rot90(pi_board, k=i)
            equi_pi = np.zeros_like(pi)
            equi_pi[:-1] = equi_pi_board.flatten()
            equi_pi[-1] = pi_pass
            extend_data.append((equi_state.copy(), cplayer, equi_pi.copy(), z))
            
            # Flip horizontally (across the W axis for 3D state)
            equi_state_flip = np.flip(equi_state, axis=2)
            equi_pi_board_flip = np.fliplr(equi_pi_board)
            equi_pi_flip = np.zeros_like(pi)
            equi_pi_flip[:-1] = equi_pi_board_flip.flatten()
            equi_pi_flip[-1] = pi_pass
            extend_data.append((equi_state_flip.copy(), cplayer, equi_pi_flip.copy(), z))
            
    return extend_data

# ====================================================
#  Replay Buffer (Stores data across iterations)
# ====================================================
REPLAY_BUFFER = []

def add_to_replay_buffer(new_data):
    global REPLAY_BUFFER
    REPLAY_BUFFER.extend(new_data)
    if len(REPLAY_BUFFER) > MAX_REPLAY_BUFFER_SIZE:
        excess = len(REPLAY_BUFFER) - MAX_REPLAY_BUFFER_SIZE
        REPLAY_BUFFER = REPLAY_BUFFER[excess:]

def sample_from_replay_buffer(sample_size=1024):
    global REPLAY_BUFFER
    if len(REPLAY_BUFFER) == 0:
        return []
    if sample_size >= len(REPLAY_BUFFER):
        return REPLAY_BUFFER
    return random.sample(REPLAY_BUFFER, sample_size)

# ====================================================
#          Training the Network
# ====================================================
def train_policy_value_net(net, data, batch_size=TRAIN_BATCH_SIZE, epochs=TRAIN_EPOCHS, lr=LEARNING_RATE, optimizer=None):
    """
    data: list of (nn_input, current_player, pi_flat, z)
    We'll do:
      - policy loss (cross entropy w.r.t. pi_flat)
      - value loss (MSE vs. z)
    """
    if optimizer is None:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    net.train()

    X_inputs = []
    policy_targets = []
    value_targets = []
    board_size = net.board_size

    for (nn_inp, cplayer, pi_flat, z) in data:
        X_inputs.append(nn_inp)
        policy_targets.append(pi_flat)
        value_targets.append(z)

    policy_targets = np.array(policy_targets, dtype=np.float32)
    value_targets = np.array(value_targets, dtype=np.float32).reshape(-1,1)

    n_samples = len(data)
    indices = np.arange(n_samples)
    total_batches = (n_samples + batch_size - 1) // batch_size

    for ep in range(epochs):
        epoch_start_time = time.time()
        np.random.shuffle(indices)
        batch_losses = []
        for batch_idx, start_idx in enumerate(range(0, n_samples, batch_size)):
            end_idx = start_idx + batch_size
            excerpt = indices[start_idx:end_idx]

            batch_inputs = [X_inputs[i] for i in excerpt]
            inp = boards_to_tensor(batch_inputs)
            tgt_p = torch.from_numpy(policy_targets[excerpt]).to(device)
            tgt_v = torch.from_numpy(value_targets[excerpt]).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out_p, out_v = net(inp)
                out_p = out_p[:, :board_size*board_size + 1]

                logp = F.log_softmax(out_p, dim=1)
                policy_loss = -torch.sum(tgt_p * logp, dim=1).mean()

                value_loss = F.mse_loss(out_v, tgt_v)
                loss = policy_loss + value_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(loss.item())

            # Print live progress bar in the terminal
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print_train_progress(ep + 1, epochs, batch_idx + 1, total_batches, loss.item(), policy_loss.item(), value_loss.item(), epoch_start_time)

        logging.info(f"Epoch {ep+1}/{epochs}, Loss={np.mean(batch_losses):.4f}")

    net.eval()

# ====================================================
#             Putting it All Together
# ====================================================
def run_interactive_game(net, human_color="black"):
    board_size = net.board_size
    board_pixel_w = (board_size + 1) * GRID_SIZE
    board_pixel_h = (board_size + 1) * GRID_SIZE
    panel_height = 140

    window_width = max(board_pixel_w, 480)
    window_height = board_pixel_h + panel_height
    offset_x = (window_width - board_pixel_w) // 2
    offset_y = 0

    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"OceanGo ({board_size}x{board_size} NN MCTS)")

    font_title = pygame.font.Font(None, 26)
    font_body = pygame.font.Font(None, 21)
    font_small = pygame.font.Font(None, 18)
    font_btn = pygame.font.Font(None, 18)

    human_player = 2 if str(human_color).lower() in ("white", "w", "2") else 1
    ai_player = 3 - human_player
    h_col_str = "Black" if human_player == 1 else "White"
    ai_col_str = "White" if human_player == 1 else "Black"

    game = GoGame(board_size)
    clock = pygame.time.Clock()
    last_move = None
    game_over = False
    show_modal = False
    resigned_by = None
    score_info = None
    current_dead_stones = set()
    msg_str = "Click an intersection to play. P = Pass" if human_player == 1 else f"AI moves first as {ai_col_str}. Thinking..."

    # Button geometry in control panel
    btn_y = board_pixel_h + 82
    btn_h = 42
    margin = 16
    spacing = 10
    avail_w = window_width - 2 * margin - 2 * spacing
    btn_w = avail_w // 3

    btn_pass_rect = pygame.Rect(margin, btn_y, btn_w, btn_h)
    btn_end_rect = pygame.Rect(margin + btn_w + spacing, btn_y, btn_w, btn_h)
    btn_resign_rect = pygame.Rect(margin + (btn_w + spacing) * 2, btn_y, btn_w, btn_h)

    # Modal geometry
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

    def reset_game(swap_sides=False):
        nonlocal game, last_move, game_over, show_modal, resigned_by, score_info, current_dead_stones, human_player, ai_player, msg_str, h_col_str, ai_col_str
        if swap_sides:
            human_player = 3 - human_player
            ai_player = 3 - human_player
            h_col_str = "Black" if human_player == 1 else "White"
            ai_col_str = "White" if human_player == 1 else "Black"
        game = GoGame(board_size)
        last_move = None
        game_over = False
        show_modal = False
        resigned_by = None
        score_info = None
        current_dead_stones = set()
        if human_player == 1:
            msg_str = f"New game! You play {h_col_str} (first move). P = Pass"
        else:
            msg_str = f"New game! You play {h_col_str}. AI plays {ai_col_str} (thinking)..."

    def trigger_game_over(resigned=None):
        nonlocal game_over, show_modal, resigned_by, score_info, current_dead_stones, msg_str
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
        # Background
        screen.fill((15, 23, 42))

        # 1. Wood Go Board
        pygame.draw.rect(screen, (220, 179, 92), (offset_x, offset_y, board_pixel_w, board_pixel_h))
        pygame.draw.rect(screen, (160, 120, 50), (offset_x, offset_y, board_pixel_w, board_pixel_h), 2)

        # Grid lines
        for i in range(1, board_size + 1):
            gx = offset_x + i * GRID_SIZE
            gy = offset_y + i * GRID_SIZE
            pygame.draw.line(screen, (0, 0, 0),
                             (gx, offset_y + GRID_SIZE),
                             (gx, offset_y + board_pixel_h - GRID_SIZE), 2)
            pygame.draw.line(screen, (0, 0, 0),
                             (offset_x + GRID_SIZE, gy),
                             (offset_x + board_pixel_w - GRID_SIZE, gy), 2)

        # Star points (Hoshi)
        if board_size == 19:
            star_points = [(3, 3), (3, 9), (3, 15),
                           (9, 3), (9, 9), (9, 15),
                           (15, 3), (15, 9), (15, 15)]
        elif board_size == 13:
            star_points = [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
        elif board_size == 9:
            star_points = [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
        else:
            star_points = []
        for sx, sy in star_points:
            px = offset_x + (sy + 1) * GRID_SIZE
            py = offset_y + (sx + 1) * GRID_SIZE
            pygame.draw.circle(screen, (0, 0, 0), (px, py), 4)

        # 2. Stones (alive rendered solid, dead rendered dimmed/translucent)
        for x in range(board_size):
            for y in range(board_size):
                cell = game.board[x, y]
                if cell == 0:
                    continue
                cx = offset_x + (y + 1) * GRID_SIZE
                cy = offset_y + (x + 1) * GRID_SIZE
                radius = GRID_SIZE // 2 - 4

                is_dead = game_over and score_info and ((x, y) in score_info["dead_stones"])

                if is_dead:
                    # Draw dead stone with transparency
                    dim_surf = pygame.Surface((radius * 2 + 4, radius * 2 + 4), pygame.SRCALPHA)
                    center = (radius + 2, radius + 2)
                    if cell == 1:
                        # Dead black stone
                        pygame.draw.circle(dim_surf, (15, 15, 15, 90), center, radius)
                    else:
                        # Dead white stone
                        pygame.draw.circle(dim_surf, (250, 250, 250, 90), center, radius)
                        pygame.draw.circle(dim_surf, (20, 20, 20, 90), center, radius, 1)
                    screen.blit(dim_surf, (cx - radius - 2, cy - radius - 2))
                else:
                    # Normal alive stone
                    if cell == 1:
                        pygame.draw.circle(screen, (15, 15, 15), (cx, cy), radius)
                    elif cell == 2:
                        pygame.draw.circle(screen, (250, 250, 250), (cx, cy), radius)
                        pygame.draw.circle(screen, (20, 20, 20), (cx, cy), radius, 1)

        # 3. End-Game Scoring Squares (Added to ALL Black & White score points, including dead stones)
        if game_over and score_info:
            sq_size = 10
            half_sq = sq_size // 2

            # --- ALL BLACK SCORE POINTS (Black Squares) ---
            # Living Black stones: Black square with white border
            for r, c in score_info["b_alive_stones"]:
                cx = offset_x + (c + 1) * GRID_SIZE
                cy = offset_y + (r + 1) * GRID_SIZE
                sq_rect = pygame.Rect(cx - half_sq, cy - half_sq, sq_size, sq_size)
                pygame.draw.rect(screen, (15, 15, 15), sq_rect)
                pygame.draw.rect(screen, (240, 240, 240), sq_rect, 1)

            # Black territory points (empty territory + dead white stones)
            for r, c in score_info["b_terr_points"]:
                cx = offset_x + (c + 1) * GRID_SIZE
                cy = offset_y + (r + 1) * GRID_SIZE
                sq_rect = pygame.Rect(cx - half_sq, cy - half_sq, sq_size, sq_size)
                pygame.draw.rect(screen, (15, 15, 15), sq_rect)
                if (r, c) in score_info["dead_stones"]:
                    # Dead white stone inside black territory: outline for contrast
                    pygame.draw.rect(screen, (240, 240, 240), sq_rect, 1)

            # --- ALL WHITE SCORE POINTS (White Squares) ---
            # Living White stones: White square with dark border
            for r, c in score_info["w_alive_stones"]:
                cx = offset_x + (c + 1) * GRID_SIZE
                cy = offset_y + (r + 1) * GRID_SIZE
                sq_rect = pygame.Rect(cx - half_sq, cy - half_sq, sq_size, sq_size)
                pygame.draw.rect(screen, (250, 250, 250), sq_rect)
                pygame.draw.rect(screen, (15, 15, 15), sq_rect, 1)

            # White territory points (empty territory + dead black stones)
            for r, c in score_info["w_terr_points"]:
                cx = offset_x + (c + 1) * GRID_SIZE
                cy = offset_y + (r + 1) * GRID_SIZE
                sq_rect = pygame.Rect(cx - half_sq, cy - half_sq, sq_size, sq_size)
                pygame.draw.rect(screen, (250, 250, 250), sq_rect)
                pygame.draw.rect(screen, (15, 15, 15), sq_rect, 1)

        # Last move highlight (only while game is in progress)
        if not game_over and last_move and last_move != (-1, -1):
            lx, ly = last_move
            cx = offset_x + (ly + 1) * GRID_SIZE
            cy = offset_y + (lx + 1) * GRID_SIZE
            pygame.draw.circle(screen, (239, 68, 68), (cx, cy), 4)

        # 4. Control Panel
        panel_rect = pygame.Rect(0, board_pixel_h, window_width, panel_height)
        pygame.draw.rect(screen, (24, 32, 47), panel_rect)
        pygame.draw.line(screen, (51, 65, 85), (0, board_pixel_h), (window_width, board_pixel_h), 2)

        # Turn indicator
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

        # Score / Stone counts on top-right of panel
        if game_over and score_info and not resigned_by:
            counts_txt = font_small.render(f"Score: B {score_info['b_score']:.1f} | W {score_info['w_score']:.1f} (incl. 7.5 Komi)", True, (251, 191, 36))
        else:
            b_count = int(np.count_nonzero(game.board == 1))
            w_count = int(np.count_nonzero(game.board == 2))
            counts_txt = font_small.render(f"Stones: B {b_count} | W {w_count} (+7.5 Komi)", True, (148, 163, 184))
        screen.blit(counts_txt, (window_width - counts_txt.get_width() - 16, ind_cy - counts_txt.get_height() // 2))

        # Status / Alert message banner
        is_alert = any(k in msg_str.lower() for k in ("passed", "game over", "resign", "alert", "caution", "dead"))
        msg_color = (251, 191, 36) if is_alert else (203, 213, 225)
        msg_surf = font_small.render(msg_str, True, msg_color)
        screen.blit(msg_surf, (16, board_pixel_h + 48))

        # Control Buttons in bottom panel
        if game_over and not show_modal:
            # Buttons when user is reviewing the board with popup closed
            draw_button(screen, btn_pass_rect, "Show Score", font_btn, (37, 99, 235), (59, 130, 246))
            draw_button(screen, btn_end_rect, "Play Again (R)", font_btn, (16, 185, 129), (52, 211, 153))
            draw_button(screen, btn_resign_rect, "Swap Side (S)", font_btn, (139, 92, 246), (124, 58, 237))
        else:
            draw_button(screen, btn_pass_rect, "Pass (P)", font_btn, (37, 99, 235), (59, 130, 246))
            draw_button(screen, btn_end_rect, "Count / End", font_btn, (16, 185, 129), (52, 211, 153))
            draw_button(screen, btn_resign_rect, "Resign", font_btn, (220, 38, 38), (239, 68, 68))

        # 5. Game Over Modal Overlay
        if game_over and show_modal:
            dim_surf = pygame.Surface((window_width, board_pixel_h), pygame.SRCALPHA)
            dim_surf.fill((0, 0, 0, 160))
            screen.blit(dim_surf, (0, 0))

            # Modal Box
            pygame.draw.rect(screen, (24, 32, 47), modal_rect, border_radius=12)
            pygame.draw.rect(screen, (71, 85, 105), modal_rect, width=2, border_radius=12)

            # Close [X] Button in top-right of modal
            mouse_pos = pygame.mouse.get_pos()
            is_close_hov = modal_close_rect.collidepoint(mouse_pos)
            close_col = (220, 38, 38) if is_close_hov else (51, 65, 85)
            pygame.draw.rect(screen, close_col, modal_close_rect, border_radius=4)
            x_txt = font_small.render("X", True, (255, 255, 255))
            screen.blit(x_txt, x_txt.get_rect(center=modal_close_rect.center))

            title_surf = font_title.render("GAME OVER", True, (251, 191, 36))
            screen.blit(title_surf, (modal_rect.centerx - title_surf.get_width() // 2, modal_rect.y + 16))

            # Winner headline
            if resigned_by == human_player:
                res_surf = font_body.render(f"AI Wins ({ai_col_str} - You Resigned)", True, (239, 68, 68))
            elif resigned_by == ai_player:
                res_surf = font_body.render(f"You Win ({h_col_str} - AI Resigned)", True, (34, 197, 94))
            elif score_info:
                w = score_info["winner"]
                m = score_info["margin"]
                if w == human_player:
                    res_surf = font_body.render(f"🎉 You Won ({h_col_str})! (+{m:.1f} pts)", True, (34, 197, 94))
                elif w == ai_player:
                    res_surf = font_body.render(f"🤖 AI Won ({ai_col_str})! (+{m:.1f} pts)", True, (96, 165, 250))
                else:
                    res_surf = font_body.render("🤝 Game Tied!", True, (255, 255, 255))
            else:
                res_surf = font_body.render("Game Over", True, (255, 255, 255))
            screen.blit(res_surf, (modal_rect.centerx - res_surf.get_width() // 2, modal_rect.y + 48))

            # Score breakdown details
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

            # Modal action buttons (4 buttons: Review, Play Again, Swap Side, Exit)
            draw_button(screen, modal_review_rect, "Review", font_btn, (37, 99, 235), (59, 130, 246))
            draw_button(screen, modal_play_again_rect, "Play Again (R)", font_btn, (16, 185, 129), (52, 211, 153))
            draw_button(screen, modal_swap_rect, "Swap Side (S)", font_btn, (139, 92, 246), (124, 58, 237))
            draw_button(screen, modal_exit_rect, "Exit (Q)", font_btn, (71, 85, 105), (100, 116, 139))

        pygame.display.flip()

    while True:
        clock.tick(30)
        render_frame()

        # Handle events
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
                    elif event.key == pygame.K_ESCAPE or event.key == pygame.K_SPACE or event.key == pygame.K_b:
                        if show_modal:
                            show_modal = False
                            msg_str = "Reviewing board: Click stones to toggle Dead/Alive. [S] = Swap Sides."
                        else:
                            pygame.quit()
                            sys.exit()
                else:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_p and game.current_player == human_player:
                        # Human passed via keyboard
                        game.make_move(-1, -1)
                        last_move = (-1, -1)
                        logging.info(f"Interactive Game: Human ({h_col_str}) passed.")
                        if game.is_game_over():
                            trigger_game_over()
                            msg_str = "Both players passed. Game Over!"
                        else:
                            msg_str = "You passed. AI is thinking..."

                    elif (event.key == pygame.K_e or event.key == pygame.K_c) and not game_over:
                        # End and score
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
                            # Dismiss modal by clicking outside
                            show_modal = False
                            msg_str = "Reviewing board: Click stones to toggle Dead/Alive. [S] = Swap Sides."
                    else:
                        # Modal is hidden, user is reviewing board
                        if btn_pass_rect.collidepoint((mx, my)):
                            show_modal = True
                            msg_str = "Game over dialog open."
                        elif btn_end_rect.collidepoint((mx, my)):
                            reset_game(swap_sides=False)
                        elif btn_resign_rect.collidepoint((mx, my)):
                            reset_game(swap_sides=True)
                        elif my < board_pixel_h:
                            # Click on Go board during review: toggle stone group Dead / Alive!
                            grid_x = round((my - offset_y) / GRID_SIZE - 1)
                            grid_y = round((mx - offset_x) / GRID_SIZE - 1)
                            if 0 <= grid_x < board_size and 0 <= grid_y < board_size:
                                cx = offset_x + (grid_y + 1) * GRID_SIZE
                                cy = offset_y + (grid_x + 1) * GRID_SIZE
                                if (mx - cx) ** 2 + (my - cy) ** 2 <= (GRID_SIZE * 0.45) ** 2:
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
                    # In-game button clicks
                    if btn_pass_rect.collidepoint((mx, my)):
                        if game.current_player == human_player:
                            game.make_move(-1, -1)
                            last_move = (-1, -1)
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
                    elif my < board_pixel_h:
                        # Click on Go board
                        if game.current_player == human_player:
                            grid_x = round((my - offset_y) / GRID_SIZE - 1)
                            grid_y = round((mx - offset_x) / GRID_SIZE - 1)
                            if 0 <= grid_x < board_size and 0 <= grid_y < board_size:
                                cx = offset_x + (grid_y + 1) * GRID_SIZE
                                cy = offset_y + (grid_x + 1) * GRID_SIZE
                                if (mx - cx) ** 2 + (my - cy) ** 2 <= (GRID_SIZE * 0.45) ** 2:
                                    if game.is_valid_move(grid_x, grid_y):
                                        game.make_move(grid_x, grid_y)
                                        last_move = (grid_x, grid_y)
                                        logging.info(f"Interactive Game: Human ({h_col_str}) played ({grid_x}, {grid_y})")
                                        msg_str = f"You played ({grid_x}, {grid_y}). AI is thinking..."
                                    else:
                                        msg_str = f"Illegal move at ({grid_x}, {grid_y}). Try another intersection."

        # AI Turn Handling
        if not game_over and game.current_player == ai_player:
            msg_str = f"AI is thinking ({ai_col_str})..."
            render_frame()

            logging.info(f"Interactive Game: AI thinking ({ai_col_str})...")
            root = NN_MCTS_Node(game.copy())
            nn_batch_evaluate_and_expand([root], net=net)
            root.N = 1
            root.W = -root.value
            root.Q = -root.value
            pi_dict = mcts_run(root, net, n_simulations=N_MCTS_SIMS, temp=1e-3, batch_size=MCTS_BATCH_SIZE)

            if len(pi_dict) == 0:
                logging.info(f"Interactive Game: AI ({ai_col_str}) forced pass (no moves).")
                game.make_move(-1, -1)
                last_move = (-1, -1)
                if game.is_game_over():
                    trigger_game_over()
                    msg_str = "Both players passed. Game Over!"
                else:
                    msg_str = "⚠️ AI passed! Click [Pass] to finish, or keep playing."
            else:
                best_move = max(pi_dict.items(), key=lambda elem: elem[1])[0]
                if best_move == (-1, -1):
                    logging.info(f"Interactive Game: AI ({ai_col_str}) voluntarily passed.")
                    game.make_move(-1, -1)
                    last_move = (-1, -1)
                    if game.is_game_over():
                        trigger_game_over()
                        msg_str = "Both players passed. Game Over!"
                    else:
                        msg_str = "⚠️ AI passed! Click [Pass] to finish, or keep playing."
                else:
                    logging.info(f"Interactive Game: AI ({ai_col_str}) played: {best_move}")
                    game.make_move(*best_move)
                    last_move = best_move
                    msg_str = f"AI played {best_move}. Your turn ({h_col_str})."

def main():
    global BOARD_SIZE, SCREEN_SIZE, DIRICHLET_ALPHA
    init_environment()
    parser = argparse.ArgumentParser(description="OceanGo - AlphaZero styled Go AI")
    parser.add_argument("--model", type=str, default="", help="Path to specific model to load (e.g. checkpoints/policy_value_net_iter_5.pth)")
    parser.add_argument("--play", action="store_true", help="Play interactively immediately with the Pygame window")
    parser.add_argument("--train", action="store_true", help="Start AlphaZero self-play training loop")
    parser.add_argument("--color", type=str, default="black", choices=["black", "white"], help="Player color: 'black' (moves first, default) or 'white' (AI moves first)")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick 5-second sanity test of self-play, MCTS, and neural training")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory where checkpoints and models will be saved")
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE, help=f"Go board dimension: 9, 13, or 19 (default: {BOARD_SIZE})")
    parser.add_argument("--games", type=int, default=N_GAMES_PER_ITER, help=f"Number of self-play games per iteration (default: {N_GAMES_PER_ITER})")
    parser.add_argument("--sims", type=int, default=N_MCTS_SIMS, help=f"Number of MCTS simulations per move (default: {N_MCTS_SIMS})")
    parser.add_argument("--iters", type=int, default=N_ITER, help=f"Number of outer loop iterations (default: {N_ITER})")
    args = parser.parse_args()

    BOARD_SIZE = args.board_size
    SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE
    DIRICHLET_ALPHA = get_dirichlet_alpha(BOARD_SIZE)
    
    # Detect highest existing iteration checkpoint in save_dir for seamless auto-resume
    start_iter = 0
    latest_iter_path = ""
    if os.path.isdir(args.save_dir):
        for fname in os.listdir(args.save_dir):
            if fname.startswith("policy_value_net_iter_") and fname.endswith(".pth"):
                num_Part = fname[len("policy_value_net_iter_"):-len(".pth")]
                if num_Part.isdigit():
                    it_num = int(num_Part)
                    if it_num > start_iter:
                        start_iter = it_num
                        latest_iter_path = os.path.join(args.save_dir, fname)

    # Decide which model to load: explicit --model > save_dir/policy_value_net.pth > save_dir/latest_iter > ./policy_value_net.pth
    if args.model:
        load_path = args.model
    elif os.path.isfile(os.path.join(args.save_dir, "policy_value_net.pth")):
        load_path = os.path.join(args.save_dir, "policy_value_net.pth")
    elif latest_iter_path and os.path.isfile(latest_iter_path):
        load_path = latest_iter_path
    else:
        load_path = "policy_value_net.pth"

    # 1) Create or load network
    net = PolicyValueNet(board_size=BOARD_SIZE).to(device)
    
    if os.path.isfile(load_path):
        state_dict = torch.load(load_path, map_location=device)
        # Check if the checkpoint matches the requested board size
        expected_actions = BOARD_SIZE * BOARD_SIZE + 1
        ckpt_actions = state_dict["policy_fc.weight"].shape[0] if "policy_fc.weight" in state_dict else expected_actions
        if ckpt_actions != expected_actions:
            ckpt_board = int(round(math.sqrt(ckpt_actions - 1)))
            if args.model:
                print(f"Checkpoint '{load_path}' was trained on {ckpt_board}x{ckpt_board}; adjusting BOARD_SIZE to {ckpt_board}.")
                BOARD_SIZE = ckpt_board
                SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE
                DIRICHLET_ALPHA = get_dirichlet_alpha(BOARD_SIZE)
                net = PolicyValueNet(board_size=BOARD_SIZE).to(device)
                net.load_state_dict(state_dict)
            else:
                print(f"Existing checkpoint '{load_path}' is for {ckpt_board}x{ckpt_board}, but --board-size is {BOARD_SIZE}x{BOARD_SIZE}. Starting fresh {BOARD_SIZE}x{BOARD_SIZE} model.")
                start_iter = 0
        else:
            print(f"Loading existing model parameters from {load_path}...")
            logging.info(f"Loading existing model parameters from {load_path}")
            net.load_state_dict(state_dict)
            if start_iter > 0 and not args.dry_run and not args.play:
                print(f"Detected checkpoint history up to iteration {start_iter}. Will resume from iteration {start_iter + 1}.")
                logging.info(f"Resuming training from iteration {start_iter + 1}.")
    else:
        start_iter = 0
        print(f"No existing model found. Starting fresh {BOARD_SIZE}x{BOARD_SIZE} model from scratch.")
        logging.info(f"No existing model found. Starting a fresh {BOARD_SIZE}x{BOARD_SIZE} model.")
    net.eval()

    # Handle Dry-Run verification mode immediately (no prompts)
    if args.dry_run:
        print("\n[Dry-Run] Running quick end-to-end verification...")
        print(f"[Dry-Run] Hardware Device: {device_name} ({device.type.upper()})")
        print(f"[Dry-Run] Board Size: {BOARD_SIZE}x{BOARD_SIZE} (Temp cutoff: move {get_temp_threshold(BOARD_SIZE)}, Dirichlet alpha: {DIRICHLET_ALPHA})")
        t0 = time.time()
        test_data = self_play_one_game(net=net, board_size=BOARD_SIZE, n_mcts_sims=5, game_idx=1)
        print(f"[Dry-Run] 1 self-play game completed ({len(test_data)} states generated in {time.time()-t0:.2f}s).")
        aug_data = get_equi_data(test_data)
        print(f"[Dry-Run] Symmetrical augmentation produced {len(aug_data)} training samples.")
        print("[Dry-Run] Testing neural network backpropagation pass...")
        train_policy_value_net(net, aug_data, batch_size=min(16, len(aug_data)), epochs=1, lr=LEARNING_RATE)
        os.makedirs(args.save_dir, exist_ok=True)
        test_save_path = os.path.join(args.save_dir, "dry_run_test.pth")
        torch.save(net.state_dict(), test_save_path)
        print(f"[Dry-Run] Checkpoint write confirmed: {test_save_path}")
        print("[Dry-Run] SUCCESS: Everything is functional and verified ready for full training!\n")
        return

    # Determine mode: Play or Train
    should_play = args.play
    if not args.play and not args.train:
        print("\nSelect OceanGo Mode:")
        print("  [1] Play Go as Black (You move first vs AI White) [Default]")
        print("  [2] Play Go as White (AI moves first as Black)")
        print("  [3] Start AlphaZero AI Self-Play Training")
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

    # If the user wants to play against the model, boot up the UI and exit after the game.
    if should_play:
        print(f"Starting interactive Pygame window (You play {args.color.capitalize()})...")
        logging.info(f"Starting interactive Pygame window (You play {args.color.capitalize()}).")
        run_interactive_game(net, human_color=args.color)
        return

    checkpoints_dir = args.save_dir
    os.makedirs(checkpoints_dir, exist_ok=True)
    n_games = args.games
    n_sims = args.sims
    n_iters = args.iters

    # Persistent SGD optimizer with momentum across iterations
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    train_state_path = os.path.join(checkpoints_dir, f"train_state_{BOARD_SIZE}x{BOARD_SIZE}.pth")
    if start_iter > 0 and os.path.isfile(train_state_path):
        try:
            train_state = torch.load(train_state_path, map_location=device, weights_only=False)
            if "optimizer" in train_state:
                optimizer.load_state_dict(train_state["optimizer"])
            if "replay_buffer" in train_state and isinstance(train_state["replay_buffer"], list):
                add_to_replay_buffer(train_state["replay_buffer"])
                print(f"Restored optimizer momentum and {len(REPLAY_BUFFER)} replay buffer states from {train_state_path}.")
        except Exception as e:
            logging.warning(f"Could not restore auxiliary train_state ({e}); continuing with fresh optimizer state.")

    # 2) Self-play / 3) Train 
    total_start_time = time.time()
    end_iter = n_iters if start_iter < n_iters else start_iter + n_iters

    for iteration in range(start_iter, end_iter):
        iteration_start_time = time.time()
        current_lr = get_scheduled_lr(iteration, base_lr=LEARNING_RATE)
        print(f"==== Self-Play iteration {iteration+1}/{end_iter} (Board: {BOARD_SIZE}x{BOARD_SIZE}, LR: {current_lr}) ====")
        logging.info(f"==== Self-Play iteration {iteration+1}/{end_iter} (Board: {BOARD_SIZE}x{BOARD_SIZE}, LR: {current_lr}) ====")

        # Gather data from self-play (Parallelized)
        iteration_data = []
        print(f"  Starting {n_games} self-play games simultaneously (Parallel Execution)...")
        logging.info(f"  Starting {n_games} parallel games in iteration {iteration+1}...")

        # Dispatch parallel self-play games explicitly via multi-processing (Bypasses Python GIL)
        print(f"  Allocating {PARALLEL_THREADS} parallel processing cores (Central GPU Queue Mode).")
        logging.info(f"  Allocating {PARALLEL_THREADS} parallel processing cores (Central GPU Queue Mode).")

        mp_manager = mp.Manager()
        req_queue = mp_manager.Queue()
        res_queues = [mp_manager.Queue() for _ in range(PARALLEL_THREADS)]
        progress_queue = mp_manager.Queue()
        
        gpu_thread = threading.Thread(target=gpu_server_thread, args=(net, req_queue, res_queues, PARALLEL_THREADS))
        gpu_thread.start()

        with concurrent.futures.ProcessPoolExecutor(max_workers=PARALLEL_THREADS) as executor:
            futures = []
            games_per_worker = max(1, n_games // PARALLEL_THREADS)
            
            for w_id in range(PARALLEL_THREADS):
                num_games = games_per_worker
                if w_id == PARALLEL_THREADS - 1:
                    num_games = n_games - games_per_worker * (PARALLEL_THREADS - 1)
                
                if num_games > 0:
                    f = executor.submit(self_play_wrapper, BOARD_SIZE, n_sims, 1.0, w_id, num_games, req_queue, res_queues[w_id], progress_queue)
                    futures.append(f)
            
            completed_games = 0
            iteration_games_start_time = time.time()
            
            # Live Progress Bar Loop
            while completed_games < n_games:
                try:
                    progress_queue.get(timeout=0.5)
                    completed_games += 1
                    print_progress_bar(completed_games, n_games, iteration_games_start_time, prefix="Self-Play")
                    if completed_games % 25 == 0 or completed_games == n_games:
                        logging.info(f"  Self-play progress: {completed_games}/{n_games} games completed.")
                except queue.Empty:
                    pass
            
            for future in futures:
                game_data = future.result()
                augmented_data = get_equi_data(game_data)
                iteration_data.extend(augmented_data)
                req_queue.put("DONE")

        gpu_thread.join()

        # Add iteration_data to global replay buffer
        add_to_replay_buffer(iteration_data)
        print(f"Total new data this iteration: {len(iteration_data)} states (Replay buffer size: {len(REPLAY_BUFFER)}).")
        logging.info(f"Total new data in iteration {iteration+1}: {len(iteration_data)} states (Replay buffer: {len(REPLAY_BUFFER)}).")

        # Sample from the replay buffer for training
        train_data = sample_from_replay_buffer(sample_size=TRAIN_SAMPLE_SIZE)
        print(f"Training on replay buffer sample of size: {len(train_data)}")
        logging.info(f"Training on replay buffer sample of size: {len(train_data)}")

        if len(train_data) > 0:
            train_start_time = time.time()
            train_policy_value_net(net, train_data, batch_size=TRAIN_BATCH_SIZE, epochs=TRAIN_EPOCHS, lr=current_lr, optimizer=optimizer)
            train_end_time = time.time()
            logging.info(f"Training completed in {train_end_time - train_start_time:.2f} seconds.")
        else:
            print("No training data available yet.")

        # 4) Save the net (overwrites latest file and logs a checkpoint)
        torch.save(net.state_dict(), "policy_value_net.pth")
        checkpoint_path = os.path.join(checkpoints_dir, f"policy_value_net_iter_{iteration+1}.pth")
        torch.save(net.state_dict(), checkpoint_path)
        torch.save({"optimizer": optimizer.state_dict(), "replay_buffer": REPLAY_BUFFER}, train_state_path)
        # Also copy to checkpoints_dir/policy_value_net.pth if saving to custom dir (e.g. Google Drive)
        if checkpoints_dir != "." and checkpoints_dir != "checkpoints":
            torch.save(net.state_dict(), os.path.join(checkpoints_dir, "policy_value_net.pth"))
        print(f"Model state saved sequentially to {checkpoint_path}.")
        logging.info(f"Model saved at iteration {iteration+1} to checkpoint folder and base 'policy_value_net.pth'.")

        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        logging.info(f"Iteration {iteration+1} duration: {iteration_duration:.2f} seconds.")

    total_end_time = time.time()
    total_selfplay_time = total_end_time - total_start_time
    print(f"\nTraining complete! Total time: {total_selfplay_time:.2f}s. Final checkpoints saved to '{checkpoints_dir}'.")
    logging.info(f"Total self-play time across all iterations: {total_selfplay_time:.2f} seconds.")

if __name__ == "__main__":
    mp.freeze_support()
    main()
