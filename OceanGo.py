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
# Number of MCTS leaves to evaluate in parallel on the GPU. Higher = better GPU utilization during search.
MCTS_BATCH_SIZE = 256
# Exploration constant for PUCT formulation. Higher = explores more unvisited tree nodes prior to exploiting.
C_PUCT = 1.0
# Shape of Dirichlet noise. AlphaZero uses 0.03 for 19x19, higher (0.15) for 9x9 to ensure healthy exploration.
DIRICHLET_ALPHA = 0.03 if BOARD_SIZE >= 19 else 0.15
# Ratio of the Dirichlet noise added to the root node (0.25 means 25% noise / 75% real model policy).
DIRICHLET_EPSILON = 0.25

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
        temp_board = self.board.copy()
        temp_board[x, y] = self.current_player
        
        # Remove captures
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                if temp_board[nx, ny] == 3 - self.current_player:
                    if not self._has_liberty((nx, ny), temp_board):
                        self._remove_group((nx, ny), temp_board)
        
        # Check suicide
        if not self._has_liberty((x, y), temp_board):
            return False
            
        # Check Ko (positional superko)
        board_hash = temp_board.tobytes()
        if board_hash in self.history:
            return False
            
        return True

    def make_move(self, x, y, check_valid=True):
        if check_valid and not self.is_valid_move(x, y):
            return False
            
        if x == -1 and y == -1:
            self.passes_in_a_row += 1
        else:
            self.passes_in_a_row = 0
            self.board[x, y] = self.current_player
            
            # Remove captures
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
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
        from collections import deque
        color = board_state[start[0], start[1]]
        visited = set()
        q = deque([start])
        while q:
            cx, cy = q.pop()
            if (cx, cy) not in visited:
                visited.add((cx, cy))
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if board_state[nx, ny] == 0:
                            return True
                        if board_state[nx, ny] == color and (nx, ny) not in visited:
                            q.append((nx, ny))
        return False

    def _remove_group(self, start, board_state):
        from collections import deque
        color = board_state[start[0], start[1]]
        visited = set()
        q = deque([start])
        while q:
            cx, cy = q.pop()
            if (cx, cy) not in visited:
                visited.add((cx, cy))
                board_state[cx, cy] = 0
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if board_state[nx, ny] == color and (nx, ny) not in visited:
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
        if self.passes_in_a_row >= 2:
            return True
        if self._is_game_over_cache is not None:
            return self._is_game_over_cache
        if self._legal_moves_cache is not None:
            self._is_game_over_cache = len(self._legal_moves_cache) <= 1
            return self._is_game_over_cache
        # Early-exit scan: stop as soon as any single legal board placement is found
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == 0 and self._try_move(x, y):
                    self._is_game_over_cache = False
                    return False
        self._is_game_over_cache = True
        return True

    def get_winner(self):
        # Chinese area scoring with Komi
        b_stones = np.count_nonzero(self.board == 1)
        w_stones = np.count_nonzero(self.board == 2)
        b_terr = 0
        w_terr = 0
        visited = set()

        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == 0 and (r, c) not in visited:
                    from collections import deque
                    q = deque([(r, c)])
                    group = set([(r, c)])
                    touches_black = False
                    touches_white = False

                    while q:
                        cx, cy = q.pop()
                        visited.add((cx, cy))
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            nx, ny = cx+dx, cy+dy
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                                if self.board[nx, ny] == 1:
                                    touches_black = True
                                elif self.board[nx, ny] == 2:
                                    touches_white = True
                                elif self.board[nx, ny] == 0 and (nx, ny) not in group:
                                    group.add((nx, ny))
                                    q.append((nx, ny))
                    
                    if touches_black and not touches_white:
                        b_terr += len(group)
                    elif touches_white and not touches_black:
                        w_terr += len(group)

        komi = 7.5
        b_score = b_stones + b_terr
        w_score = w_stones + w_terr + komi

        if b_score > w_score:
            return 1
        elif w_score > b_score:
            return 2
        return 0

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
    
    output = f"\r  {prefix}: [{bar}] {completed}/{total} ({pct*100:5.1f}%) | {speed:4.1f} games/s | Elapsed: {elapsed_str} | ETA: {eta_str}"
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
            while len(requests) < 512:
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

    logging.info(f"  Starting new self-play game {game_idx} on worker {worker_id}.")
    
    while not game.is_game_over() and move_count < max_moves:
        current_temp = 1.0 if move_count < 30 else 1e-3

        root = NN_MCTS_Node(game.copy())
        nn_batch_evaluate_and_expand([root], net=net, worker_id=worker_id, req_queue=req_queue, res_queue=res_queue)

        # Add Dirichlet noise to the root node for exploration
        moves = list(root.children.keys())
        if len(moves) > 0:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
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
def train_policy_value_net(net, data, batch_size=TRAIN_BATCH_SIZE, epochs=TRAIN_EPOCHS, lr=LEARNING_RATE):
    """
    data: list of (nn_input, current_player, pi_flat, z)
    We'll do:
      - policy loss (cross entropy w.r.t. pi_flat)
      - value loss (MSE vs. z)
    """
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
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
def run_interactive_game(net):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Go Game (NN MCTS)")

    game = GoGame(BOARD_SIZE)
    clock = pygame.time.Clock()
    last_move = None

    def draw_board(screen, g: GoGame, last_mv):
        # Wood board color
        screen.fill((220, 179, 92))
        for i in range(1, BOARD_SIZE + 1):
            pygame.draw.line(screen, (0,0,0),
                             (i*GRID_SIZE, GRID_SIZE),
                             (i*GRID_SIZE, SCREEN_SIZE - GRID_SIZE), 2)
            pygame.draw.line(screen, (0,0,0),
                             (GRID_SIZE, i*GRID_SIZE),
                             (SCREEN_SIZE - GRID_SIZE, i*GRID_SIZE), 2)
                             
        # Draw star points (Hoshi) for 19x19 board
        if BOARD_SIZE == 19:
            star_points = [(3, 3), (3, 9), (3, 15),
                           (9, 3), (9, 9), (9, 15),
                           (15, 3), (15, 9), (15, 15)]
            for sx, sy in star_points:
                px = (sy + 1) * GRID_SIZE
                py = (sx + 1) * GRID_SIZE
                pygame.draw.circle(screen, (0,0,0), (px, py), 4)

        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                cell = g.board[x,y]
                cx = (y+1)*GRID_SIZE
                cy = (x+1)*GRID_SIZE
                if cell == 1:
                    pygame.draw.circle(screen, (0,0,0), (cx,cy), GRID_SIZE//2 - 5)
                elif cell == 2:
                    pygame.draw.circle(screen, (255,255,255), (cx,cy), GRID_SIZE//2 - 5)
                    pygame.draw.circle(screen, (0,0,0), (cx,cy), GRID_SIZE//2 - 5, 1) # Outline for white
                    
        # Highlight last move
        if last_mv and last_mv != (-1, -1):
            lx, ly = last_mv
            cx = (ly+1)*GRID_SIZE
            cy = (lx+1)*GRID_SIZE
            pygame.draw.circle(screen, (255, 0, 0), (cx,cy), 4)
            
        pygame.display.flip()

    while True:
        if game.is_game_over():
            draw_board(screen, game, last_move)
            w = game.get_winner()
            if w == 0:
                print("\nGame over! It's a tie.")
                logging.info("Interactive Game: Game over! It's a tie.")
            elif w == 1:
                print("\nGame over! You (Black) win!")
                logging.info("Interactive Game: Game over! Player 1 (Human) won.")
            else:
                print("\nGame over! AI (White) wins!")
                logging.info("Interactive Game: Game over! Player 2 (AI) won.")

            print("Press any key or close the window to exit.")
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                        waiting = False
                        break
                clock.tick(15)
            pygame.quit()
            sys.exit()

        draw_board(screen, game, last_move)
        clock.tick(15)

        # Player 1 = human
        if game.current_player == 1:
            x,y = None,None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        x, y = -1, -1 # Pass
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    px, py = event.pos
                    grid_x = round((py/GRID_SIZE) - 1)
                    grid_y = round((px/GRID_SIZE) - 1)
                    if 0 <= grid_x < BOARD_SIZE and 0 <= grid_y < BOARD_SIZE:
                        x,y = grid_x, grid_y
            if x is not None:
                if game.is_valid_move(x,y):
                    game.make_move(x,y)
                    last_move = (x, y)
                    if (x, y) == (-1, -1):
                        print("Human passed.")
                        logging.info("Interactive Game: Human passed.")
                    else:
                        print(f"Human played: ({x}, {y})")
                        logging.info(f"Interactive Game: Human played ({x}, {y})")
        else:
            # Player 2 = AI
            print("AI thinking...", end="", flush=True)
            logging.info("Interactive Game: AI thinking...")
            root = NN_MCTS_Node(game.copy())
            nn_batch_evaluate_and_expand([root], net=net)
            pi_dict = mcts_run(root, net, n_simulations=N_MCTS_SIMS, batch_size=MCTS_BATCH_SIZE)
            if len(pi_dict) == 0:
                print(" AI has no moves, passing!")
                logging.info("Interactive Game: AI forced pass (no moves).")
                game.make_move(-1, -1)
                last_move = (-1, -1)
            else:
                best_move = max(pi_dict.items(), key=lambda elem: elem[1])[0]
                if best_move == (-1, -1):
                    print(" AI passes!")
                    logging.info("Interactive Game: AI voluntarily passed.")
                else:
                    print(f" AI plays: {best_move}")
                    logging.info(f"Interactive Game: AI played: {best_move}")
                game.make_move(*best_move)
                last_move = best_move

def main():
    init_environment()
    parser = argparse.ArgumentParser(description="OceanGo - AlphaZero styled Go AI")
    parser.add_argument("--model", type=str, default="", help="Path to specific model to load (e.g. checkpoints/policy_value_net_iter_5.pth)")
    parser.add_argument("--play", action="store_true", help="Play interactively immediately with the Pygame window")
    parser.add_argument("--train", action="store_true", help="Start AlphaZero self-play training loop")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick 5-second sanity test of self-play, MCTS, and neural training")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory where checkpoints and models will be saved")
    parser.add_argument("--games", type=int, default=N_GAMES_PER_ITER, help=f"Number of self-play games per iteration (default: {N_GAMES_PER_ITER})")
    parser.add_argument("--sims", type=int, default=N_MCTS_SIMS, help=f"Number of MCTS simulations per move (default: {N_MCTS_SIMS})")
    parser.add_argument("--iters", type=int, default=N_ITER, help=f"Number of outer loop iterations (default: {N_ITER})")
    args = parser.parse_args()

    # 1) Create or load network
    net = PolicyValueNet(board_size=BOARD_SIZE).to(device)
    
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
    
    if os.path.isfile(load_path):
        print(f"Loading existing model parameters from {load_path}...")
        logging.info(f"Loading existing model parameters from {load_path}")
        net.load_state_dict(torch.load(load_path, map_location=device))
        if start_iter > 0 and not args.dry_run and not args.play:
            print(f"Detected checkpoint history up to iteration {start_iter}. Will resume from iteration {start_iter + 1}.")
            logging.info(f"Resuming training from iteration {start_iter + 1}.")
    else:
        start_iter = 0
        print("No existing model found. Starting from scratch.")
        logging.info("No existing model found. Starting a fresh model.")
    net.eval()

    # Handle Dry-Run verification mode immediately (no prompts)
    if args.dry_run:
        print("\n[Dry-Run] Running quick end-to-end verification...")
        print(f"[Dry-Run] Hardware Device: {device_name} ({device.type.upper()})")
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
        print("  [1] Play Go interactively (Pygame Window) [Default]")
        print("  [2] Start AlphaZero AI Self-Play Training")
        choice = ""
        try:
            if sys.stdin.isatty():
                choice = input("\nEnter choice (1 or 2, default is 1): ").strip()
            else:
                choice = "1"
        except (EOFError, KeyboardInterrupt):
            choice = "1"

        if choice == "2":
            should_play = False
        else:
            should_play = True

    # If the user wants to play against the model, boot up the UI and exit after the game.
    if should_play:
        print("Starting interactive Pygame window...")
        logging.info("Starting interactive Pygame window.")
        run_interactive_game(net)
        return

    checkpoints_dir = args.save_dir
    os.makedirs(checkpoints_dir, exist_ok=True)
    n_games = args.games
    n_sims = args.sims
    n_iters = args.iters

    # 2) Self-play / 3) Train 
    total_start_time = time.time()
    end_iter = n_iters if start_iter < n_iters else start_iter + n_iters

    for iteration in range(start_iter, end_iter):
        iteration_start_time = time.time()
        print(f"==== Self-Play iteration {iteration+1}/{end_iter} ====")
        logging.info(f"==== Self-Play iteration {iteration+1}/{end_iter} ====")

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
        print(f"Total new data this iteration: {len(iteration_data)} states.")
        logging.info(f"Total new data in iteration {iteration+1}: {len(iteration_data)} states.")

        # Sample from the replay buffer for training
        train_data = sample_from_replay_buffer(sample_size=TRAIN_SAMPLE_SIZE)
        print(f"Training on replay buffer sample of size: {len(train_data)}")
        logging.info(f"Training on replay buffer sample of size: {len(train_data)}")

        if len(train_data) > 0:
            train_start_time = time.time()
            train_policy_value_net(net, train_data, batch_size=TRAIN_BATCH_SIZE, epochs=TRAIN_EPOCHS, lr=LEARNING_RATE)
            train_end_time = time.time()
            logging.info(f"Training completed in {train_end_time - train_start_time:.2f} seconds.")
        else:
            print("No training data available yet.")

        # 4) Save the net (overwrites latest file and logs a checkpoint)
        torch.save(net.state_dict(), "policy_value_net.pth")
        checkpoint_path = os.path.join(checkpoints_dir, f"policy_value_net_iter_{iteration+1}.pth")
        torch.save(net.state_dict(), checkpoint_path)
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
