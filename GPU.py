import logging
import math
import random
import copy
import time
import sys
import os
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------------------
# Global config
# ----------------------------------------
BOARD_SIZE = 13
GRID_SIZE = 50
SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------
# Logging
# ----------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
log_start = time.strftime("%Y-%m-%d %H:%M:%S")
logging.basicConfig(filename=f"{script_dir}/logs.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.info(f"\n\nStarting log at {log_start}")

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
    def __init__(self, board_size=BOARD_SIZE, num_channels=64, num_res_blocks=5):
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
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch_size, 2, board_size, board_size)
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
        self.history.add(hash(self.board.tobytes()))
        self.passes_in_a_row = 0
        self.history_states = [np.zeros((board_size, board_size), dtype=np.int32) for _ in range(8)]
        self.history_states[0] = self.board.copy()

    def copy(self):
        g = GoGame(self.board_size)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.history = self.history.copy()
        g.passes_in_a_row = self.passes_in_a_row
        g.history_states = [b.copy() for b in self.history_states]
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
        board_hash = hash(temp_board.tobytes())
        if board_hash in self.history:
            return False
            
        return True

    def make_move(self, x, y):
        if not self.is_valid_move(x, y):
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
        self.history.add(hash(self.board.tobytes()))
        self.history_states.insert(0, self.board.copy())
        self.history_states.pop()
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
        moves = [(-1, -1)]  # Pass is always a legal move
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == 0 and self._try_move(x, y):
                    moves.append((x, y))
        return moves

    def is_game_over(self):
        return self.passes_in_a_row >= 2 or len(self.get_legal_moves()) <= 1

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
#    MCTS Node & Basic MCTS (w/ policy/value net)
# ====================================================
class NN_MCTS_Node:
    def __init__(self, game_state: GoGame, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.unexpanded_moves = self.game_state.get_legal_moves()

        # MCTS stats
        self.N = 0
        self.W = 0
        self.Q = 0
        self.virtual_loss = 0  # for parallel batched MCTS
        self.P = 0  
        self.policy_map = {}  
        self.value = 0        

def mcts_select(root: "NN_MCTS_Node"):
    node = root
    search_path = [node]
    while True:
        if len(node.unexpanded_moves) > 0:
            return node, search_path
        if not node.children:
            return node, search_path
        best_score = -999999
        best_child = None
        
        for mv, ch in node.children.items():
            c_puct = 1.0
            total_n = ch.N + ch.virtual_loss
            U = c_puct * ch.P * math.sqrt(node.N + node.virtual_loss) / (1 + total_n)
            Q_val = (ch.W - ch.virtual_loss * 1.0) / total_n if total_n > 0 else 0
            score = Q_val + U
            if score > best_score:
                best_score = score
                best_child = ch
        
        node = best_child
        search_path.append(node)

def nn_batch_evaluate(nodes, net):
    batch = []
    for nd in nodes:
        batch.append(nd.game_state.get_nn_input())

    input_tensor = boards_to_tensor(batch)
    with torch.no_grad():
        policy_logits, value_out = net(input_tensor)
    policy_logits = policy_logits.cpu().numpy()
    value_out = value_out.squeeze(1).cpu().numpy()

    for i, nd in enumerate(nodes):
        val = value_out[i]
        nd.value = float(val)
        p_logits = policy_logits[i]
        legal_moves = nd.unexpanded_moves
        board_s = nd.game_state.board_size

        move_indices = []
        for (x,y) in legal_moves:
            if x == -1 and y == -1:
                move_indices.append(board_s * board_s)
            else:
                move_indices.append(x * board_s + y)
        selected_logits = [p_logits[idx] for idx in move_indices]
        if len(selected_logits) > 0:
            probs = softmax(np.array(selected_logits))
        else:
            probs = []

        nd.policy_map = {}
        for mv, p in zip(legal_moves, probs):
            nd.policy_map[mv] = p

def mcts_expand(node: "NN_MCTS_Node"):
    for mv in node.unexpanded_moves:
        ng = node.game_state.copy()
        ng.make_move(*mv)
        child_node = NN_MCTS_Node(ng, parent=node, move=mv)
        child_node.P = node.policy_map.get(mv, 0.0)
        node.children[mv] = child_node
    node.unexpanded_moves = []

def apply_virtual_loss(search_path, loss=1.0):
    for node in search_path:
        node.virtual_loss += loss

def remove_virtual_loss(search_path, loss=1.0):
    for node in search_path:
        node.virtual_loss -= loss

def mcts_backup(search_path, value):
    # Backward pass for the current search path modifying true W/N 
    for node in reversed(search_path):
        node.N += 1
        node.W += value
        node.Q = node.W / node.N
        value = -value

def mcts_run(root: "NN_MCTS_Node", net, n_simulations=50, temp=1.0, batch_size=8):
    sims = 0
    while sims < n_simulations:
        # Collect a batch of leaves concurrently and apply virtual loss so paths diverge
        batch_paths = []
        batch_leaves = []
        
        current_batch_size = min(batch_size, n_simulations - sims)
        for _ in range(current_batch_size):
            leaf, path = mcts_select(root)
            apply_virtual_loss(path)
            batch_paths.append(path)
            batch_leaves.append(leaf)

        # Batch evaluate neural network logic on completely unique paths together 
        nn_batch_evaluate(batch_leaves, net)

        # Backpropagate actual values and remove virtual loss
        for path, leaf in zip(batch_paths, batch_leaves):
            mcts_expand(leaf)
            remove_virtual_loss(path)
            mcts_backup(path, leaf.value)
            
        sims += current_batch_size

    move_N = [(mv, ch.N) for mv, ch in root.children.items()]
    if not move_N:
        return {}
    moves, visits = zip(*move_N)
    visits = np.array(visits, dtype=np.float32)
    if temp < 1e-3:
        best_idx = np.argmax(visits)
        probs = np.zeros_like(visits)
        probs[best_idx] = 1.0
    else:
        v = visits ** (1.0 / temp)
        probs = v / np.sum(v)
    policy_dict = dict(zip(moves, probs))
    return policy_dict

# ====================================================
#        Self-Play: Generating Training Data
# ====================================================
def self_play_one_game(net, board_size=BOARD_SIZE, n_mcts_sims=50, temp=1.0):
    game = GoGame(board_size)
    data = []  # store (nn_input, player, pi, z)
    move_count = 0

    while not game.is_game_over():
        current_temp = 1.0 if move_count < 30 else 1e-3

        root = NN_MCTS_Node(game.copy())
        nn_batch_evaluate([root], net)
        mcts_expand(root)

        # Add Dirichlet noise to the root node for exploration
        moves = list(root.children.keys())
        if len(moves) > 0:
            noise = np.random.dirichlet([0.03] * len(moves))
            for i, mv in enumerate(moves):
                root.children[mv].P = 0.75 * root.children[mv].P + 0.25 * noise[i]

        pi_dict = mcts_run(root, net, n_mcts_sims, temp=current_temp, batch_size=8)
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

    winner = game.get_winner()
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
MAX_REPLAY_BUFFER_SIZE = 30000  # Example limit

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
def train_policy_value_net(net, data, batch_size=32, epochs=5, lr=1e-3):
    """
    data: list of (nn_input, current_player, pi_flat, z)
    We'll do:
      - policy loss (cross entropy w.r.t. pi_flat)
      - value loss (MSE vs. z)
    """
    optimizer = optim.Adam(net.parameters(), lr=lr)
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

    for ep in range(epochs):
        np.random.shuffle(indices)
        batch_losses = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            excerpt = indices[start_idx:end_idx]

            batch_inputs = [X_inputs[i] for i in excerpt]
            inp = boards_to_tensor(batch_inputs)
            tgt_p = torch.from_numpy(policy_targets[excerpt]).to(device)
            tgt_v = torch.from_numpy(value_targets[excerpt]).to(device)

            optimizer.zero_grad()
            out_p, out_v = net(inp)
            out_p = out_p[:, :board_size*board_size + 1]

            logp = F.log_softmax(out_p, dim=1)
            policy_loss = -torch.sum(tgt_p * logp, dim=1).mean()

            value_loss = F.mse_loss(out_v, tgt_v)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        print(f"Epoch {ep+1}/{epochs}, Loss={np.mean(batch_losses):.4f}")
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

    def draw_board(screen, g: GoGame):
        screen.fill((255, 255, 255))
        for i in range(1, BOARD_SIZE + 1):
            pygame.draw.line(screen, (0,0,0),
                             (i*GRID_SIZE, GRID_SIZE),
                             (i*GRID_SIZE, SCREEN_SIZE - GRID_SIZE), 2)
            pygame.draw.line(screen, (0,0,0),
                             (GRID_SIZE, i*GRID_SIZE),
                             (SCREEN_SIZE - GRID_SIZE, i*GRID_SIZE), 2)
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                cell = g.board[x,y]
                cx = (y+1)*GRID_SIZE
                cy = (x+1)*GRID_SIZE
                if cell == 1:
                    pygame.draw.circle(screen, (0,0,0), (cx,cy), GRID_SIZE//2 - 5)
                elif cell == 2:
                    pygame.draw.circle(screen, (200,200,200), (cx,cy), GRID_SIZE//2 - 5)
        pygame.display.flip()

    while True:
        if game.is_game_over():
            w = game.get_winner()
            if w == 0:
                print("Game over! It's a tie.")
            else:
                print(f"Game over! Winner = Player {w}")
            pygame.quit()
            sys.exit()

        draw_board(screen, game)
        clock.tick(15)

        # Player 1 = human
        if game.current_player == 1:
            x,y = None,None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    px, py = event.pos
                    grid_x = round((py/GRID_SIZE) - 1)
                    grid_y = round((px/GRID_SIZE) - 1)
                    if 0 <= grid_x < BOARD_SIZE and 0 <= grid_y < BOARD_SIZE:
                        x,y = grid_x, grid_y
            if x is not None:
                if game.is_valid_move(x,y):
                    game.make_move(x,y)
        else:
            # Player 2 = AI
            print("AI thinking...")
            root = NN_MCTS_Node(game.copy())
            nn_batch_evaluate([root], net)
            mcts_expand(root)
            pi_dict = mcts_run(root, net, n_simulations=50, batch_size=8)
            if len(pi_dict) == 0:
                print("AI has no moves, passing?")
                game.make_move(-1, -1)
                break
            best_move = max(pi_dict.items(), key=lambda elem: elem[1])[0]
            if best_move == (-1, -1):
                print("AI passes!")
            game.make_move(*best_move)

def main():
    # 1) Create or load network
    net = PolicyValueNet(board_size=BOARD_SIZE).to(device)
    if os.path.isfile("policy_value_net.pth"):
        print("Loading existing model parameters...")
        logging.info("Loading existing model parameters from policy_value_net.pth")
        net.load_state_dict(torch.load("policy_value_net.pth", map_location=device))
    else:
        print("No existing model found. Starting from scratch.")
        logging.info("No existing model found. Starting a fresh model.")
    net.eval()

    # 2) Self-play / 3) Train 
    N_ITER = 9            # number of self-play iterations
    N_GAMES_PER_ITER = 9  # how many self-play games each iteration
    N_MCTS_SIMS = 500     # MCTS simulations
    TRAIN_SAMPLE_SIZE = 1000

    total_start_time = time.time()

    for iteration in range(N_ITER):
        iteration_start_time = time.time()
        print(f"==== Self-Play iteration {iteration+1} ====")
        logging.info(f"==== Self-Play iteration {iteration+1} ====")

        # Gather data from self-play
        iteration_data = []
        for g in range(N_GAMES_PER_ITER):
            game_start_time = time.time()
            print(f"  Self-play game {g+1}...")
            logging.info(f"  Starting self-play game {g+1} in iteration {iteration+1}...")

            game_data = self_play_one_game(net,
                                           board_size=BOARD_SIZE,
                                           n_mcts_sims=N_MCTS_SIMS,
                                           temp=1.0)
            game_data = get_equi_data(game_data)
            iteration_data.extend(game_data)

            game_end_time = time.time()
            game_duration = game_end_time - game_start_time
            print(f"  Finished game {g+1}, collected {len(game_data)} states in {game_duration:.2f} sec.")
            logging.info(f"  Finished game {g+1}, collected {len(game_data)} states in {game_duration:.2f} sec.")

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
            train_policy_value_net(net, train_data, batch_size=32, epochs=5, lr=1e-3)
            train_end_time = time.time()
            logging.info(f"Training completed in {train_end_time - train_start_time:.2f} seconds.")
        else:
            print("No training data available yet.")

        # 4) Save the net (overwrites old file)
        torch.save(net.state_dict(), "policy_value_net.pth")
        print("Model saved.\n")
        logging.info(f"Model saved at iteration {iteration+1}.")

        iteration_end_time = time.time()
        iteration_duration = iteration_end_time - iteration_start_time
        logging.info(f"Iteration {iteration+1} duration: {iteration_duration:.2f} seconds.")

    total_end_time = time.time()
    total_selfplay_time = total_end_time - total_start_time
    logging.info(f"Total self-play time across all iterations: {total_selfplay_time:.2f} seconds.")

    # 5) Launch an interactive game with the trained net
    run_interactive_game(net)

if __name__ == "__main__":
    main()
