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

# ====================================================
#             Policy-Value Network
# ====================================================

class PolicyValueNet(nn.Module):
    """
    A small CNN-based policy/value network for Go or board games.
    Input:  (batch_size, 2, board_size, board_size)
    Output: (policy_logits, value)
      - policy_logits: shape = (batch_size, board_size*board_size + 1)
                       (the +1 can represent "pass" if you implement it)
      - value: shape = (batch_size, 1), in [-1, 1].
    """
    def __init__(self, board_size=BOARD_SIZE):
        super().__init__()
        self.board_size = board_size

        # A small ConvNet
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        # Policy head
        self.policy_conv = nn.Conv2d(32, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size,
                                   board_size*board_size+1)

        # Value head
        self.value_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size*board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch_size, 2, board_size, board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Policy
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        # Value
        v = F.relu(self.value_conv(x))
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

    def copy(self):
        g = GoGame(self.board_size)
        g.board = self.board.copy()
        g.current_player = self.current_player
        return g

    def get_state_np(self):
        return (self.board.copy(), self.current_player)

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def is_valid_move(self, x, y):
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return False
        return (self.board[x, y] == 0)

    def is_suicide(self, x, y):
        # place stone temporarily and check if new stone has liberty
        if not self.is_valid_move(x, y):
            return False
        temp = self.board.copy()
        temp[x, y] = self.current_player
        return not self._has_liberty((x, y), temp)

    def _has_liberty(self, start, temp_board):
        from collections import deque
        color = temp_board[start[0], start[1]]
        visited = set()
        q = deque([start])
        while q:
            cx, cy = q.pop()
            if (cx, cy) not in visited:
                visited.add((cx, cy))
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if temp_board[nx, ny] == 0:
                            return True
                        if temp_board[nx, ny] == color and (nx, ny) not in visited:
                            q.append((nx, ny))
        return False

    def make_move(self, x, y):
        if self.is_valid_move(x, y) and not self.is_suicide(x, y):
            self.board[x, y] = self.current_player
            self.remove_captured_stones(3 - self.current_player)
            self.switch_player()
            return True
        return False

    def remove_captured_stones(self, color):
        temp = self.board
        visited = set()
        captured_positions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if temp[i, j] == color and (i, j) not in visited:
                    group, has_lib = self._find_group((i, j))
                    visited |= group
                    if not has_lib:
                        captured_positions.extend(list(group))
        for (x, y) in captured_positions:
            self.board[x, y] = 0

    def _find_group(self, start):
        from collections import deque
        color = self.board[start[0], start[1]]
        group = set()
        has_lib = False
        q = deque([start])
        while q:
            cx, cy = q.pop()
            if (cx, cy) not in group:
                group.add((cx, cy))
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        if self.board[nx, ny] == 0:
                            has_lib = True
                        elif self.board[nx, ny] == color and (nx, ny) not in group:
                            q.append((nx, ny))
        return group, has_lib

    def get_legal_moves(self):
        moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y] == 0 and not self.is_suicide(x, y):
                    moves.append((x, y))
        return moves

    def is_game_over(self):
        return len(self.get_legal_moves()) == 0

    def get_winner(self):
        b_stones = np.count_nonzero(self.board == 1)
        w_stones = np.count_nonzero(self.board == 2)
        if b_stones > w_stones:
            return 1
        elif w_stones > b_stones:
            return 2
        return 0

# ====================================================
#  Helper: Convert boards to PyTorch Tensor
# ====================================================

def boards_to_tensor(batch_of_boards):
    """
    batch_of_boards: list of (board_np, current_player)
    Return: tensor of shape [batch_size, 2, board_size, board_size] on GPU
    """
    arr_list = []
    for (board_np, cplayer) in batch_of_boards:
        c0 = (board_np == cplayer).astype(np.float32)
        c1 = (board_np == (3 - cplayer)).astype(np.float32)
        stacked = np.stack([c0, c1], axis=0)  # shape (2, board_size, board_size)
        arr_list.append(stacked)
    # shape = (batch_size, 2, board_size, board_size)
    arr_np = np.stack(arr_list, axis=0)
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
        self.P = 0  # prior from net
        self.policy_map = {}  # (move) -> prior
        self.value = 0        # net value for the node state

def mcts_select(root: NN_MCTS_Node):
    """
    Descend the tree with PUCT formula until a node that isn't fully expanded is found.
    """
    node = root
    while True:
        if len(node.unexpanded_moves) > 0:
            return node
        if not node.children:
            return node
        # select child
        best_score = -999999
        best_child = None
        for mv, ch in node.children.items():
            c_puct = 1.0
            U = c_puct * ch.P * math.sqrt(node.N) / (1 + ch.N)
            score = ch.Q + U
            if score > best_score:
                best_score = score
                best_child = ch
        node = best_child

def nn_batch_evaluate(nodes, net):
    """
    Evaluate each node's state in a single GPU batch.
    For each node:
      - node.policy_map is filled with {move: prior}
      - node.value is set from net
    """
    batch = []
    for nd in nodes:
        board_np, cplayer = nd.game_state.get_state_np()
        batch.append((board_np, cplayer))

    input_tensor = boards_to_tensor(batch)
    with torch.no_grad():
        policy_logits, value_out = net(input_tensor)
    policy_logits = policy_logits.cpu().numpy()
    value_out = value_out.squeeze(1).cpu().numpy()

    for i, nd in enumerate(nodes):
        val = value_out[i]  # from net in [-1,1]
        nd.value = float(val)
        p_logits = policy_logits[i]
        legal_moves = nd.unexpanded_moves
        board_s = nd.game_state.board_size

        # extract logits for each move
        move_indices = [x*board_s + y for (x,y) in legal_moves]
        selected_logits = [p_logits[idx] for idx in move_indices]
        if len(selected_logits) > 0:
            probs = softmax(np.array(selected_logits))
        else:
            probs = []

        nd.policy_map = {}
        for mv, p in zip(legal_moves, probs):
            nd.policy_map[mv] = p

def mcts_expand(node: NN_MCTS_Node):
    """
    Expand all unexpanded moves for 'node'.
    Each child gets prior = node.policy_map[move].
    """
    for mv in node.unexpanded_moves:
        ng = node.game_state.copy()
        ng.make_move(*mv)
        child_node = NN_MCTS_Node(ng, parent=node, move=mv)
        child_node.P = node.policy_map.get(mv, 0.0)
        node.children[mv] = child_node
    node.unexpanded_moves = []

def mcts_backup(node: NN_MCTS_Node, value):
    """
    Backup the value up the tree, flipping sign for each parent's perspective.
    """
    while node is not None:
        node.N += 1
        node.W += value
        node.Q = node.W / node.N
        value = -value
        node = node.parent

def mcts_run(root: NN_MCTS_Node, net, n_simulations=50, temp=1.0):
    """
    Run MCTS from root node for n_simulations.
    Return a policy distribution over root's children.
    """
    for _ in range(n_simulations):
        leaf = mcts_select(root)
        # evaluate leaf
        nn_batch_evaluate([leaf], net)
        # expand
        mcts_expand(leaf)
        # backup
        mcts_backup(leaf, leaf.value)
        
        if _ % 1000 == 0:
            print(f"Simulations: {_+1}/{n_simulations}, N={root.N}")

    # build policy vector for root moves
    move_N = [(mv, ch.N) for mv, ch in root.children.items()]
    if not move_N:
        # no children => no legal moves => pass or uniform
        return {}

    moves, visits = zip(*move_N)
    visits = np.array(visits, dtype=np.float32)
    if temp < 1e-3:
        # pick argmax if temp=0
        best_idx = np.argmax(visits)
        probs = np.zeros_like(visits)
        probs[best_idx] = 1.0
    else:
        # softmax with temperature
        v = visits ** (1.0 / temp)
        probs = v / np.sum(v)

    policy_dict = dict(zip(moves, probs))
    return policy_dict

# ====================================================
#        Self-Play: Generating Training Data
# ====================================================

def self_play_one_game(net, board_size=BOARD_SIZE, n_mcts_sims=50, temp=1.0):
    """
    Play one game with two MCTS players using the same net.
    Return a list of (board_state, current_player, pi_flat, z).
    """
    game = GoGame(board_size)
    data = []  # store (board, player, pi, z)

    while not game.is_game_over():
        root = NN_MCTS_Node(game.copy())
        # Evaluate & expand root
        nn_batch_evaluate([root], net)
        mcts_expand(root)

        # Run MCTS
        pi_dict = mcts_run(root, net, n_mcts_sims, temp=temp)
        # Flatten pi_dict into a vector of length board_size*board_size
        pi_flat = np.zeros(board_size * board_size, dtype=np.float32)
        for mv, p in pi_dict.items():
            idx = mv[0]*board_size + mv[1]
            pi_flat[idx] = p

        # Sample a move from pi_dict (or pick max) for training exploration
        if len(pi_dict) > 0:
            moves, probs = zip(*pi_dict.items())
            selected_move = random.choices(moves, weights=probs, k=1)[0]
        else:
            # no legal moves => break
            break

        # Record data
        board_cp = game.board.copy()
        current_player = game.current_player
        data.append((board_cp, current_player, pi_flat))

        # Make the move
        game.make_move(*selected_move)

    # At game end, assign outcome
    winner = game.get_winner()  # 1 or 2 or 0
    for i, (board_cp, player, pi_flat) in enumerate(data):
        if winner == 0:
            z = 0
        elif winner == player:
            z = 1
        else:
            z = -1
        data[i] = (board_cp, player, pi_flat, z)
    return data

# ====================================================
#          Training the Network
# ====================================================

def train_policy_value_net(net, data, batch_size=32, epochs=5, lr=1e-3):
    """
    data: list of (board_np, current_player, pi_flat, z)
    We'll do:
      - policy loss (cross entropy w.r.t. pi_flat)
      - value loss (MSE vs. z)
    """
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()

    X_boards = []
    policy_targets = []
    value_targets = []
    board_size = net.board_size

    for (board_np, cplayer, pi_flat, z) in data:
        X_boards.append((board_np, cplayer))
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

            batch_boards = [X_boards[i] for i in excerpt]
            inp = boards_to_tensor(batch_boards)  # (B, 2, board_size, board_size)
            tgt_p = torch.from_numpy(policy_targets[excerpt]).to(device) # (B, board_size^2)
            tgt_v = torch.from_numpy(value_targets[excerpt]).to(device)  # (B,1)

            optimizer.zero_grad()
            out_p, out_v = net(inp)  
            # out_p: (B, board_size^2+1), out_v: (B,1)
            # We'll ignore the last "pass" dimension for now:
            out_p = out_p[:, :board_size*board_size]

            # Policy loss (cross-entropy)
            logp = F.log_softmax(out_p, dim=1)  # (B, board_size^2)
            policy_loss = -torch.sum(tgt_p * logp, dim=1).mean()

            # Value loss (MSE)
            value_loss = F.mse_loss(out_v, tgt_v)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        print(f"Epoch {ep+1}/{epochs}, Loss={np.mean(batch_losses):.4f}")
    net.eval()

# ====================================================
#             Putting it All Together
# ====================================================

def run_interactive_game(net):
    """
    Let Player1 be human, Player2 be the trained net (via MCTS).
    """
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
                game.make_move(x,y)
        else:
            # Player 2 = AI
            print("AI thinking...")
            root = NN_MCTS_Node(game.copy())
            # Evaluate & expand root
            nn_batch_evaluate([root], net)
            mcts_expand(root)
            # Run MCTS
            pi_dict = mcts_run(root, net, n_simulations=50)
            if len(pi_dict) == 0:
                print("AI has no moves, passing?")
                break
            # pick move with max prob
            best_move = max(pi_dict.items(), key=lambda x: x[1])[0]
            game.make_move(*best_move)

def main():
    """
    Main loop:
     1) Load an existing model if found, or create a fresh one.
     2) Self-play a few games, collect data
     3) Train the net on that data
     4) Save the net
     5) Optionally launch an interactive game.
    """
    # -------------------------------------------------------
    # 1) Create or load network
    # -------------------------------------------------------
    net = PolicyValueNet(board_size=BOARD_SIZE).to(device)
    if os.path.isfile("policy_value_net.pth"):
        print("Loading existing model parameters...")
        net.load_state_dict(torch.load("policy_value_net.pth", map_location=device))
    else:
        print("No existing model found. Starting from scratch.")
    net.eval()

    # -------------------------------------------------------
    # 2) Self-play / 3) Train 
    # -------------------------------------------------------
    N_ITER = 10          # number of self-play iterations
    N_GAMES_PER_ITER = 10  # how many self-play games each iteration
    N_MCTS_SIMS = 10000      # MCTS simulations
    all_data = []

    for iteration in range(N_ITER):
        print(f"==== Self-Play iteration {iteration+1} ====")

        # Gather data from a few self-play games
        iteration_data = []
        for g in range(N_GAMES_PER_ITER):
            print(f"  Self-play game {g+1}...")
            game_data = self_play_one_game(net, 
                                           board_size=BOARD_SIZE, 
                                           n_mcts_sims=N_MCTS_SIMS, 
                                           temp=1.0)
            iteration_data.extend(game_data)
            print(f"  Finished game {g+1}, collected {len(game_data)} states.")
        print(f"Total new data: {len(iteration_data)} states.")

        # Train the network on new data
        # (Optionally combine with old data in a replay buffer.)
        if len(iteration_data) > 0:
            print("Training on new data...")
            train_policy_value_net(net, iteration_data, batch_size=32, epochs=5, lr=1e-3)

        # Save the net after this iteration
        torch.save(net.state_dict(), "policy_value_net.pth")
        print("Model saved.\n")

    # -------------------------------------------------------
    # 4) Launch an interactive game with the trained net
    # -------------------------------------------------------
    run_interactive_game(net)

if __name__ == "__main__":
    main()
