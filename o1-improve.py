import math
import copy
import random
import multiprocessing
import time
import datetime
import pygame
import sys

############################################################
#                      CONFIG                              #
############################################################

BOARD_SIZE = 9
GRID_SIZE = 80
SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE

# Number of CPU cores to use:
NUM_PROCESSES = 14

# MCTS settings:
MCTS_MODE = "time"        # "iterations" or "time"
MCTS_ITERATIONS = 1000    # total MCTS playouts (if MCTS_MODE="iterations")
MCTS_TIME_LIMIT = 1.0     # time in seconds (if MCTS_MODE="time")

KOMI = 6.5                # Komi for White
USE_TRANSPOSITION_TABLE = True

# Transposition Table settings
TT_MAX_SIZE = 800_000     # maximum entries in TT (random eviction when full)

############################################################
#                 ZOBRIST HASHING SETUP                    #
############################################################

random.seed(42)  # For reproducibility (optional)

ZOBRIST_TABLE = [
    [
        [random.getrandbits(64) for _ in range(3)]  # index 0..2 => color
        for _ in range(BOARD_SIZE)
    ]
    for _ in range(BOARD_SIZE)
]

def compute_zobrist_hash(board):
    h = 0
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            c = board[x][y]
            if c != 0:
                h ^= ZOBRIST_TABLE[x][y][c]
    return h

############################################################
#                      GO GAME                             #
############################################################

class GoGame:
    """
    A simplified Go game (9x9):
      - board[x][y] in {0=empty,1=black,2=white}
      - pass => two consecutive passes => game over
      - area scoring + small komi
    """
    def __init__(self, board_size=BOARD_SIZE, komi=KOMI):
        self.board_size = board_size
        self.board = [[0]*board_size for _ in range(board_size)]
        self.current_player = 1  # black=1, white=2
        self.consecutive_passes = 0
        self.komi = komi

    def copy_game(self):
        new_g = GoGame(self.board_size, self.komi)
        new_g.board = copy.deepcopy(self.board)
        new_g.current_player = self.current_player
        new_g.consecutive_passes = self.consecutive_passes
        return new_g

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def pass_move(self):
        self.consecutive_passes += 1
        self.switch_player()

    def make_move(self, x, y):
        if x is None and y is None:
            self.pass_move()
            return True

        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return False
        if self.board[x][y] != 0:
            return False

        self.board[x][y] = self.current_player
        self.consecutive_passes = 0
        self.remove_captured_stones(x, y)
        self.switch_player()
        return True

    def remove_captured_stones(self, x, y):
        placed_color = self.board[x][y]
        opp_color = 3 - placed_color
        captured = self.find_captured_stones(opp_color)
        for (cx, cy) in captured:
            self.board[cx][cy] = 0

    def find_captured_stones(self, color):
        captured = set()
        visited = set()
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == color and (i, j) not in visited:
                    group, has_lib = self.find_group_and_liberty(i, j)
                    visited |= group
                    if not has_lib:
                        captured |= group
        return captured

    def find_group_and_liberty(self, x, y):
        color = self.board[x][y]
        visited = set()
        stack = [(x, y)]
        has_liberty = False
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            for nx, ny in [(cx-1,cy), (cx+1,cy), (cx,cy-1), (cx,cy+1)]:
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board[nx][ny] == 0:
                        has_liberty = True
                    elif self.board[nx][ny] == color and (nx, ny) not in visited:
                        stack.append((nx, ny))
        return visited, has_liberty

    def is_suicide_move(self, x, y, color):
        if not (0 <= x < self.board_size and 0 <= y < self.board_size):
            return False
        if self.board[x][y] != 0:
            return False

        self.board[x][y] = color
        if self.has_liberty(x, y):
            self.board[x][y] = 0
            return False

        # check if captures
        opp_color = 3 - color
        captured = self.find_captured_stones(opp_color)
        self.board[x][y] = 0
        if captured:
            return False
        return True

    def has_liberty(self, x, y):
        color = self.board[x][y]
        visited = set()
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            for nx, ny in [(cx-1,cy),(cx+1,cy),(cx,cy-1),(cx,cy+1)]:
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if self.board[nx][ny] == 0:
                        return True
                    if self.board[nx][ny] == color and (nx, ny) not in visited:
                        stack.append((nx, ny))
        return False

    def get_legal_moves(self):
        moves = [(None, None)]
        col = self.current_player
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x][y] == 0:
                    if not self.is_suicide_move(x, y, col):
                        moves.append((x, y))
        return moves

    def is_game_over(self):
        return self.consecutive_passes >= 2

    def get_winner(self):
        if not self.is_game_over():
            return None
        black_stones = 0
        white_stones = 0
        for row in self.board:
            black_stones += row.count(1)
            white_stones += row.count(2)
        black_area, white_area = self.area_score()
        black_score = black_stones + black_area
        white_score = white_stones + white_area + self.komi
        if black_score > white_score:
            return 1
        elif white_score > black_score:
            return 2
        else:
            return 0

    def area_score(self):
        visited = [[False]*self.board_size for _ in range(self.board_size)]
        black_area = 0
        white_area = 0
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x][y] == 0 and not visited[x][y]:
                    stack = [(x,y)]
                    visited[x][y] = True
                    region = []
                    boundary_colors = set()
                    while stack:
                        cx, cy = stack.pop()
                        region.append((cx, cy))
                        for nx, ny in [(cx-1,cy),(cx+1,cy),(cx,cy-1),(cx,cy+1)]:
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                                if self.board[nx][ny] == 0 and not visited[nx][ny]:
                                    visited[nx][ny] = True
                                    stack.append((nx, ny))
                                elif self.board[nx][ny] in (1,2):
                                    boundary_colors.add(self.board[nx][ny])
                    if len(boundary_colors) == 1:
                        c = boundary_colors.pop()
                        if c == 1:
                            black_area += len(region)
                        else:
                            white_area += len(region)
        return black_area, white_area

############################################################
#        BOUNDED TRANSPOSITION TABLE (RANDOM EVICTION)     #
############################################################

def store_in_tt(tt, key, node):
    """
    Insert (key -> node) in the TT, evicting a random entry if TT is full.
    """
    if len(tt) >= TT_MAX_SIZE:
        # random eviction
        random_key = random.choice(list(tt.keys()))
        del tt[random_key]
    tt[key] = node

############################################################
#               MCTS NODE & HELPER FUNCTIONS               #
############################################################

class MCTSTreeNode:
    """
    Node in the MCTS tree:
      - parent
      - move
      - state
      - children
      - unvisited_moves
      - visit_count
      - total_value (white perspective)
      - zobrist_hash
    """
    def __init__(self, parent=None, move=None, state=None, zobrist_hash=0):
        self.parent = parent
        self.move = move
        self.state = state
        self.children = []
        self.unvisited_moves = []
        self.visit_count = 0
        self.total_value = 0.0
        self.zobrist_hash = zobrist_hash

    def is_fully_expanded(self):
        return len(self.unvisited_moves) == 0

def best_child(node, c=1.0):
    best_val = float('-inf')
    best_node = None
    for child in node.children:
        if child.visit_count == 0:
            return child
        exploitation = child.total_value / child.visit_count
        exploration = c * math.sqrt(math.log(node.visit_count) / child.visit_count)
        ucb = exploitation + exploration
        if ucb > best_val:
            best_val = ucb
            best_node = child
    return best_node

def tree_policy(node, transposition_table):
    """Selection: descend until leaf or not expanded or game over."""
    while not node.state.is_game_over():
        if not node.is_fully_expanded():
            return expand(node, transposition_table)
        else:
            node = best_child(node)
    return node

def expand(node, transposition_table):
    move = node.unvisited_moves.pop()
    new_state = node.state.copy_game()
    new_state.make_move(move[0], move[1])
    z_hash = compute_zobrist_hash(new_state.board)

    # transposition table check
    if USE_TRANSPOSITION_TABLE and z_hash in transposition_table:
        child = copy.copy(transposition_table[z_hash])  # shallow copy to re-parent
        child.parent = node
    else:
        child = MCTSTreeNode(parent=node, move=move, state=new_state, zobrist_hash=z_hash)
        child.unvisited_moves = new_state.get_legal_moves()
        if USE_TRANSPOSITION_TABLE:
            store_in_tt(transposition_table, z_hash, child)

    node.children.append(child)
    return child

def heuristic_rollout(state):
    """
    Simple capturing-heuristic rollout:
      - if any capturing moves exist, pick from them
      - else random
    """
    sim_game = state.copy_game()
    while not sim_game.is_game_over():
        moves = sim_game.get_legal_moves()
        capturing_moves = []
        for mv in moves:
            if mv == (None, None):
                continue
            x,y = mv
            c = sim_game.current_player
            if sim_game.board[x][y] != 0:
                continue
            sim_game.board[x][y] = c
            opp = 3 - c
            captured = sim_game.find_captured_stones(opp)
            sim_game.board[x][y] = 0
            if captured:
                capturing_moves.append(mv)
        if capturing_moves:
            choice = random.choice(capturing_moves)
        else:
            choice = random.choice(moves)
        sim_game.make_move(choice[0], choice[1])

    w = sim_game.get_winner()
    if w == 2:
        return 1.0
    elif w == 1:
        return -1.0
    else:
        return 0.0

def backup(node, result):
    while node is not None:
        node.visit_count += 1
        node.total_value += result
        node = node.parent

############################################################
#        MCTS: TIME-BASED VS FIXED-ITERATION MODES         #
############################################################

def run_mcts_iterations(root, iterations=1000):
    """
    Run MCTS for 'iterations' playouts (fixed iteration mode).
    """
    transposition_table = {}
    if USE_TRANSPOSITION_TABLE:
        transposition_table[root.zobrist_hash] = root

    for _ in range(iterations):
        leaf = tree_policy(root, transposition_table)
        outcome = heuristic_rollout(leaf.state)
        backup(leaf, outcome)

def run_mcts_time(root, time_limit=5.0):
    """
    Run MCTS until 'time_limit' seconds have elapsed (time-based mode).
    """
    start = time.time()
    transposition_table = {}
    if USE_TRANSPOSITION_TABLE:
        transposition_table[root.zobrist_hash] = root

    while True:
        now = time.time()
        if now - start > time_limit:
            break
        leaf = tree_policy(root, transposition_table)
        outcome = heuristic_rollout(leaf.state)
        backup(leaf, outcome)

############################################################
#         PARALLEL MCTS WORKERS (TIME or ITERATIONS)       #
############################################################

def mcts_worker(board, iterations, time_limit, use_time_mode, return_dict, pid):
    """
    Unified worker:
      - If use_time_mode=True => run_mcts_time
      - Else => run_mcts_iterations
    Then return stats for root.children
    """
    root_game = board.copy_game()
    root_hash = compute_zobrist_hash(root_game.board)
    root_node = MCTSTreeNode(None, None, root_game, zobrist_hash=root_hash)
    root_node.unvisited_moves = root_game.get_legal_moves()

    if use_time_mode:
        # time-based mode
        run_mcts_time(root_node, time_limit)
    else:
        # iteration-based mode
        run_mcts_iterations(root_node, iterations)

    child_stats = {}
    for child in root_node.children:
        child_stats[child.move] = (child.visit_count, child.total_value)
    return_dict[pid] = child_stats

def ai_play_parallel(board,
                     mcts_mode=MCTS_MODE,
                     total_iterations=MCTS_ITERATIONS,
                     time_limit=MCTS_TIME_LIMIT,
                     num_processes=NUM_PROCESSES):
    """
    Main parallel MCTS entry point:
      - If mcts_mode == "time", each process runs MCTS for time_limit/num_processes (approx total = time_limit)
      - If mcts_mode == "iterations", each runs total_iterations//num_processes
    Collect children stats => pick best move
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []

    if mcts_mode == "time":
        time_per_proc = time_limit / num_processes
        for pid in range(num_processes):
            p = multiprocessing.Process(
                target=mcts_worker,
                args=(board, None, time_per_proc, True, return_dict, pid)
            )
            processes.append(p)
            p.start()
    else:
        # "iterations"
        iters_per_proc = total_iterations // num_processes
        for pid in range(num_processes):
            p = multiprocessing.Process(
                target=mcts_worker,
                args=(board, iters_per_proc, None, False, return_dict, pid)
            )
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    # Merge from each worker
    aggregated = {}
    for pid, stats in return_dict.items():
        for move, (vc, tv) in stats.items():
            if move not in aggregated:
                aggregated[move] = [0, 0.0]
            aggregated[move][0] += vc
            aggregated[move][1] += tv

    best_move = None
    best_vc = -1
    best_tv = 0.0
    for m, (vc, tv) in aggregated.items():
        if vc > best_vc:
            best_vc = vc
            best_tv = tv
            best_move = m

    print(f"AI (White) picks {best_move}, visits={best_vc}, totalVal={best_tv:.2f}")
    return best_move

############################################################
#                   PYGAME UI LOOP                         #
############################################################

def draw_board(screen, game):
    screen.fill((220, 179, 92))

    for i in range(1, game.board_size + 1):
        pygame.draw.line(screen, (0,0,0), (i*GRID_SIZE, GRID_SIZE), (i*GRID_SIZE, SCREEN_SIZE - GRID_SIZE), 2)
        pygame.draw.line(screen, (0,0,0), (GRID_SIZE, i*GRID_SIZE), (SCREEN_SIZE - GRID_SIZE, i*GRID_SIZE), 2)

    # draw stones
    for x in range(game.board_size):
        for y in range(game.board_size):
            val = game.board[x][y]
            if val != 0:
                cx = (y+1)*GRID_SIZE
                cy = (x+1)*GRID_SIZE
                if val == 1:
                    pygame.draw.circle(screen, (0,0,0), (cx, cy), GRID_SIZE//2 - 5)
                else:
                    pygame.draw.circle(screen, (255,255,255), (cx, cy), GRID_SIZE//2 - 5)
    pygame.display.flip()

def get_click_coord(pos):
    px, py = pos
    col = round(px/GRID_SIZE) - 1
    row = round(py/GRID_SIZE) - 1
    if row < 0 or col < 0:
        return None, None
    return row, col

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Go with Bounded TT + Parallel MCTS (Iterations/Time)")

    game = GoGame(board_size=BOARD_SIZE, komi=KOMI)

    running = True
    draw_board(screen, game)

    while running:
        if game.is_game_over():
            winner = game.get_winner()
            if winner == 1:
                print("Game Over: Black wins.")
            elif winner == 2:
                print("Game Over: White wins.")
            else:
                print("Game Over: Tie.")
            break

        if game.current_player == 1:
            # Human (Black)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        print("Black passes.")
                        game.make_move(None, None)
                        draw_board(screen, game)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    gx, gy = get_click_coord(pos)
                    if gx is not None and 0 <= gx < BOARD_SIZE and 0 <= gy < BOARD_SIZE:
                        success = game.make_move(gx, gy)
                        if success:
                            draw_board(screen, game)
        else:
            # AI (White)
            if MCTS_MODE == "time":
                print(f"AI (White) searching up to {MCTS_TIME_LIMIT} sec with bounded TT.")
            else:
                print(f"AI (White) searching {MCTS_ITERATIONS} total iterations (bounded TT).")

            t0 = time.time()
            move = ai_play_parallel(game,
                                   mcts_mode=MCTS_MODE,
                                   total_iterations=MCTS_ITERATIONS,
                                   time_limit=MCTS_TIME_LIMIT,
                                   num_processes=NUM_PROCESSES)
            game.make_move(move[0], move[1])
            t1 = time.time()
            print(f"AI move took {t1 - t0:.2f} seconds.")
            draw_board(screen, game)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
