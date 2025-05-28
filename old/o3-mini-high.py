import random
import numpy as np
import math
import copy
import multiprocessing
import time
import datetime
import pygame
import sys

BOARD_SIZE = 9
ITERATIONS = 1000
PROCESSES_NUM = 15

GRID_SIZE = 80  # Adjust based on screen size
SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE 

# ----------------------- Go Game Class -----------------------
class GoGame:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 1  # Player 1 starts
        self.ko_point = None

    def is_valid_move(self, move):
        x, y = move
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == ' '

    def is_suicide_move(self, move):
        x, y = move
        if not self.is_valid_move(move):
            return False
        test_board = copy.deepcopy(self.board)
        test_board[x][y] = self.current_player
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and test_board[nx][ny] == ' ':
                return False
        return not self.has_liberties((x, y), test_board)

    def has_liberties(self, point, board):
        x, y = point
        color = board[x][y]
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        def dfs(point):
            x, y = point
            visited[x][y] = True
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                    if board[nx][ny] == ' ':
                        return True
                    elif not visited[nx][ny] and board[nx][ny] == color:
                        if dfs((nx, ny)):
                            return True
            return False
        return dfs(point)

    def remove_captured_stones(self):
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x][y] != ' ':
                    group = self.find_group((x, y))
                    if not self.has_liberties_in_group(group):
                        for stone in group:
                            self.board[stone[0]][stone[1]] = ' '

    def find_group(self, start):
        color = self.board[start[0]][start[1]]
        group = set()
        stack = [start]
        while stack:
            x, y = stack.pop()
            group.add((x, y))
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and (nx, ny) not in group:
                    if self.board[nx][ny] == color:
                        stack.append((nx, ny))
        return group

    def has_liberties_in_group(self, group):
        for stone in group:
            x, y = stone
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][ny] == ' ':
                    return True
        return False

    def make_move(self, x, y):
        move = (x, y)
        if self.is_valid_move(move) and not self.is_suicide_move(move):
            self.board[x][y] = self.current_player
            self.ko_point = None  # For simplicity, we do not handle ko specially here
            self.remove_captured_stones()
            self.switch_player()
            return True
        return False

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def get_legal_moves(self):
        legal_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.is_valid_move((x, y)) and not self.is_suicide_move((x, y)):
                    legal_moves.append((x, y))
        return legal_moves

    def is_game_over(self):
        return len(self.get_legal_moves()) == 0

    def get_winner(self):
        if not self.is_game_over():
            return None
        player1_stones = sum(row.count(1) for row in self.board)
        player2_stones = sum(row.count(2) for row in self.board)
        if player1_stones > player2_stones:
            return -1  # Player 1 wins
        elif player2_stones > player1_stones:
            return 1   # Player 2 wins (AI)
        else:
            return 0   # Draw

    def get_state(self):
        return copy.deepcopy(self.board)

    def render(self):
        for row in self.board:
            print(' '.join(map(str, row)))

# ----------------------- MCTS Tree Node -----------------------
class MCTSTreeNode:
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.total_value = 0
        self.state = None
        self.unvisited_moves = None  # Will hold legal moves not yet expanded

    def is_fully_expanded(self):
        return self.unvisited_moves is not None and len(self.unvisited_moves) == 0

# UCB1 function for selection (using a given exploration constant)
def ucb1_value(node, exploration_weight=1.0):
    if node.visit_count == 0:
        return math.inf
    exploitation = node.total_value / node.visit_count
    exploration = exploration_weight * math.sqrt(math.log(node.parent.visit_count) / node.visit_count)
    return exploitation + exploration

# ----------------------- MCTS: Selection & Expansion -----------------------
def tree_policy(node):
    """
    Traverse the tree (selection) until we reach a node that is either terminal
    or has some unvisited moves. Then expand one move.
    """
    while not node.state.is_game_over():
        # Initialize the node’s legal moves list if needed.
        if node.unvisited_moves is None:
            node.unvisited_moves = node.state.get_legal_moves()
        if node.unvisited_moves:
            return expand(node)
        else:
            node = best_child(node)
    return node

def best_child(node, exploration_weight=1.0):
    return max(node.children, key=lambda child: ucb1_value(child, exploration_weight))

def expand(node):
    move = random.choice(node.unvisited_moves)
    node.unvisited_moves.remove(move)
    new_state = copy.deepcopy(node.state)
    new_state.make_move(*move)
    child = MCTSTreeNode(parent=node, move=move)
    child.state = new_state
    child.unvisited_moves = new_state.get_legal_moves()
    node.children.append(child)
    return child

# ----------------------- Simulation & Backpropagation -----------------------
def simulate(node):
    """
    Play a random playout starting from the given node’s state until the game ends.
    """
    state = copy.deepcopy(node.state)
    while not state.is_game_over():
        legal_moves = state.get_legal_moves()
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        state.make_move(*move)
    # get_winner returns: -1 for player1 win, 1 for player2 win, 0 for tie.
    # For MCTS in a two-player zero-sum game, the simulation result is taken from the
    # perspective of the player who just moved.
    result = state.get_winner()
    return result if result is not None else 0

def backpropagate(node, result):
    """
    Propagate the simulation result up the tree. Note that we flip the result
    at each step so that each node’s value is from the perspective of the player
    who made the move at that node.
    """
    while node is not None:
        node.visit_count += 1
        node.total_value += result
        result = -result
        node = node.parent

# ----------------------- MCTS Worker for Parallel Execution -----------------------
def mcts_worker(root_state, iterations):
    """
    Run MCTS iterations from a copy of the current game state. Return aggregated
    statistics (visit counts and total values) for the moves from the root.
    """
    root = MCTSTreeNode()
    root.state = copy.deepcopy(root_state)
    root.unvisited_moves = root.state.get_legal_moves()

    for _ in range(iterations):
        leaf = tree_policy(root)
        simulation_result = simulate(leaf)
        backpropagate(leaf, simulation_result)

    # Aggregate statistics from the children of the root.
    move_stats = {}
    for child in root.children:
        if child.move in move_stats:
            move_stats[child.move][0] += child.visit_count
            move_stats[child.move][1] += child.total_value
        else:
            move_stats[child.move] = [child.visit_count, child.total_value]
    return move_stats

# ----------------------- AI Move Selection with Parallel MCTS -----------------------
def ai_play_parallel(board, iterations=ITERATIONS, num_processes=PROCESSES_NUM):
    """
    Use multiple processes to run independent MCTS simulations and aggregate the
    statistics for each candidate move.
    """
    root_state = copy.deepcopy(board)
    iterations_per_process = iterations // num_processes

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(mcts_worker, [(root_state, iterations_per_process) for _ in range(num_processes)])

    # Aggregate the statistics from each worker.
    aggregated_stats = {}
    for result in results:
        for move, stats in result.items():
            if move in aggregated_stats:
                aggregated_stats[move][0] += stats[0]
                aggregated_stats[move][1] += stats[1]
            else:
                aggregated_stats[move] = list(stats)

    # Fallback if no move was explored.
    legal_moves = board.get_legal_moves()
    if not aggregated_stats:
        return random.choice(legal_moves) if legal_moves else None

    # Choose the best move – here we choose the one with the highest visit count.
    best_move = max(aggregated_stats.items(), key=lambda item: item[1][0])[0]
    print("Aggregated Move Stats:")
    for move, (visits, value) in aggregated_stats.items():
        print(f"  Move: {move}  Visits: {visits}  Total Value: {value}")
    print("Best Move:", best_move)
    return best_move

# ----------------------- Rendering Helpers -----------------------
def render_game(board):
    out = ""
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == " ":
                out += ". "
                print(".", end=" ")
            elif board[i][j] == 1:
                out += "○ "
                print("○", end=" ")
            elif board[i][j] == 2:
                out += "● "
                print("●", end=" ")
        print()
        out += "\n"
    print("------------------------------------")
    out += "------------------------------------------------------------------------"
    return out

def draw_board(screen, game):
    screen.fill((0, 207, 207))
    for i in range(1, BOARD_SIZE + 1):
        pygame.draw.line(screen, (0, 0, 0), (i * GRID_SIZE, GRID_SIZE), (i * GRID_SIZE, SCREEN_SIZE - GRID_SIZE), 2)
        pygame.draw.line(screen, (0, 0, 0), (GRID_SIZE, i * GRID_SIZE), (SCREEN_SIZE - GRID_SIZE, i * GRID_SIZE), 2)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            x = (j + 1) * GRID_SIZE
            y = (i + 1) * GRID_SIZE
            if game.board[i][j] == 1:
                pygame.draw.circle(screen, (0, 0, 0), (x, y), GRID_SIZE // 2 - 10)
            elif game.board[i][j] == 2:
                pygame.draw.circle(screen, (194, 194, 194), (x, y), GRID_SIZE // 2 - 10)
    pygame.display.flip()

# ----------------------- Main Game Loop -----------------------
def main():
    date = datetime.datetime.now()
    current_time = date.strftime("%Y.%b.%d_%Hh%Mm")
    
    board_size = BOARD_SIZE
    game = GoGame(board_size)
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Go Game")
    
    while not game.is_game_over():
        px, py = 0, 0
        draw_board(screen, game)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                print("Player move:")
                px, py = event.pos
                print("Screen pos:", px, py)
                px = round(px / GRID_SIZE)
                py = round(py / GRID_SIZE)
        
        # Human player (Player 1) makes a move if a click occurred.
        if game.current_player == 1 and (px != 0 and py != 0):
            # Adjust for the grid offset.
            if game.make_move(py - 1, px - 1):
                print("Player moved.")
                render_game(game.get_state())
        # AI move (Player 2) using MCTS.
        elif game.current_player == 2:
            print("AI thinking...")
            start_time = time.time()
            ai_move = ai_play_parallel(game)
            if ai_move is not None:
                game.make_move(*ai_move)
            end_time = time.time()
            print("AI moved:", ai_move, "Time =", end_time - start_time)
            render_game(game.get_state())
    
    winner = game.get_winner()
    if winner == 0:
        print("It's a tie!")
    elif winner == -1:
        print("You win!")
    elif winner == 1:
        print("AI wins!")

if __name__ == "__main__":
    main()
