import random
import numpy as np
import math
import copy
import time

# Define the board size as a constant
BOARD_SIZE = 3

# Define the GoGame class to represent the game board and rules
class GoGame:
    def __init__(self, board_size=BOARD_SIZE):
        # Initialize the game board and set the current player
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1
        self.ko = None  # Ko rule: Store the previous move's position

    def is_valid_move(self, x, y):
        # Check if a move is valid by verifying it's within the board and on an empty intersection
        if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == 0:
            # Ko rule: Check if the move is not at the same position as the previous move
            if (x, y) != self.ko:
                return True
        return False

    def make_move(self, x, y):
        # Check if the move is valid
        if not self.is_valid_move(x, y):
            print("Invalid move. Try again.")
            return

        # Make the move by updating the board with the current player's stone
        self.board[x][y] = self.current_player

        # Check for captured stones
        for i, j in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if 0 <= i < self.board_size and 0 <= j < self.board_size:
                if self.board[i][j] == 3 - self.current_player:
                    group = self.find_group(i, j)
                    if not self.has_liberties(group):
                        self.remove_stones(group)

        # Switch players
        self.current_player = 3 - self.current_player

    def find_group(self, x, y):
        group = set()
        visited = set()
        player = self.board[x][y]

        def dfs(i, j):
            if (i, j) in visited:
                return
            visited.add((i, j))
            if self.board[i][j] == player:
                group.add((i, j))
                for a, b in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                    if 0 <= a < self.board_size and 0 <= b < self.board_size:
                        dfs(a, b)

        dfs(x, y)
        return group

    def has_liberties(self, group):
        for x, y in group:
            for i, j in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if 0 <= i < self.board_size and 0 <= j < self.board_size and self.board[i][j] == 0:
                    return True
        return False

    def remove_stones(self, group):
        for x, y in group:
            self.board[x][y] = 0

    def get_legal_moves(self):
        # Get a list of legal moves (empty intersections) on the current board
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.is_valid_move(i, j):
                    legal_moves.append((i, j))
        return legal_moves

    def is_game_over(self):
        # Check if the game is over by examining the entire board
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    return False
        return True

    def get_winner(self):
        # Determine the winner based on the sum of stones on the board
        # return np.sign(np.sum(self.board))
        count1 = 0
        count2 = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 1:
                    count1 += 1
                elif self.board[i][j] == 2:
                    count2 += 1
        if (count1 > count2):
            return 1
        elif (count1 < count2):
            return 2
        else:
            return 0

    def get_state(self):
        # Create a deep copy of the current board state
        return copy.deepcopy(self.board)

# Define the MCTSTreeNode class to represent nodes in the Monte Carlo Tree Search
class MCTSTreeNode:
    def __init__(self, parent=None, move=None):
        # Initialize a node with a parent, move, children, and statistics
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.total_value = 0
        self.state = None
        self.unvisited_moves = None

    def is_fully_expanded(self):
        # Check if a node is fully expanded by examining unvisited moves
        return not self.unvisited_moves

# Define the UCB1 value function for node selection
def ucb1_value(node, exploration_weight=1.0):
    if node.visit_count == 0:
        return math.inf
    exploitation = node.total_value / node.visit_count
    exploration = exploration_weight * math.sqrt(math.log(node.parent.visit_count) / node.visit_count)
    return exploitation + exploration

# Define the node selection phase of MCTS
def select(node):
    while not node.is_fully_expanded():
        unvisited_moves = node.unvisited_moves
        move = random.choice(unvisited_moves)
        unvisited_moves.remove(move)
        new_state = copy.deepcopy(node.state)
        new_state.make_move(*move)
        child = MCTSTreeNode(parent=node, move=move)
        child.state = new_state
        node.children.append(child)
        return child
    return max(node.children, key=ucb1_value)

# Define the node expansion phase of MCTS
def expand(node):
    unvisited_moves = node.unvisited_moves
    if not unvisited_moves:
        return
    move = random.choice(unvisited_moves)
    unvisited_moves.remove(move)
    new_state = copy.deepcopy(node.state)
    new_state.make_move(*move)
    child = MCTSTreeNode(parent=node, move=move)
    child.state = new_state
    node.children.append(child)

# Define the simulation phase of MCTS
def simulate(node):
    state = copy.deepcopy(node.state)
    while not state.is_game_over():
        legal_moves = state.get_legal_moves()
        move = random.choice(legal_moves)
        state.make_move(*move)
    print(state.get_winner())
    return 1

# Define the backpropagation phase of MCTS
def backpropagate(node, result):
    while node is not None:
        node.visit_count += 1
        node.total_value += result
        node = node.parent

# Define the main MCTS function
def mcts(root, iterations):
    for _ in range(iterations):
        node = select(root)
        expand(node)
        result = simulate(node)
        backpropagate(node, result)

# Define the AI's move selection using MCTS
def ai_play(board, iterations=1000):
    root = MCTSTreeNode()
    root.state = board
    root.unvisited_moves = board.get_legal_moves()
    mcts(root, iterations)
    best_move = max(root.children, key=lambda c: c.visit_count)
    return best_move.move

# Define the rendering function for the game board
def render_game(board):
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                print(".", end=" ")  # An empty intersection
            elif board[i][j] == 1:
                print("O", end=" ")  # Player 1's stone
            elif board[i][j] == 2:
                print("X", end=" ")  # Player 2's stone
        print()
    print("------------------------------------")

# Define the self-play function
def self_play(board_size, iterations=1000):
    game = GoGame(board_size)

    while not game.is_game_over():
        render_game(game.get_state())
        ai_move = ai_play(game, iterations)
        game.make_move(*ai_move)

    render_game(game.get_state())
    winner = game.get_winner()
    
    print(winner)
    if winner == 1:
        print("Player 1 wins!")
    elif winner == 2:
        print("Player 2 wins!")
    elif winner == 0:
        print("It's a tie!")
    f = open("data.txt", "a")
    f.write(str(winner)+"\n")
    time.sleep(3)
    self_play(BOARD_SIZE)

if __name__ == "__main__":
    self_play(BOARD_SIZE)
