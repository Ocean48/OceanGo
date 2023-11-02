import random
import numpy as np
import math
import copy
import gym
from gym import spaces

# Define the board size as a constant
BOARD_SIZE = 3

# Define the GoGame class to represent the game board and rules


class GoGame:
    def __init__(self, board_size=BOARD_SIZE):
        # Initialize the game board and set the current player
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1

    def is_valid_move(self, x, y):
        # Check if a move is valid by verifying it's within the board and on an empty intersection
        if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == 0:
            return True
        return False

    def make_move(self, x, y):
        # Make a move by updating the board with the current player's stone and switching players
        if self.is_valid_move(x, y):
            self.board[x][y] = self.current_player
            self.current_player = 3 - self.current_player  # Switch players

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
        return np.sign(np.sum(self.board))

    def get_state(self):
        # Create a deep copy of the current board state
        return copy.deepcopy(self.board)

    def render(self):
        # Render the current board state
        for i in range(self.board_size):
            print(" ".join([str(self.board[i][j])
                  for j in range(self.board_size)]))
        print()
    # ... (include all the methods from the original GoGame class)

# Define the MCTSTreeNode class


class MCTSTreeNode:
    def __init__(self, parent=None, move=None):
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

    # ... (include all the methods from the original MCTSTreeNode class)

# Define the UCB1 value function for node selection


def ucb1_value(node, exploration_weight=1.0):
    if node.visit_count == 0:
        return math.inf
    exploitation = node.total_value / node.visit_count
    exploration = exploration_weight * \
        math.sqrt(math.log(node.parent.visit_count) / node.visit_count)
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
    return state.get_winner()

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
    for _ in range(iterations):
        node = select(root)
        expand(node)
        result = simulate(node)
        backpropagate(node, result)
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

# Define the Gym environment for the Go game


class GoEnv(gym.Env):
    def __init__(self, board_size=9):
        super(GoEnv, self).__init__()
        self.board_size = board_size
        self.action_space = spaces.Discrete(board_size**2)
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(board_size, board_size), dtype=np.int8)
        self.game = GoGame(board_size)

    def reset(self):
        self.game = GoGame(self.board_size)
        return self.game.get_state()

    def get_legal_moves(self):
        # Generate legal moves (empty intersections) on the current board
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.game.is_valid_move(i, j):
                    legal_moves.append((i, j))
        return legal_moves

    def step(self, action):
        x, y = action
        if self.game.is_valid_move(x, y):
            self.game.make_move(x, y)
        else:
            return self.game.get_state(), -1, True, {}

        if self.game.is_game_over():
            winner = self.game.get_winner()
            return self.game.get_state(), winner, True, {}

        return self.game.get_state(), 0, False, {}

    def render(self, mode='human'):
        render_game(self.game.get_state())


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


def main():
    board_size = 3  # Adjust the board size as needed
    env = GoEnv(board_size)

    while True:
        state = env.reset()
        done = False
        while not done:
            legal_moves = env.get_legal_moves()  # Get legal moves
            if len(legal_moves) == 0:
                break  # Handle the case where there are no legal moves
            action = random.choice(legal_moves)  # Randomly select a move
            state, reward, done, info = env.step(action)
            env.render()


if __name__ == "__main__":
    main()
