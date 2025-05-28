import numpy as np
import math
import copy
import random

BOARD_SIZE = 3

class GoGame:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = 1

    def is_valid_move(self, x, y):
        if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == 0:
            return True
        return False

    def make_move(self, x, y):
        if self.is_valid_move(x, y):
            self.board[x][y] = self.current_player
            self.current_player = 3 - self.current_player  # Switch players

    def get_legal_moves(self):
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.is_valid_move(i, j):
                    legal_moves.append((i, j))
        return legal_moves

    def is_game_over(self):
        return len(self.get_legal_moves()) == 0

    def get_winner(self):
        territory = np.zeros((self.board_size, self.board_size), dtype=int)

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    territory[i][j] = 0  # Empty intersection
                else:
                    territory[i][j] = self.board[i][j]  # Initialize with stone colors

        for i in range(self.board_size):
            for j in range(self.board_size):
                if territory[i][j] == 0:
                    group, liberty = self.find_group(i, j, territory)
                    if liberty == 0:
                        self.remove_group(group, territory)

        black_stones = np.sum(self.board == 1)
        white_stones = np.sum(self.board == 2)
        territory_points = np.sum(territory)

        if black_stones + territory_points > white_stones:
            return 1  # Player 1 (Black) wins
        elif black_stones + territory_points < white_stones:
            return 2  # Player 2 (White) wins
        else:
            return 0  # It's a tie

    def find_group(self, x, y, territory):
        color = territory[x][y]
        group = set()
        liberty = 0
        stack = [(x, y)]

        while stack:
            i, j = stack.pop()
            if territory[i][j] == 0:
                liberty += 1
            elif territory[i][j] == color and (i, j) not in group:
                group.add((i, j))
                stack.extend([(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)])

        return group, liberty

    def remove_group(self, group, territory):
        for i, j in group:
            territory[i][j] = 0

    def get_state(self):
        return np.copy(self.board)

    def render(self):
        for i in range(self.board_size):
            print(" ".join([str(self.board[i][j]) for j in range(self.board_size)]))
        print()

def ucb1_value(node, exploration_weight=1.0):
    if node.visit_count == 0:
        return math.inf
    exploitation = node.total_value / node.visit_count
    exploration = exploration_weight * math.sqrt(math.log(node.parent.visit_count) / node.visit_count)
    return exploitation + exploration

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
        return not self.unvisited_moves

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

def simulate(node):
    state = copy.deepcopy(node.state)
    while not state.is_game_over():
        legal_moves = state.get_legal_moves()
        move = random.choice(legal_moves)
        state.make_move(*move)
    return state.get_winner()

def backpropagate(node, result):
    while node is not None:
        node.visit_count += 1
        node.total_value += result
        node = node.parent

def mcts(root, iterations):
    for _ in range(iterations):
        node = select(root)
        expand(node)
        result = simulate(node)
        backpropagate(node, result)

def ai_play(board, iterations=1000):
    root = MCTSTreeNode()
    root.state = board
    root.unvisited_moves = board.get_legal_moves()
    mcts(root, iterations)
    best_move = max(root.children, key=lambda c: c.visit_count)
    return best_move.move

def main():
    board_size = BOARD_SIZE  # Change to your desired board size
    game = GoGame(board_size)

    while not game.is_game_over():
        game.render()
        if game.current_player == 1:
            x, y = map(int, input("Enter your move (x y): ").split())
            x -= 1  # Adjust the input by subtracting 1 from the row coordinate
            y -= 1  # Adjust the input by subtracting 1 from the column coordinate
            if game.is_valid_move(x, y):
                game.make_move(x, y)
        else:
            ai_move = ai_play(game)
            game.make_move(*ai_move)

    game.render()
    winner = game.get_winner()
    if winner == 0:
        print("It's a tie!")
    elif winner == 1:
        print("Player 1 (Black) wins!")
    elif winner == 2:
        print("Player 2 (White) wins!")

if __name__ == "__main__":
    main()
