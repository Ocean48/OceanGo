import random
import numpy as np
import math
import copy
import time
import datetime
# import gym
# from gym import spaces


BOARD_SIZE = 6
ITERATIONS = 1000

# Define the GoGame class to represent the game board and rules
class GoGame:
    def __init__(self, board_size=BOARD_SIZE):
        # Initialize the game board and set the current player
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 1  # Player 1
        self.ko_point = None

    def is_valid_move(self, move):
        x, y = move
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == ' '

    def is_suicide_move(self, move):
        x, y = move
        if not self.is_valid_move(move):
            return False
        test_board = [row[:] for row in self.board]
        test_board[x][y] = self.current_player
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and test_board[nx][ny] == ' ':
                return False
        return not self._has_liberties((x, y), test_board)

    def _has_liberties(self, point, board):
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
                    color = self.board[x][y]
                    group = self._find_group((x, y))
                    if not self._has_liberties_in_group(group):
                        # Remove the captured stones
                        for stone in group:
                            self.board[stone[0]][stone[1]] = ' '

    def _find_group(self, start):
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

    def _has_liberties_in_group(self, group):
        for stone in group:
            x, y = stone
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][ny] == ' ':
                    return True
        return False

    def make_move(self, x, y):
        move = x, y
        if self.is_valid_move(move) and not self.is_suicide_move(move):
            self.board[x][y] = self.current_player
            if self.ko_point == move:
                self.ko_point = None
            else:
                self.ko_point = None
            self.remove_captured_stones()  # Remove captured stones after each move
            self._switch_player()
            return True
        return False


    def _switch_player(self):
        self.current_player = 3 - self.current_player  # Toggle between player 1 and player 2

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
            return 1  # Player 1 wins
        elif player2_stones > player1_stones:
            return 2  # Player 2 wins
        else:
            return 0  # Draw

    # Create a deep copy of the current board state
    def get_state(self):
        return copy.deepcopy(self.board)
    
    def render(self):
        for row in self.board:
            print(' '.join(map(str, row)))

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

    # Check if a node is fully expanded by examining unvisited moves
    def is_fully_expanded(self):
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
def ai_play(board, iterations=ITERATIONS):
    root = MCTSTreeNode()
    root.state = board
    root.unvisited_moves = board.get_legal_moves()
    mcts(root, iterations)
    best_move = max(root.children, key=lambda c: c.visit_count)
    return best_move.move

# Define the rendering function for the game board
def render_game(board):
    out = ""
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == " ":
                out += ". "
                print(".", end=" ")  # An empty intersection
            elif board[i][j] == 1:
                out += "● "
                print("●", end=" ")  # Player 1's stone  ○⚪◯
            elif board[i][j] == 2:
                out += "○ "
                print("○", end=" ")  # Player 2's stone   ●⚫⬤
            # ⬜⬛➕
        print()
        out += "\n"
    print("------------------------------------")
    out += "------------------------------------------------------------------------"
    return out

# Define the main game loop
def main():
    
    date = datetime.datetime.now()
    current_time = date.strftime("%Y.%b.%d_%Hh%Mm")
    file = open("data/single/"+current_time+".txt", "a", encoding='utf-8')
    file.write("Single:\n"+"Board size: "+str(BOARD_SIZE) + "\nIterations:"+str(ITERATIONS)+"\n")
    
    
    board_size = BOARD_SIZE
    game = GoGame(board_size)
    start = 0

    while not game.is_game_over():
        game_board_out = render_game(game.get_state())
        file.write(game_board_out+"\n")
        if game.current_player == 1:
            x, y = map(int, input("Enter your move (x y): ").split())
            x -= 1  # Adjust the input by subtracting 1 from the row coordinate
            y -= 1  # Adjust the input by subtracting 1 from the column coordinate
            start = time.time()
            game.make_move(x, y)
        else:
            start = time.time()
            ai_move = ai_play(game)
            game.make_move(*ai_move)
        end = time.time()
        print(3-game.current_player, " Time=", end - start)
        file.write(str(3-game.current_player)+ " Time="+ str(end - start)+"\n")

    game_board_out = render_game(game.get_state())
    file.write(game_board_out+"\n")
    winner = game.get_winner()
    if winner == 0:
        print("It's a tie!")
        file.write("Tie")
    elif winner == 1:
        print("You win!")
        file.write("You win")
    else:
        print("AI wins!")
        file.write("AI win")
        
    

if __name__ == "__main__":
    main()
