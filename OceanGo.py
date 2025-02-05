import random
import numpy as np
import math
import copy
import multiprocessing
import time
import datetime
import pygame
import sys

BOARD_SIZE = 13
ITERATIONS = 50000
PROCESSES_NUM = 18

GRID_SIZE = 80 # Adjust based on screen size
SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE 


# The `GoGame` class represents a game of Go, providing methods for making moves, checking move
# validity, capturing stones, determining game over conditions, and getting the winner.
class GoGame:
    def __init__(self, board_size=BOARD_SIZE):
        """
        Initialize a Go game instance.

        Args:
            board_size (int): The size of the game board.
        """
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 1  # Player 1
        self.ko_point = None

    def is_valid_move(self, move):
        """
        Check if a move is valid.

        Args:
            move (tuple): The coordinates of the move (x, y).

        Returns:
            bool: True if the move is valid, False otherwise.
        """
        x, y = move
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == ' '

    def is_suicide_move(self, move):
        """
        Check if a move leads to suicide.

        Args:
            move (tuple): The coordinates of the move (x, y).

        Returns:
            bool: True if the move is a suicide move, False otherwise.
        """
        x, y = move
        if not self.is_valid_move(move):
            return False
        # test_board = [row[:] for row in self.board]   # old version
        test_board = copy.deepcopy(self.board)
        test_board[x][y] = self.current_player
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and test_board[nx][ny] == ' ':
                return False
        return not self.has_liberties((x, y), test_board)

    def has_liberties(self, point, board):
        """
        Check if a group of stones has liberties.

        Args:
            point (tuple): The coordinates of the stone (x, y).
            board (list): The current game board.

        Returns:
            bool: True if the group has liberties, False otherwise.
        """
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
        """
        Remove captured stones from the board.
        """
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x][y] != ' ':
                    color = self.board[x][y]
                    group = self.find_group((x, y))
                    if not self.has_liberties_in_group(group):
                        # Remove the captured stones
                        for stone in group:
                            self.board[stone[0]][stone[1]] = ' '

    def find_group(self, start):
        """
        Find the group of stones connected to a given stone.

        Args:
            start (tuple): The coordinates of the starting stone (x, y).

        Returns:
            set: A set of stone coordinates representing the group.
        """
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
        """
        Check if a group of stones has liberties
        
        Args:
            group (set): The set of coordinates representing the group of stones.
        
        Returns:
            bool: True if the group has liberties, False otherwise.
        """
        for stone in group:
            x, y = stone
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx][ny] == ' ':
                    return True
        return False

    def make_move(self, x, y):
        """
        Make a move on the board.

        Args:
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.

        Returns:
            bool: True if the move is successful, False otherwise.
        """
        move = x, y
        if self.is_valid_move(move) and not self.is_suicide_move(move):
            self.board[x][y] = self.current_player
            if self.ko_point == move:
                self.ko_point = None
            else:
                self.ko_point = None
            self.remove_captured_stones()  # Remove captured stones after each move
            self.switch_player()
            return True
        return False


    def switch_player(self):
        """
        Toggle between player 1 and player 2
        """
        self.current_player = 3 - self.current_player

    def get_legal_moves(self):
        """
        Get a list of legal moves.

        Returns:
            list: A list of legal moves.
        """
        legal_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.is_valid_move((x, y)) and not self.is_suicide_move((x, y)):
                    legal_moves.append((x, y))
        return legal_moves

    def is_game_over(self):
        """
        Check if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return len(self.get_legal_moves()) == 0

    def get_winner(self):
        """
        Get the winner of the game.

        Returns:
            int: -1 if player wins, 1 if AI wins, 0 for a tie
        """
        if not self.is_game_over():
            return None
        player1_stones = sum(row.count(1) for row in self.board)
        player2_stones = sum(row.count(2) for row in self.board)
        if player1_stones > player2_stones:
            return -1  # Player 1 wins
        elif player2_stones > player1_stones:
            return 1  # Player 2 wins (AI)
        else:
            return 0  # Draw

    # Create a deep copy of the current board state
    def get_state(self):
        """
        Get a deep copy of the current board state.

        Returns:
            list: A deep copy of the current board state.
        """

        return copy.deepcopy(self.board)
    
    def render(self):
        """
        Render the current board state.
        """
        for row in self.board:
            print(' '.join(map(str, row)))

# The `MCTSTreeNode` class represents a node in a Monte Carlo Tree Search algorithm with methods for
# initializing a node and checking if it is fully expanded.
class MCTSTreeNode:
    def __init__(self, parent=None, move=None):
        """
        Initialize a node in the MCTS tree.

        Args:
            parent (MCTSTreeNode): The parent node. Defaults to None.
            move (tuple): The move associated with the node. Defaults to None.
        """
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.total_value = 0
        self.state = None
        self.unvisited_moves = None

    # Check if a node is fully expanded by examining unvisited moves
    def is_fully_expanded(self):
        """
        Check if the node is fully expanded.

        Returns:
            bool: True if the node is fully expanded, False otherwise.
        """
        return not self.unvisited_moves

# Define the UCB1 value function for node selection
def ucb1_value(node, exploration_weight=1.0):
    """
    Calculate the UCB1 value for a node.

    Args:
        node (MCTSTreeNode): The node for which to calculate the UCB1 value.
        exploration_weight (float): The exploration weight parameter. Defaults to 1.0.

    Returns:
        float: The UCB1 value for the node.
    """
    if node.visit_count == 0:
        return math.inf
    exploitation = node.total_value / node.visit_count
    exploration = exploration_weight * math.sqrt(math.log(node.parent.visit_count) / node.visit_count)
    return exploitation + exploration

# Define the node selection phase of MCTS
def select(node):
    """
    Select a node for exploration in the MCTS tree.

    Args:
        node (MCTSTreeNode): The current node in the tree.

    Returns:
        MCTSTreeNode: The selected child node.
    """
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
    """
    Expand a node in the MCTS tree.

    Args:
        node (MCTSTreeNode): The node to expand.
    """
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
    """
    Simulate a game from a given node.

    Args:
        node (MCTSTreeNode): The node from which to simulate the game.

    Returns:
        int: The result of the simulated game (-1 for player win, 1 for AI win, 0 for Tie).
    """
    state = copy.deepcopy(node.state)
    while not state.is_game_over():
        legal_moves = state.get_legal_moves()
        move = random.choice(legal_moves)
        state.make_move(*move)
    return state.get_winner()

# Define the backpropagation phase of MCTS
def backpropagate(node, result):
    """
    Backpropagate the result of a simulation through the MCTS tree.

    Args:
        node (MCTSTreeNode): The node to start backpropagation from.
        result (int): The result of the simulation.
    """
    while node is not None:
        node.visit_count += 1
        node.total_value += result
        node = node.parent

# Define the main MCTS function for parallel execution
def mcts_parallel(root, iterations, return_dict):
    """
    Perform MCTS with parallel execution.

    Args:
        root (MCTSTreeNode): The root node of the MCTS tree.
        iterations (int): The number of iterations to run MCTS.
        return_dict (multiprocessing.Manager.dict): A shared dictionary to store results.
    """
    for _ in range(iterations):
        node = select(root)
        expand(node)
        result = simulate(node)
        backpropagate(node, result)
    
    return_dict[root] = root.children

# Define the AI's move selection using MCTS with parallel execution
def ai_play_parallel(board, iterations=ITERATIONS, num_processes=PROCESSES_NUM):
    """
    Select the AI's move using MCTS with parallel execution.

    Args:
        board (GoGame): The current game board.
        iterations (int): The number of MCTS iterations. Defaults to ITERATIONS.
        num_processes (int): The number of parallel processes. Defaults to PROCESSES_NUM.

    Returns:
        tuple: The coordinates of the AI's chosen move (x, y).
    """
    root = MCTSTreeNode()
    root.state = board
    root.unvisited_moves = board.get_legal_moves()

    # Split the work among multiple processes
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=mcts_parallel, args=(root, iterations // num_processes, return_dict))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Aggregate results from different processes
    children = []
    for child_list in return_dict.values():
        children.extend(child_list)
        
    for n in children:
        print_tree_structure(n, 1)

    best_move = max(children, key=lambda c: c.visit_count)
    print("Best Move:", best_move.move, "Visit Count:", best_move.visit_count)
    
    return best_move.move

# This function is for testing purposes
def print_tree_structure(node, depth):
    """
    Print the tree structure starting from the given node.

    Args:
        node (MCTSTreeNode): The root node of the tree to print.
        depth (int): The current depth of the node in the tree.
    """
    if node is not None:
        print("  " * depth, f"Move: {node.move}, Visit Count: {node.visit_count}")
        for child in node.children:
            print_tree_structure(child, depth + 1)
            
# Define the rendering function for the game board in text
def render_game(board):
    """
    Render the game board in text format.

    Args:
        board (list): The current game board.

    Returns:
        str: The rendered game board.
    """
    out = ""
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == " ":
                out += ". "
                print(".", end=" ")  # An empty intersection
            elif board[i][j] == 1:
                out += "○ "
                print("○", end=" ")  # Player 1's stone  ○⚪◯
            elif board[i][j] == 2:
                out += "● "
                print("●", end=" ")  # Player 2's stone   ●⚫⬤
            # ⬜⬛➕
        print()
        out += "\n"
    print("------------------------------------")
    out += "------------------------------------------------------------------------"
    return out

# This function uses pygame to create the game UI
def draw_board(screen, game):
    """
    Draw the game board using Pygame.

    Args:
        screen (pygame.Surface): The Pygame screen surface.
        game (GoGame): The current game instance.
    """
    screen.fill((0, 207, 207))  # Fill the screen with white color

    # Draw grid lines
    for i in range(1, BOARD_SIZE + 1):
        pygame.draw.line(screen, (0, 0, 0), (i * GRID_SIZE, 0+GRID_SIZE), (i * GRID_SIZE, SCREEN_SIZE-GRID_SIZE), 2)
        pygame.draw.line(screen, (0, 0, 0), (0+GRID_SIZE, i * GRID_SIZE), (SCREEN_SIZE-GRID_SIZE, i * GRID_SIZE), 2)

    # Draw stones at the intersections
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            # Calculate the coordinates of the intersection point
            x = (j + 1) * GRID_SIZE
            y = (i + 1) * GRID_SIZE

            if game.board[i][j] == 1:
                pygame.draw.circle(screen, (0, 0, 0), (x, y), GRID_SIZE // 2 - 10)
            elif game.board[i][j] == 2:
                pygame.draw.circle(screen, (194, 194, 194), (x, y), GRID_SIZE // 2 - 10)

    pygame.display.flip()
    

# Define the main game loop
def main():
    """
    The main game loop.
    """
    
    date = datetime.datetime.now()
    current_time = date.strftime("%Y.%b.%d_%Hh%Mm")
    # file = open("data/"+current_time+".txt", "a", encoding='utf-8')
    # file.write("Board size: "+str(BOARD_SIZE) + "\nIterations:"+str(ITERATIONS) + "\nProcesses number:"+str(PROCESSES_NUM)+"\n")
    
    start = 0
    
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
                print("Player")
                px, py = event.pos
                print(px,py)
                px = round(px/GRID_SIZE)
                py = round(py/GRID_SIZE)
        
        if game.current_player == 1 and (px!=0 and py!=0):
            game.make_move(py-1, px-1)
            game_board_out = render_game(game.get_state())
            # file.write("\n"+str(3-game.current_player) + "\n" + game_board_out)
        elif game.current_player == 2:
            print("else")
            print("AI")
            start = time.time()
            ai_move = ai_play_parallel(game)  # Use the parallel AI function
            game.make_move(*ai_move)
            end = time.time()
            print(3-game.current_player, " Time=", end - start)
            game_board_out = render_game(game.get_state())
            # file.write("\n"+str(3-game.current_player) + "   Time= " + str(end - start)+"\n"+game_board_out)
            

    winner = game.get_winner()
    if winner == 0:
        print("It's a tie!")
        # file.write("Tie")
    elif winner == -1:
        print("You win!")
        # file.write("You win")
    elif winner == 1:
        print("AI wins!")
        # file.write("AI win")
        
    

if __name__ == "__main__":
    main()