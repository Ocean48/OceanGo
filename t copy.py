import random
import numpy as np
import math
import copy
import multiprocessing
import time
import datetime
import pygame
import sys
# import gym
# from gym import spaces


BOARD_SIZE = 6
ITERATIONS = 1000
PROCESSES_NUM = 7

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
        # test_board = [row[:] for row in self.board]   # old version
        test_board = copy.deepcopy(self.board)
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

# Define the main MCTS function for parallel execution
def mcts_parallel(root, iterations, return_dict):
    for _ in range(iterations):
        node = select(root)
        expand(node)
        result = simulate(node)
        backpropagate(node, result)
    
    return_dict[root] = root.children

# Define the AI's move selection using MCTS with parallel execution
def ai_play_parallel(board, iterations=ITERATIONS, num_processes=PROCESSES_NUM):
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

    best_move = max(children, key=lambda c: c.visit_count)
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


SCREEN_SIZE = (BOARD_SIZE + 1) * 100
GRID_SIZE = 100

def draw_board(screen, game, x, y):
    screen.fill((0, 207, 207))  # Fill the screen with white color
    print("draw", x ,y)

    # Draw grid lines
    for i in range(1, BOARD_SIZE + 1):
        pygame.draw.line(screen, (0, 0, 0), (i * GRID_SIZE, 0+GRID_SIZE), (i * GRID_SIZE, SCREEN_SIZE-GRID_SIZE), 2)
        pygame.draw.line(screen, (0, 0, 0), (0+GRID_SIZE, i * GRID_SIZE), (SCREEN_SIZE-GRID_SIZE, i * GRID_SIZE), 2)

    # Draw stones at the intersections
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            # Calculate the coordinates of the intersection point
            x *= 100
            y *= 100

            if game.board[i][j] == 1:
                pygame.draw.circle(screen, (0, 0, 0), (x, y), GRID_SIZE // 2 - 5)
            elif game.board[i][j] == 2:
                pygame.draw.circle(screen, (194, 194, 194), (x, y), GRID_SIZE // 2 - 5)

    pygame.display.flip()


# Define the main game loop
def main():
    date = datetime.datetime.now()
    current_time = date.strftime("%Y.%b.%d_%Hh%Mm")
    # file = open("data/multi/old/"+current_time+".txt", "a", encoding='utf-8')
    # file.write("Multi - old rules:\n"+"Board size: "+str(BOARD_SIZE) + "\nIterations:"+str(ITERATIONS) + "\nProcesses number:"+str(PROCESSES_NUM)+"\n")
    
    start = 0
    
    board_size = BOARD_SIZE
    game = GoGame(board_size)
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Go Game")
    
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.quit()
    #             sys.exit()
    #         elif event.type == pygame.MOUSEBUTTONDOWN and game.current_player == 1:
    #             x, y = event.pos
    #             x //= GRID_SIZE
    #             y //= GRID_SIZE
    #             game.make_move(x, y)

    #     game_board_out = render_game(game.get_state())
    #     draw_board(screen, game)

    #     if game.current_player == 2:
    #         ai_move = ai_play_parallel(game)
    #         game.make_move(*ai_move)

    #     pygame.time.delay(100)
    x , y = 0, 0

    while not game.is_game_over():
        
        # game_board_out = render_game(game.get_state())
        draw_board(screen, game, x ,y)
        # file.write(game_board_out+"\n")
        
    
        # game_board_out = render_game(game.get_state())
        # file.write(game_board_out+"\n")
        if game.current_player == 1:
            for event in pygame.event.get():
                # print(3-game.current_player, " : ",pygame.mouse.get_pos)
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and game.current_player == 1:
                    # x, y = event.pos
                    x, y = pygame.mouse.get_pos()
                    print(str(game.current_player)+":",x,y)
                    x = x // 100
                    y = y // 100
                    # x //= GRID_SIZE
                    # y //= GRID_SIZE
                    # game.make_move(x, y)
                
            # x, y = map(int, input("Enter your move (x y): ").split())
            # x -= 1  # Adjust the input by subtracting 1 from the row coordinate
            # y -= 1  # Adjust the input by subtracting 1 from the column coordinate
            # start = time.time()
            # game.make_move(x, y)
            # ai_move = ai_play_parallel(game)  # Use the parallel AI function
            # game.make_move(*ai_move)
        else:
            for event in pygame.event.get():
                # print(3-game.current_player, " : ",pygame.mouse.get_pos)
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and game.current_player == 1:
                    # x, y = event.pos
                    x, y = pygame.mouse.get_pos()
                    print(str(game.current_player)+":",x,y)
                    x = x / 100
                    y = y / 100
                    # game.make_move(x, y)
            # start = time.time()
            # ai_move = ai_play_parallel(game)  # Use the parallel AI function
            # game.make_move(*ai_move)
        game.current_player = 3-game.current_player
        end = time.time()
        # print(3-game.current_player, " Time=", end - start)
        # file.write(str(3-game.current_player) + "   Time= " + str(end - start)+"\n")
            

    # game_board_out = render_game(game.get_state())
    # file.write(game_board_out+"\n")
    winner = game.get_winner()
    if winner == 0:
        print("It's a tie!")
        # file.write("Tie")
    elif winner == 1:
        print("You win!")
        # file.write("You win")
    else:
        print("AI wins!")
        # file.write("AI win")
        
    

if __name__ == "__main__":
    main()