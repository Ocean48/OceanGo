import random
import math
import copy
import multiprocessing
import time
import datetime
import pygame
import sys
import hashlib

BOARD_SIZE = 9
ITERATIONS = 10000
PROCESSES_NUM = 16  # Adjusted for practical purposes

GRID_SIZE = 30  # Adjust based on screen size
SCREEN_SIZE = (BOARD_SIZE + 1) * GRID_SIZE

KOMI = 6.5  # Komi added to White's score

class GoGame:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.board = [[' ' for _ in range(board_size)] for _ in range(board_size)]
        self.current_player = 1  # Player 1 (Black)
        self.ko_point = None
        self.previous_boards = []  # For Ko rule
        self.passes = 0  # Count consecutive passes
        self.captured_stones = {1: 0, 2: 0}
        self.komi = KOMI  # Komi added to White's score

    def copy(self, reset_previous_boards=False):
        new_game = GoGame(self.board_size)
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
        new_game.ko_point = self.ko_point
        if reset_previous_boards:
            new_game.previous_boards = []
        else:
            new_game.previous_boards = copy.deepcopy(self.previous_boards)
        new_game.passes = self.passes
        new_game.captured_stones = self.captured_stones.copy()
        new_game.komi = self.komi
        return new_game

    def is_valid_move(self, move):
        if move == 'pass':
            return True
        x, y = move
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == ' '

    def is_suicide_move(self, move):
        x, y = move
        if not self.is_valid_move(move):
            return True
        test_board = copy.deepcopy(self.board)
        test_board[x][y] = self.current_player
        if self.has_liberties((x, y), test_board):
            return False
        # Check if move captures any opponent stones
        opponent = 3 - self.current_player
        for nx, ny in self.get_adjacent_points(x, y):
            if test_board[nx][ny] == opponent:
                if not self.has_liberties((nx, ny), test_board):
                    return False
        return True

    def has_liberties(self, point, board):
        x, y = point
        color = board[x][y]
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        stack = [point]
        while stack:
            cx, cy = stack.pop()
            visited[cx][cy] = True
            for nx, ny in self.get_adjacent_points(cx, cy):
                if board[nx][ny] == ' ':
                    return True
                elif board[nx][ny] == color and not visited[nx][ny]:
                    stack.append((nx, ny))
        return False

    def get_adjacent_points(self, x, y):
        points = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                points.append((nx, ny))
        return points

    def remove_captured_stones(self, x, y):
        opponent = 3 - self.current_player
        captured = []
        for nx, ny in self.get_adjacent_points(x, y):
            if self.board[nx][ny] == opponent:
                group = self.find_group((nx, ny))
                if not self.group_has_liberties(group):
                    for gx, gy in group:
                        self.board[gx][gy] = ' '
                    captured.extend(group)
        # Handle suicide
        if not captured:
            group = self.find_group((x, y))
            if not self.group_has_liberties(group):
                for gx, gy in group:
                    self.board[gx][gy] = ' '
                captured.extend(group)
        self.captured_stones[opponent] += len(captured)

    def find_group(self, start):
        color = self.board[start[0]][start[1]]
        group = set()
        stack = [start]
        while stack:
            x, y = stack.pop()
            group.add((x, y))
            for nx, ny in self.get_adjacent_points(x, y):
                if self.board[nx][ny] == color and (nx, ny) not in group:
                    stack.append((nx, ny))
        return group

    def group_has_liberties(self, group):
        for x, y in group:
            for nx, ny in self.get_adjacent_points(x, y):
                if self.board[nx][ny] == ' ':
                    return True
        return False

    def make_move(self, move):
        if move == 'pass':
            self.passes += 1
            self.switch_player()
            return True
        x, y = move
        if not self.is_valid_move(move):
            return False
        if self.is_suicide_move(move):
            return False
        # Copy board to test for Ko
        previous_board = copy.deepcopy(self.board)
        self.board[x][y] = self.current_player
        self.remove_captured_stones(x, y)
        # Check for Ko
        board_hash = self.board_hash()
        if board_hash in self.previous_boards:
            # Undo move
            self.board = previous_board
            return False
        else:
            self.previous_boards.append(board_hash)
            self.passes = 0  # Reset pass count
            self.switch_player()
            return True

    def switch_player(self):
        self.current_player = 3 - self.current_player

    def get_legal_moves(self):
        legal_moves = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x][y] == ' ':
                    move = (x, y)
                    if not self.is_suicide_move(move):
                        # Check Ko
                        test_game = self.copy()
                        if test_game.make_move(move):
                            legal_moves.append(move)
        legal_moves.append('pass')
        return legal_moves

    def is_game_over(self):
        return self.passes >= 2

    def get_winner(self):
        territory = self.count_territory()
        score1 = territory[1] + self.captured_stones[1]
        score2 = territory[2] + self.captured_stones[2] + self.komi
        if score1 > score2:
            return 1
        elif score2 > score1:
            return 2
        else:
            return 0

    def count_territory(self):
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        territory = {1: 0, 2: 0}
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x][y] == ' ' and not visited[x][y]:
                    area, owner = self.flood_fill_territory((x, y), visited)
                    if owner in [1, 2]:
                        territory[owner] += len(area)
        return territory

    def flood_fill_territory(self, start, visited):
        area = set()
        queue = [start]
        owner = None
        while queue:
            x, y = queue.pop()
            if visited[x][y]:
                continue
            visited[x][y] = True
            area.add((x, y))
            for nx, ny in self.get_adjacent_points(x, y):
                if self.board[nx][ny] == ' ':
                    queue.append((nx, ny))
                elif self.board[nx][ny] != ' ':
                    if owner is None:
                        owner = self.board[nx][ny]
                    elif owner != self.board[nx][ny]:
                        owner = 'neutral'
        return area, owner

    def board_hash(self):
        board_str = ''.join([''.join(map(str, row)) for row in self.board])
        return hashlib.md5(board_str.encode('utf-8')).hexdigest()

    def render(self):
        for row in self.board:
            print(' '.join(map(str, row)))

    def get_state(self):
        return copy.deepcopy(self.board)

class MCTSTreeNode:
    def __init__(self, parent=None, move=None, state=None):
        self.parent = parent
        self.move = move
        self.children = []
        self.visit_count = 0
        self.total_value = 0
        self.state = state
        if state:
            self.unvisited_moves = state.get_legal_moves()
        else:
            self.unvisited_moves = []

    def is_fully_expanded(self):
        return not self.unvisited_moves

    def is_terminal_node(self):
        return self.state.is_game_over()

def ucb1_value(node, exploration_weight=1.4):
    if node.visit_count == 0:
        return float('inf')
    exploitation = node.total_value / node.visit_count
    exploration = exploration_weight * math.sqrt(math.log(node.parent.visit_count) / node.visit_count)
    return exploitation + exploration

def select(node):
    while not node.is_terminal_node():
        if not node.is_fully_expanded():
            return expand(node)
        else:
            node = best_child(node)
    return node

def expand(node):
    while node.unvisited_moves:
        move = node.unvisited_moves.pop()
        next_state = node.state.copy()
        if not next_state.make_move(move):
            continue  # Skip invalid move
        child_node = MCTSTreeNode(parent=node, move=move, state=next_state)
        node.children.append(child_node)
        return child_node
    return node  # If no valid moves, return node

def best_child(node):
    choices_weights = [
        (child, ucb1_value(child))
        for child in node.children
    ]
    return max(choices_weights, key=lambda x: x[1])[0]

def simulate(node):
    state = node.state.copy(reset_previous_boards=True)
    while not state.is_game_over():
        legal_moves = state.get_legal_moves()
        move = random.choice(legal_moves)
        state.make_move(move)
    winner = state.get_winner()
    if winner == 0:
        return 0.5
    elif winner == node.state.current_player:
        return 1
    else:
        return 0

def backpropagate(node, result):
    while node is not None:
        node.visit_count += 1
        node.total_value += result
        node = node.parent

def mcts_worker(game_state, iterations):
    root = MCTSTreeNode(state=game_state.copy())
    for i in range(iterations):
        node = select(root)
        result = simulate(node)
        backpropagate(node, result)
        if i % 10 == 0:  # Print every 100 iterations
            print(f"AI thinking... Iteration {i}")
    # Collect visit counts of root's children
    move_visits = {}
    for child in root.children:
        move_visits[child.move] = child.visit_count
    return move_visits

def ai_play(game, iterations=ITERATIONS, num_processes=PROCESSES_NUM):
    manager = multiprocessing.Manager()
    game_state = game.copy()
    iterations_per_process = iterations // num_processes
    pool = multiprocessing.Pool(processes=num_processes)
    results = []
    for _ in range(num_processes):
        results.append(pool.apply_async(mcts_worker, args=(game_state, iterations_per_process)))
    pool.close()
    pool.join()
    move_visits_total = {}
    for res in results:
        move_visits = res.get()
        for move, visits in move_visits.items():
            if move in move_visits_total:
                move_visits_total[move] += visits
            else:
                move_visits_total[move] = visits
    # Choose the move with the highest visit count
    if not move_visits_total:
        return 'pass'
    best_move = max(move_visits_total.items(), key=lambda x: x[1])[0]
    print(f"AI selected move: {best_move}")
    game_state.make_move(best_move)
    print("Board after AI's move:")
    game_state.render()
    return best_move

def draw_board(screen, game):
    screen.fill((222, 184, 135))  # Wood color

    # Draw grid lines
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, (0, 0, 0),
                         ((i + 1) * GRID_SIZE, GRID_SIZE),
                         ((i + 1) * GRID_SIZE, SCREEN_SIZE - GRID_SIZE), 1)
        pygame.draw.line(screen, (0, 0, 0),
                         (GRID_SIZE, (i + 1) * GRID_SIZE),
                         (SCREEN_SIZE - GRID_SIZE, (i + 1) * GRID_SIZE), 1)

    # Draw star points
    if BOARD_SIZE == 9:
        star_coords = [(2, 2), (6, 2), (2, 6), (6, 6), (4, 4)]
    elif BOARD_SIZE == 13:
        star_coords = [(3, 3), (9, 3), (3, 9), (9, 9), (6, 6)]
    elif BOARD_SIZE == 19:
        star_coords = [(3, 3), (9, 3), (15, 3), (3, 9), (9, 9), (15, 9), (3, 15), (9, 15), (15, 15)]
    else:
        star_coords = []

    for i, j in star_coords:
        x = (j + 1) * GRID_SIZE
        y = (i + 1) * GRID_SIZE
        pygame.draw.circle(screen, (0, 0, 0), (x, y), 4)

    # Draw stones at the intersections
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            x = (j + 1) * GRID_SIZE
            y = (i + 1) * GRID_SIZE

            if game.board[i][j] == 1:
                pygame.draw.circle(screen, (0, 0, 0), (x, y), GRID_SIZE // 2 - 2)
            elif game.board[i][j] == 2:
                pygame.draw.circle(screen, (255, 255, 255), (x, y), GRID_SIZE // 2 - 2)
                pygame.draw.circle(screen, (0, 0, 0), (x, y), GRID_SIZE // 2 - 2, 1)

def get_board_position(pos):
    x, y = pos
    col = round((x - GRID_SIZE) / GRID_SIZE)
    row = round((y - GRID_SIZE) / GRID_SIZE)
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    else:
        return None

def main():
    board_size = BOARD_SIZE
    game = GoGame(board_size)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE + 50))
    pygame.display.set_caption("Go Game")

    font = pygame.font.SysFont(None, 24)

    clock = pygame.time.Clock()

    ai_thinking = False

    while True:
        draw_board(screen, game)

        # Display current player
        if game.current_player == 1:
            player_text = "Current Player: Black (You)"
        else:
            player_text = "Current Player: White (AI)"
        text = font.render(player_text, True, (0, 0, 0))
        screen.blit(text, (10, SCREEN_SIZE + 10))

        # Display captured stones
        captured_text = f"Captured Stones - Black: {game.captured_stones[1]}, White: {game.captured_stones[2]}"
        text = font.render(captured_text, True, (0, 0, 0))
        screen.blit(text, (10, SCREEN_SIZE + 30))

        pygame.display.flip()
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if game.is_game_over():
                continue
            if game.current_player == 1:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    board_pos = get_board_position(pos)
                    if board_pos:
                        row, col = board_pos
                        move = (row, col)
                        if game.make_move(move):
                            pass
                        else:
                            print("Invalid move!")
                    else:
                        print("Clicked outside the board!")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        game.make_move('pass')
            elif game.current_player == 2 and not ai_thinking:
                ai_thinking = True
                start = time.time()
                ai_move = ai_play(game)
                game.make_move(ai_move)
                end = time.time()
                print(f"AI move: {ai_move}, Time taken: {end - start:.2f} seconds")
                ai_thinking = False

        if game.is_game_over():
            winner = game.get_winner()
            territory = game.count_territory()
            score1 = territory[1] + game.captured_stones[1]
            score2 = territory[2] + game.captured_stones[2] + game.komi
            if winner == 0:
                result_text = "It's a tie!"
            else:
                result_text = f"Player {winner} wins!"
            text = font.render(result_text, True, (0, 0, 0))
            screen.blit(text, (10, 10))
            score_text = f"Scores - Black: {score1}, White: {score2}"
            text = font.render(score_text, True, (0, 0, 0))
            screen.blit(text, (10, 30))
            pygame.display.flip()
            print(result_text)
            print(score_text)
            continue

if __name__ == "__main__":
    main()
