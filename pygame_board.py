import pygame
import sys

BOARD_SIZE = 6
SCREEN_SIZE = 600
GRID_SIZE = SCREEN_SIZE // (BOARD_SIZE + 1)
P = 1
def draw_board(screen, game):
    global P
    
    screen.fill((255, 255, 255))  # Fill the screen with white color

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
            
            if P == 1:
                pygame.draw.circle(screen, (0, 0, 0), (x, y), GRID_SIZE // 2 - 5)
            elif P == 2:
                pygame.draw.circle(screen, (255, 255, 255), (x, y), GRID_SIZE // 2 - 5)

            print(game)
    P = 3-P

    pygame.display.flip()
    
    
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Go Game")
x = 0
y = 0
game = (x, y)
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN and P == 1:
            x, y = event.pos
            x //= GRID_SIZE
            y //= GRID_SIZE
    
            game = (x, y)
    draw_board(screen, game)

    pygame.time.delay(100)
