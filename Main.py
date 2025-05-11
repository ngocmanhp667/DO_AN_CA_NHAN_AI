import pygame
from Algorithms import (
    bfs, dfs, ucs, iddfs, dls, a_star, ida_star,
    simple_hill_climbing, steepest_ascent_hill_climbing,
    greedy_best_first_search, simulated_annealing,
    stochastic_hill_climbing, local_beam_search,
    genetic_algorithm, min_conflicts_search,
    backtracking_search, backtracking_forward_checking,
    q_learning_solver
)

INITIAL_STATE = ((1, 2, 3), (4, 5, 6), (0, 7, 8))
#INITIAL_STATE = ((2, 6, 5), (8, 7, 0), (4, 3, 1))
goal = ((1, 2, 3), (4, 5, 6), (7, 8, 0))

WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
GRAY = (150, 150, 150)
LIGHT_BLUE = (100, 150, 255)
DARK_BLUE = (50, 100, 200)
RED = (200, 50, 50)

pygame.init()
screen = pygame.display.set_mode((1000, 800))
pygame.display.set_caption("8 Puzzle Solver")
clock = pygame.time.Clock()

def draw_text(text, x, y, size=40, color=BLACK):
    font = pygame.font.Font(None, size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def draw_puzzle(state):
    tile_size = 150
    padding = 10
    start_x, start_y = 100, 100
    for i in range(3):
        for j in range(3):
            value = state[i][j]
            x = start_x + j * (tile_size + padding)
            y = start_y + i * (tile_size + padding)
            if value != 0:
                pygame.draw.rect(screen, LIGHT_BLUE, (x, y, tile_size, tile_size), border_radius=20)
                pygame.draw.rect(screen, DARK_BLUE, (x, y, tile_size, tile_size), width=5, border_radius=20)
                draw_text(str(value), x + 50, y + 50, size=60, color=WHITE)
            else:
                pygame.draw.rect(screen, WHITE, (x, y, tile_size, tile_size), border_radius=20)

def main():
    running = True
    state = INITIAL_STATE
    path = None
    step = 0

    ANIMATE_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(ANIMATE_EVENT, 500)

    # ✅ Nhóm 1: 9 thuật toán chính
    algo_group_1 = {
        "BFS": bfs,
        "DFS": lambda s, g: dfs(s, g, depth_limit=50),
        "UCS": ucs,
        "IDDFS": lambda s, g: iddfs(s, g, max_depth=50),
        "DLS": lambda s, g: dls(s, g, depth_limit=50),
        "A*": a_star,
        "IDA*": ida_star,
        "BT": lambda s, g: backtracking_search(s, g, depth_limit=50),
        "BT+FC": lambda s, g: backtracking_forward_checking(s, g, depth_limit=50)
    }

    # ✅ Nhóm 2: 9 thuật toán mở rộng (có Q-Learn)
    algo_group_2 = {
        "GREEDY": greedy_best_first_search,
        "SHC": simple_hill_climbing,
        "SAHC": steepest_ascent_hill_climbing,
        "SA": simulated_annealing,
        "STOCH": stochastic_hill_climbing,
        "BEAM": local_beam_search,
        "GA": genetic_algorithm,
        "MC": min_conflicts_search,
        "Q-Learn": q_learning_solver
    }

    button_height = 45
    button_width = 100
    gap = 10

    buttons_1 = []
    for i, (name, func) in enumerate(algo_group_1.items()):
        rect = pygame.Rect(10 + i * (button_width + gap), 600, button_width, button_height)
        buttons_1.append((rect, name, func))

    buttons_2 = []
    for i, (name, func) in enumerate(algo_group_2.items()):
        rect = pygame.Rect(10 + i * (button_width + gap), 660, button_width, button_height)
        buttons_2.append((rect, name, func))

    reset_button = pygame.Rect(600, 300, 120, 50)
    close_button = pygame.Rect(920, 20, 60, 30)

    while running:
        screen.fill(GRAY)
        draw_text("8-Puzzle Solver", 250, 20, size=50, color=WHITE)

        current_state = state if path is None else (path[step] if step < len(path) else path[-1])
        draw_puzzle(current_state)

        for rect, label, _ in buttons_1:
            color = LIGHT_BLUE if rect.collidepoint(pygame.mouse.get_pos()) else DARK_BLUE
            pygame.draw.rect(screen, color, rect, border_radius=15)
            draw_text(label, rect.x + 5, rect.y + 8, size=26, color=WHITE)

        for rect, label, _ in buttons_2:
            color = LIGHT_BLUE if rect.collidepoint(pygame.mouse.get_pos()) else DARK_BLUE
            pygame.draw.rect(screen, color, rect, border_radius=15)
            draw_text(label, rect.x + 5, rect.y + 8, size=26, color=WHITE)

        reset_color = LIGHT_BLUE if reset_button.collidepoint(pygame.mouse.get_pos()) else DARK_BLUE
        pygame.draw.rect(screen, reset_color, reset_button, border_radius=12)
        draw_text("RESET", reset_button.x + 20, reset_button.y + 8, size=35, color=WHITE)

        pygame.draw.rect(screen, RED, close_button, border_radius=8)
        draw_text("X", close_button.x + 20, close_button.y + 3, size=30, color=WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if close_button.collidepoint(mouse_pos):
                    running = False
                elif reset_button.collidepoint(mouse_pos):
                    path = None
                    step = 0
                    state = INITIAL_STATE
                else:
                    for rect, label, func in buttons_1 + buttons_2:
                        if rect.collidepoint(mouse_pos):
                            print(f"Running {label}...")
                            path = func(state, goal)
                            step = 0
            elif event.type == ANIMATE_EVENT:
                if path is not None and step < len(path) - 1:
                    step += 1

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
