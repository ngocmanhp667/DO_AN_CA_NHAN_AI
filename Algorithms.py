import math
import random
import heapq
from collections import deque

def find_blank(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

def generate_moves(state):
    x, y = find_blank(state)
    moves = []
    directions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
    for direction, (dx, dy) in directions.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [list(row) for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            moves.append((tuple(tuple(row) for row in new_state), direction))  # ✅ trả về (state, direction)
    return moves


def heuristic(state, goal):
    h = 0
    for i in range(3):
        for j in range(3):
            if state[i][j] != goal[i][j] and state[i][j] != 0:
                h += 1
    return h

def a_star(start, goal):
    open_list = [(0 + heuristic(start, goal), 0, start, [start])]
    closed_list = set()
    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        if current == goal:
            return path
        closed_list.add(current)
        for new_state, _ in generate_moves(current):
            if new_state not in closed_list:
                heapq.heappush(open_list, (g + 1 + heuristic(new_state, goal), g + 1, new_state, path + [new_state]))
    return None

def ida_star(start, goal):
    def search(path, g, bound):
        current = path[-1]
        f = g + heuristic(current, goal)
        if f > bound:
            return f
        if current == goal:
            return path
        min_bound = float('inf')
        for new_state, _ in generate_moves(current):
            if new_state not in path:
                path.append(new_state)
                result = search(path, g + 1, bound)
                if isinstance(result, list):
                    return result
                min_bound = min(min_bound, result)
                path.pop()
        return min_bound

    bound = heuristic(start, goal)
    while True:
        result = search([start], 0, bound)
        if isinstance(result, list):
            return result
        if result == float('inf'):
            return None
        bound = result

def bfs(start, goal):
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path
        for new_state, _ in generate_moves(current):
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [new_state]))
    return None

def dfs(start, goal, depth_limit=50):
    stack = [(start, [start], 0)]
    while stack:
        current, path, depth = stack.pop()
        if current == goal:
            return path
        if depth < depth_limit:
            for new_state, _ in generate_moves(current):
                if new_state not in path:
                    stack.append((new_state, path + [new_state], depth + 1))
    return None

def ucs(start, goal):
    queue = [(0, start, [start])]
    visited = {}
    while queue:
        cost, current, path = heapq.heappop(queue)
        if current == goal:
            return path
        if current in visited and visited[current] <= cost:
            continue
        visited[current] = cost
        for new_state, _ in generate_moves(current):
            heapq.heappush(queue, (cost + 1, new_state, path + [new_state]))
    return None
def iddfs(start, goal, max_depth=30):
    def dls(node, path, depth):
        if node == goal:
            return path
        if depth == 0:
            return None
        neighbors = generate_moves(node)
        # ✅ Sắp xếp theo độ gần đích
        neighbors.sort(key=lambda mv: heuristic(mv[0], goal))
        for new_state, _ in neighbors:
            if new_state not in path:
                result = dls(new_state, path + [new_state], depth - 1)
                if result:
                    return result
        return None

    for depth in range(max_depth + 1):
        result = dls(start, [start], depth)
        if result:
            return result
    return None

def dls(start, goal, depth_limit=50):
    def recursive_dls(node, path, limit):
        if node == goal:
            return (path, False)
        if limit == 0:
            return (None, True)
        cutoff_occurred = False
        for new_state, _ in generate_moves(node):
            if new_state not in path:
                result, cutoff = recursive_dls(new_state, path + [new_state], limit - 1)
                if cutoff:
                    cutoff_occurred = True
                elif result is not None:
                    return (result, False)
        return (None, cutoff_occurred)

    solution, _ = recursive_dls(start, [start], depth_limit)
    return solution

def simple_hill_climbing(start, goal):
    current = start
    path = [current]
    while True:
        neighbors = [s for s, _ in generate_moves(current)]
        neighbors = [n for n in neighbors if n not in path]
        if not neighbors:
            break
        next_state = min(neighbors, key=lambda x: heuristic(x, goal))
        if heuristic(next_state, goal) >= heuristic(current, goal):
            break
        current = next_state
        path.append(current)
        if current == goal:
            return path
    return None

def steepest_ascent_hill_climbing(start, goal):
    current = start
    path = [current]
    while True:
        neighbors = [s for s, _ in generate_moves(current)]
        neighbors = [n for n in neighbors if n not in path]
        if not neighbors:
            break
        best_neighbor = min(neighbors, key=lambda x: heuristic(x, goal))
        if heuristic(best_neighbor, goal) >= heuristic(current, goal):
            break
        current = best_neighbor
        path.append(current)
        if current == goal:
            return path
    return None
def greedy_best_first_search(start, goal):
    from heapq import heappush, heappop
    open_list = [(heuristic(start, goal), start, [start])]
    visited = set()

    while open_list:
        h, current, path = heappop(open_list)
        if current == goal:
            return path
        visited.add(current)
        for new_state, _ in generate_moves(current):
            if new_state not in visited:
                heappush(open_list, (heuristic(new_state, goal), new_state, path + [new_state]))
    return None
def simulated_annealing(start, goal, initial_temp=1000, cooling_rate=0.99, min_temp=1e-3):
    current = start
    path = [current]
    temperature = initial_temp

    while temperature > min_temp:
        neighbors = [s for s, _ in generate_moves(current)]
        neighbors = [n for n in neighbors if n not in path]

        if not neighbors:
            break

        next_state = random.choice(neighbors)
        delta_e = heuristic(current, goal) - heuristic(next_state, goal)

        if delta_e > 0 or math.exp(delta_e / temperature) > random.random():
            current = next_state
            path.append(current)

            if current == goal:
                return path

        temperature *= cooling_rate

    return None
def stochastic_hill_climbing(start, goal):
    import random

    current = start
    path = [current]

    while True:
        neighbors = [s for s, _ in generate_moves(current)]
        better_neighbors = [n for n in neighbors if heuristic(n, goal) < heuristic(current, goal)]

        if not better_neighbors:
            break

        next_state = random.choice(better_neighbors)
        current = next_state
        path.append(current)

        if current == goal:
            return path

    return None
def local_beam_search(start, goal, k=2):
    current_states = [start]
    paths = {start: [start]}

    for _ in range(100):  # Giới hạn 100 bước để tránh lặp vô tận
        all_neighbors = []
        for state in current_states:
            neighbors = [s for s, _ in generate_moves(state)]
            for neighbor in neighbors:
                if neighbor not in paths:
                    paths[neighbor] = paths[state] + [neighbor]
                    all_neighbors.append(neighbor)

        if not all_neighbors:
            return None

        # Chọn k trạng thái có heuristic thấp nhất
        current_states = sorted(all_neighbors, key=lambda x: heuristic(x, goal))[:k]

        for state in current_states:
            if state == goal:
                return paths[state]

    return None
def heuristic(state, goal):
    return sum(
        1 for i in range(3) for j in range(3)
        if state[i][j] != goal[i][j] and state[i][j] != 0
    )

# Sinh các nước đi hợp lệ
def generate_moves(state):
    x, y = find_blank(state)
    moves = []
    directions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
    for direction, (dx, dy) in directions.items():
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [list(row) for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            moves.append((tuple(tuple(row) for row in new_state), direction))
    return moves

# Chuyển đổi
def flat_to_matrix(flat):
    return tuple(tuple(flat[i*3:(i+1)*3]) for i in range(3))

def matrix_to_flat(matrix):
    return [cell for row in matrix for cell in row]

# Kiểm tra trạng thái có thể giải
def is_solvable(puzzle_flat):
    inv_count = 0
    for i in range(len(puzzle_flat)):
        for j in range(i + 1, len(puzzle_flat)):
            if puzzle_flat[i] and puzzle_flat[j] and puzzle_flat[i] > puzzle_flat[j]:
                inv_count += 1
    return inv_count % 2 == 0

# Tạo trạng thái ngẫu nhiên hợp lệ
def random_state():
    while True:
        flat = [i for i in range(9)]
        random.shuffle(flat)
        if is_solvable(flat):
            return flat_to_matrix(flat)
def mutate(state):
    neighbors = generate_moves(state)
    return random.choice(neighbors)[0] if neighbors else state

def genetic_algorithm(start, goal, population_size=10, generations=500):
    population = [start] + [random_state() for _ in range(population_size - 1)]
    best_paths = {start: [start]}

    for gen in range(generations):
        population = [p for p in population if isinstance(p, tuple) and len(p) == 3]
        if not population:
            return None

        population.sort(key=lambda x: heuristic(x, goal))
        if population[0] == goal:
            return best_paths[population[0]]

        next_gen = []
        for i in range(population_size):
            parent = population[i % len(population)]
            if parent not in best_paths:
                best_paths[parent] = [parent]

            for _ in range(5):
                child = mutate(parent)
                if child != parent:
                    break
            else:
                continue

            if child not in best_paths or len(best_paths[child]) > len(best_paths[parent]) + 1:
                best_paths[child] = best_paths[parent] + [child]

            next_gen.append(child)

        if not next_gen:
            break
        population = next_gen

    return None    
def min_conflicts_search(start, goal, max_steps=100):
    import random

    current = start
    path = [current]

    for _ in range(max_steps):
        if current == goal:
            return path

        neighbors = [s for s, _ in generate_moves(current)]
        if not neighbors:
            break

        # Chọn láng giềng có ít xung đột nhất (heuristic nhỏ nhất)
        next_state = min(neighbors, key=lambda x: heuristic(x, goal))
        if heuristic(next_state, goal) >= heuristic(current, goal):
            next_state = random.choice(neighbors)

        current = next_state
        path.append(current)

    return None
def backtracking_search(start, goal, depth_limit=50):
    stack = [(start, [start], 0)]
    visited = set()

    while stack:
        current, path, depth = stack.pop()

        if current == goal:
            return path

        if depth < depth_limit:
            visited.add(current)
            for new_state, _ in generate_moves(current):
                if new_state not in visited:
                    stack.append((new_state, path + [new_state], depth + 1))

    return None
def backtracking_forward_checking(start, goal, depth_limit=50):
    stack = [(start, [start], 0)]
    visited = set([start])  # thêm start vào visited

    while stack:
        current, path, depth = stack.pop()

        if current == goal:
            return path

        if depth < depth_limit:
            current_h = heuristic(current, goal)
            for new_state, _ in generate_moves(current):
                if new_state not in path and new_state not in visited:
                    if heuristic(new_state, goal) <= current_h:  # forward checking
                        visited.add(new_state)
                        stack.append((new_state, path + [new_state], depth + 1))

    return None
def q_learning_solver(start, goal, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.2):
    import random

    def get_state_key(state):
        return str(state)

    Q = {}
    for _ in range(episodes):
        state = start
        visited = [state]
        for _ in range(100):
            key = get_state_key(state)
            if key not in Q:
                Q[key] = {}

            moves = generate_moves(state)
            next_states = [s for s, _ in moves]

            if not next_states:
                break

            if random.random() < epsilon:
                next_state = random.choice(next_states)
            else:
                max_q = -float('inf')
                best = None
                for s in next_states:
                    s_key = get_state_key(s)
                    q_val = Q.get(s_key, {}).get("value", 0)
                    if q_val > max_q:
                        max_q = q_val
                        best = s
                next_state = best if best else random.choice(next_states)

            reward = 100 if next_state == goal else -1

            old_q = Q[key].get("value", 0)
            future_q = Q.get(get_state_key(next_state), {}).get("value", 0)
            Q[key]["value"] = old_q + alpha * (reward + gamma * future_q - old_q)

            state = next_state
            visited.append(state)

            if state == goal:
                break

    # Tìm đường đi từ start đến goal
    state = start
    path = [state]
    for _ in range(100):
        key = get_state_key(state)
        if key not in Q:
            break
        moves = generate_moves(state)
        next_states = [s for s, _ in moves]
        if not next_states:
            break
        best = max(next_states, key=lambda s: Q.get(get_state_key(s), {}).get("value", -float('inf')))
        path.append(best)
        if best == goal:
            return path
        state = best
    return None
def dqn_solver(start, goal, episodes=3000, gamma=0.95, alpha=0.1, epsilon=0.2):
    import random

    def get_key(state):
        return str(state)

    Q = {}

    for _ in range(episodes):
        state = start
        for _ in range(100):
            key = get_key(state)
            if key not in Q:
                Q[key] = {}
            moves = generate_moves(state)
            if not moves:
                break
            if random.random() < epsilon:
                next_state, _ = random.choice(moves)
            else:
                next_state = max(
                    [s for s, _ in moves],
                    key=lambda s: Q.get(get_key(s), {}).get("v", 0),
                    default=state
                )
            reward = 100 if next_state == goal else -1
            Q[key]["v"] = Q[key].get("v", 0) + alpha * (reward + gamma * Q.get(get_key(next_state), {}).get("v", 0) - Q[key].get("v", 0))
            state = next_state
            if state == goal:
                break

    # reconstruct path
    state = start
    path = [state]
    for _ in range(100):
        moves = generate_moves(state)
        next_state = max([s for s, _ in moves], key=lambda s: Q.get(get_key(s), {}).get("v", -1), default=None)
        if not next_state or next_state in path:
            break
        path.append(next_state)
        if next_state == goal:
            return path
        state = next_state
    return None
def sarsa_solver(start, goal, episodes=3000, alpha=0.1, gamma=0.95, epsilon=0.2):
    import random

    def get_key(state):
        return str(state)

    Q = {}

    for _ in range(episodes):
        state = start
        for _ in range(100):
            key = get_key(state)
            Q.setdefault(key, {})
            moves = generate_moves(state)
            if not moves:
                break
            next_state, _ = random.choice(moves)
            next_key = get_key(next_state)
            reward = 100 if next_state == goal else -1
            Q[key]["v"] = Q.get(key, {}).get("v", 0) + alpha * (
                reward + gamma * Q.get(next_key, {}).get("v", 0) - Q.get(key, {}).get("v", 0)
            )
            state = next_state
            if state == goal:
                break

    # path theo Q
    state = start
    path = [state]
    for _ in range(100):
        moves = generate_moves(state)
        next_state = max([s for s, _ in moves], key=lambda s: Q.get(get_key(s), {}).get("v", -1), default=None)
        if not next_state or next_state in path:
            break
        path.append(next_state)
        if next_state == goal:
            return path
        state = next_state
    return None
def partial_observation_search(start, goal, max_steps=200):
    import random

    def get_local_observation(state):
        x, y = find_blank(state)
        obs = {}
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 3 and 0 <= ny < 3:
                obs[(nx, ny)] = state[nx][ny]
        return obs

    current = start
    path = [current]
    visited = {current}

    for _ in range(max_steps):
        if current == goal:
            return path

        neighbors = [s for s, _ in generate_moves(current) if s not in visited]
        if not neighbors:
            break

        # Ưu tiên neighbor có heuristic tốt hơn
        next_state = min(neighbors, key=lambda s: heuristic(s, goal), default=None)
        if next_state is None:
            break

        visited.add(next_state)
        path.append(next_state)
        current = next_state

        if current == goal:
            return path

    return None
def no_observation_search(start, goal, max_steps=200):
    from itertools import permutations

    def is_solvable(flat):
        inv_count = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] and flat[j] and flat[i] > flat[j]:
                    inv_count += 1
        return inv_count % 2 == 0

    def flat_to_matrix(flat):
        return tuple(tuple(flat[i*3:(i+1)*3]) for i in range(3))

    def generate_all_states():
        all_states = set()
        for p in permutations(range(9)):
            flat = list(p)
            if is_solvable(flat):
                all_states.add(flat_to_matrix(flat))
        return all_states

    belief = generate_all_states()
    for _ in range(max_steps):
        if len(belief) == 1 and goal in belief:
            return [goal]
        next_belief = set()
        for state in belief:
            for new_state, _ in generate_moves(state):
                next_belief.add(new_state)
        if next_belief == belief or not next_belief:
            break
        belief = next_belief
    return None
