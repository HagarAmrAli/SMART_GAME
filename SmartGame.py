import numpy as np
from sklearn.svm import SVC
import random
import math

# Game grid settings
GRID_SIZE = 10
grid = np.zeros((GRID_SIZE, GRID_SIZE))  # 0: free, 1: obstacle

# Helper functions
def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def is_valid_position(x, y):
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x, y] == 0

# 1. Particle Swarm Optimization (PSO) for movement optimization
def pso_optimize(start, target, num_particles, max_iterations):
    particles = np.random.uniform(0, 1, (num_particles, 2))  # [speed, angle]
    velocities = np.random.uniform(-0.1, 0.1, (num_particles, 2))
    p_best = particles.copy()
    p_best_fitness = [float('inf')] * num_particles
    g_best = particles[0]
    g_best_fitness = float('inf')

    w, c1, c2 = 0.5, 1.0, 1.0  # PSO parameters

    for _ in range(max_iterations):
        for i in range(num_particles):
            speed, angle = particles[i]
            # Simulate movement: estimate time to target
            dist = manhattan_distance(start, target)
            time_to_target = dist / max(0.1, speed)  # Avoid division by zero
            fitness = time_to_target + abs(angle)  # Penalize large angles

            if fitness < p_best_fitness[i]:
                p_best_fitness[i] = fitness
                p_best[i] = particles[i]

            if fitness < g_best_fitness:
                g_best_fitness = fitness
                g_best = particles[i]

        for i in range(num_particles):
            r1, r2 = np.random.random(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (p_best[i] - particles[i]) +
                             c2 * r2 * (g_best - particles[i]))
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)  # Keep in bounds

    return g_best  # Optimal [speed, angle]

# 2. Ant Colony Optimization (ACO) for pathfinding
def aco_pathfinding(start, target, num_ants, max_iterations):
    pheromones = np.ones((GRID_SIZE, GRID_SIZE, GRID_SIZE, GRID_SIZE)) * 0.1
    best_path = None
    best_length = float('inf')

    alpha, beta, rho, Q = 1.0, 2.0, 0.5, 100.0  # ACO parameters

    for _ in range(max_iterations):
        paths = []
        for _ in range(num_ants):
            path = [start]
            current = start
            while current != target:
                x, y = current
                neighbors = [(x+dx, y+dy) for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]
                             if is_valid_position(x+dx, y+dy)]
                if not neighbors:
                    break

                probs = []
                for nx, ny in neighbors:
                    heuristic = 1.0 / (manhattan_distance((nx, ny), target) + 1)
                    pheromone = pheromones[x, y, nx, ny]
                    probs.append((pheromone ** alpha) * (heuristic ** beta))
                probs = np.array(probs) / sum(probs)

                next_pos = random.choices(neighbors, probs)[0]
                path.append(next_pos)
                current = next_pos

                if current == target:
                    paths.append(path)
                    break

        # Update pheromones
        pheromones *= (1 - rho)  # Evaporation
        for path in paths:
            length = len(path) - 1
            if length < best_length:
                best_length = length
                best_path = path
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                pheromones[x1, y1, x2, y2] += Q / length

    return best_path

# 3. Support Vector Machine (SVM) for decision-making
def train_svm():
    # Simulated training data: [health, distance_to_target] -> action (1: attack, -1: retreat)
    X = [[0.8, 2], [0.2, 2], [0.5, 5], [0.9, 8]]  # Health (0-1), distance
    y = [1, -1, -1, 1]  # Attack or retreat
    model = SVC(kernel='rbf')
    model.fit(X, y)
    return model

def svm_decision(model, health, distance):
    return model.predict([[health, distance]])[0]

# 4. Evolutionary Algorithm (EA) for strategy optimization
def ea_optimize(num_generations, pop_size):
    population = np.random.uniform(0, 1, (pop_size, 2))  # [move_weight, avoid_weight]
    for _ in range(num_generations):
        fitnesses = []
        for individual in population:
            move_weight, avoid_weight = individual
            # Simulate fitness: higher move_weight prioritizes target, avoid_weight avoids obstacles
            fitness = move_weight * 10 - avoid_weight * len(obstacles)  # Simplified
            fitnesses.append(fitness)

        # Selection
        parents = population[np.argsort(fitnesses)[-pop_size//2:]]
        offspring = []
        for _ in range(pop_size - len(parents)):
            p1, p2 = random.choices(parents, k=2)
            child = (p1 + p2) / 2  # Crossover
            child += np.random.normal(0, 0.1, 2)  # Mutation
            child = np.clip(child, 0, 1)
            offspring.append(child)
        population = np.vstack([parents, offspring])

    return population[np.argmax(fitnesses)]  # Best strategy

# 5. Perceptron for pursuit decision
def train_perceptron():
    weights = np.random.uniform(-1, 1, 2)  # [health, distance]
    bias = 0
    learning_rate = 0.1
    # Simulated training data: [health, distance] -> pursue (1) or not (0)
    X = [[0.8, 2], [0.2, 2], [0.5, 5], [0.9, 8]]
    y = [1, 0, 0, 1]
    for _ in range(100):
        for x, target in zip(X, y):
            pred = 1 if np.dot(weights, x) + bias > 0 else 0
            if pred != target:
                weights += learning_rate * (target - pred) * np.array(x)
                bias += learning_rate * (target - pred)
    return weights, bias

def perceptron_decision(weights, bias, health, distance):
    return 1 if np.dot(weights, [health, distance]) + bias > 0 else 0

# Main game loop
def main():
    global grid, obstacles
    obstacles = []

    # User inputs
    print("Enter target position (x, y) between 0 and 9:")
    target_x = int(input("Target x: "))
    target_y = int(input("Target y: "))
    if not is_valid_position(target_x, target_y):
        print("Invalid target position!")
        return
    target = (target_x, target_y)

    num_obstacles = int(input("Enter number of obstacles: "))
    print("Enter obstacle positions (x, y):")
    for _ in range(num_obstacles):
        x = int(input("Obstacle x: "))
        y = int(input("Obstacle y: "))
        if is_valid_position(x, y) and (x, y) != target and (x, y) != (0, 0):
            grid[x, y] = 1
            obstacles.append((x, y))

    health = float(input("Enter agent health (0 to 1): "))
    max_iterations = int(input("Enter number of iterations for PSO/ACO/EA: "))

    start = (0, 0)

    # Run AI algorithms
    print("\nRunning AI algorithms...")

    # PSO: Optimize movement
    speed, angle = pso_optimize(start, target, num_particles=20, max_iterations=max_iterations)
    print(f"PSO: Optimal speed = {speed:.2f}, angle = {angle:.2f}")

    # ACO: Find path
    path = aco_pathfinding(start, target, num_ants=10, max_iterations=max_iterations)
    if path:
        print(f"ACO: Shortest path = {path}")
    else:
        print("ACO: No path found!")

    # SVM: Attack or retreat
    svm_model = train_svm()
    distance = manhattan_distance(start, target)
    action = svm_decision(svm_model, health, distance)
    print(f"SVM: Agent should {'attack' if action == 1 else 'retreat'} (health={health}, distance={distance})")

    # EA: Optimize strategy
    move_weight, avoid_weight = ea_optimize(num_generations=max_iterations, pop_size=20)
    print(f"EA: Optimal strategy weights = [move: {move_weight:.2f}, avoid: {avoid_weight:.2f}]")

    # Perceptron: Pursue or not
    weights, bias = train_perceptron()
    pursue = perceptron_decision(weights, bias, health, distance)
    print(f"Perceptron: Agent should {'pursue' if pursue == 1 else 'not pursue'} target")

if __name__ == "__main__":
    main()