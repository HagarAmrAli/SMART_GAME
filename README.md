# SMART_GAME
Below is a **README.md** file for the Python game project that implements AI algorithms (PSO, ACO, SVM, EA, and Perceptron) for a game agent, as described in the provided code. The README provides an overview, setup instructions, usage details, and additional information for users or developers.

---

# AI-Powered Game Agent

This project implements a simple 2D grid-based game where an AI-controlled agent uses various artificial intelligence algorithms to navigate, make decisions, and optimize strategies. The implemented algorithms include **Particle Swarm Optimization (PSO)**, **Ant Colony Optimization (ACO)**, **Support Vector Machine (SVM)**, **Evolutionary Algorithm (EA)**, and **Perceptron**. The user provides inputs to configure the game environment, such as the target position, obstacles, and agent health.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Example](#example)
- [Extending the Project](#extending-the-project)
- [License](#license)

## Overview
The game simulates a 10x10 grid where an agent starts at position (0,0) and aims to reach a user-defined target position while avoiding obstacles. The agent leverages five AI algorithms to perform the following tasks:
- **PSO**: Optimizes movement parameters (speed and angle) to reach the target efficiently.
- **ACO**: Finds the shortest path to the target, avoiding obstacles.
- **SVM**: Decides whether the agent should attack or retreat based on health and distance to the target.
- **EA**: Evolves weights for a strategy balancing movement toward the target and obstacle avoidance.
- **Perceptron**: Decides whether to pursue the target based on health and distance.

The user interacts with the program by providing inputs to set up the game environment and configure algorithm parameters.

## Features
- Interactive command-line interface for user inputs.
- 10x10 grid-based game environment with obstacles.
- Implementation of five AI algorithms:
  - Particle Swarm Optimization for movement optimization.
  - Ant Colony Optimization for pathfinding.
  - Support Vector Machine for action classification (attack/retreat).
  - Evolutionary Algorithm for strategy optimization.
  - Perceptron for binary pursuit decisions.
- Input validation to ensure valid target and obstacle positions.
- Modular code structure for easy modification and extension.

## Requirements
- Python 3.6 or higher
- Required Python packages:
  - `numpy`
  - `scikit-learn`

## Installation
1. Clone or download this repository to your local machine.
2. Ensure Python 3.6+ is installed. You can check your Python version with:
   ```bash
   python --version
   ```
3. Install the required packages using pip:
   ```bash
   pip install numpy scikit-learn
   ```
4. Navigate to the project directory containing the `game.py` file (or whatever you name the script).

## Usage
1. Run the script using Python:
   ```bash
   python game.py
   ```
2. Follow the prompts to provide inputs:
   - **Target position**: Enter x and y coordinates (0 to 9) for the target.
   - **Number of obstacles**: Specify how many obstacles to place on the grid.
   - **Obstacle positions**: Enter x and y coordinates for each obstacle.
   - **Agent health**: Provide a value between 0 and 1 to represent the agent's health.
   - **Number of iterations**: Specify the number of iterations for PSO, ACO, and EA algorithms.
3. The program will output the results of each AI algorithm:
   - PSO: Optimal speed and angle for movement.
   - ACO: Shortest path to the target (list of coordinates).
   - SVM: Decision to attack or retreat.
   - EA: Optimal strategy weights for moving vs. avoiding obstacles.
   - Perceptron: Decision to pursue or not pursue the target.

## How It Works
The game operates on a 10x10 grid where:
- The agent starts at (0,0).
- The user defines a target position and places obstacles.
- The grid is represented as a NumPy array (0 for free cells, 1 for obstacles).

### AI Algorithms
1. **Particle Swarm Optimization (PSO)**:
   - Optimizes movement parameters (speed and angle) to minimize time to the target.
   - Uses a swarm of particles that adjust positions based on personal and global best solutions.
2. **Ant Colony Optimization (ACO)**:
   - Finds the shortest path to the target using pheromone-based pathfinding.
   - Ants explore the grid, depositing pheromones on shorter paths while avoiding obstacles.
3. **Support Vector Machine (SVM)**:
  .ConcurrentHashMap - Classifies whether the agent should attack or retreat based on health and distance to the target.
   - Trained on a small, simulated dataset using scikit-learn's SVM implementation.
4. **Evolutionary Algorithm (EA)**:
   - Evolves weights for a strategy prioritizing movement toward the target vs. avoiding obstacles.
   - Uses selection, crossover, and mutation to improve a population of solutions.
5. **Perceptron**:
   - Decides whether to pursue the target based on health and distance.
   - Implements a simple linear classifier trained on simulated data.

### User Interaction
- The program prompts the user for inputs to set up the game environment.
- Input validation ensures that target and obstacle positions are within the grid and not overlapping.
- The user specifies the number of iterations for optimization algorithms, controlling their runtime and accuracy.

## Example
**Input**:
```
Enter target position (x, y) between 0 and 9:
Target x: 9
Target y: 9
Enter number of obstacles: 2
Enter obstacle positions (x, y):
Obstacle x: 5
Obstacle y: 5
Obstacle x: 4
Obstacle y: 4
Enter agent health (0 to 1): 0.8
Enter number of iterations for PSO/ACO/EA: 50
```

**Output**:
```
Running AI algorithms...
PSO: Optimal speed = 0.95, angle = 0.12
ACO: Shortest path = [(0, 0), (0, 1), (1, 1), ..., (9, 9)]
SVM: Agent should attack (health=0.8, distance=18)
EA: Optimal strategy weights = [move: 0.92, avoid: 0.15]
Perceptron: Agent should pursue target
```

## Extending the Project
To enhance the project, consider the following ideas:
- **Visualization**: Integrate with Pygame or Matplotlib to display the grid, agent path, and obstacles.
- **Game Engine Integration**: Port the code to a game engine like Unity (C#) or Godot for a fully interactive game.
- **Complex Environments**: Increase grid size, add dynamic obstacles, or introduce multiple agents.
- **Advanced AI**: Replace the Perceptron with a neural network or use reinforcement learning for more sophisticated behavior.
- **Real Training Data**: Collect gameplay data to train the SVM and Perceptron instead of using simulated data.
- **Performance Optimization**: Cache ACO paths, precompute SVM models, or parallelize PSO/EA for faster execution.

To implement these extensions, modify the code in `game.py` or contact the repository maintainer for guidance.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Notes for Use
- Save the above content in a file named `README.md` in the project directory.
- Replace `game.py` with the actual name of your Python script if different.
- If you add a `LICENSE` file or other resources (e.g., images, additional scripts), update the README to reference them.
- If you host the project on a platform like GitHub, ensure the repository includes the Python script, README, and any dependencies listed in a `requirements.txt` file:
  ```text
  numpy
  scikit-learn
  ```

د
