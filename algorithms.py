import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def RandomStrategy(grid):
    """
    Randomly selects a valid action from 'up', 'down', 'left', 'right'.

    This strategy does not consider the state of the grid and is used as a baseline.
    """
    print(grid.cells)
    actions = ['up', 'down', 'left', 'right']
    return random.choice(actions)


def empty_cell_heuristic(grid):
    """
    Evaluates the grid based on the number of empty cells and the maximum tile.

    Heuristic Value = (Number of Empty Cells) + (Maximum Tile Value * 4)

    This heuristic encourages moves that keep the grid emptier (providing more space
    for future moves) and increase the maximum tile on the grid.
    """
    empty_cells = len(grid.retrieve_empty_cells())
    max_tile = max(max(row) for row in grid.cells)
    heuristic_value = empty_cells + (max_tile * 4)
    return heuristic_value


def snake_heuristic(grid):
    """
    Evaluates the grid based on a 'snake' pattern weight matrix.

    The weight matrix is designed to guide the placement of high-value tiles in a snake-like
    pattern starting from one corner, typically to maximize merging opportunities.

    Heuristic Value = Sum of (Tile Value * Corresponding Weight)
    """
    weights = [
        [16, 8, 4, 2],
        [8, 4, 2, 1],
        [4, 2, 1, 0],
        [2, 1, 0, -1]
    ]
    score = 0
    for i in range(grid.size):
        for j in range(grid.size):
            score += grid.cells[i][j] * weights[i][j]
    return score


def monotonicity_heuristic(grid):
    """
    Evaluates the grid based on how monotonic the rows and columns are.

    This heuristic measures the monotonicity (either increasing or decreasing) of the
    tile values in both rows and columns, encouraging smooth gradients.

    Heuristic Value = Higher negative value indicates better monotonicity.
    """
    totals = [0, 0, 0, 0]
    for i in range(grid.size):
        for j in range(grid.size - 1):
            current = grid.cells[i][j]
            next = grid.cells[i][j + 1]
            if current > next:
                totals[0] += next - current
            else:
                totals[1] += current - next
    for j in range(grid.size):
        for i in range(grid.size - 1):
            current = grid.cells[i][j]
            next = grid.cells[i + 1][j]
            if current > next:
                totals[2] += next - current
            else:
                totals[3] += current - next
    return max(totals[0], totals[1]) + max(totals[2], totals[3])


def smoothness_heuristic(grid):
    """
    Evaluates the grid based on how 'smooth' it is, favoring grids where neighboring
    tiles have similar values.

    This heuristic penalizes large differences between neighboring tiles, which can
    hinder merging opportunities.

    Heuristic Value = Negative sum of absolute differences between neighboring tiles.
    """
    smoothness = 0
    for i in range(grid.size):
        for j in range(grid.size):
            value = grid.cells[i][j]
            if value != 0:
                # Compare with right neighbor
                if j + 1 < grid.size and grid.cells[i][j + 1] != 0:
                    smoothness -= abs(value - grid.cells[i][j + 1])
                # Compare with bottom neighbor
                if i + 1 < grid.size and grid.cells[i + 1][j] != 0:
                    smoothness -= abs(value - grid.cells[i + 1][j])
    return smoothness


def merge_potential_heuristic(grid):
    """
    Evaluates the grid based on the number of potential merges available.

    This heuristic counts the number of adjacent tiles with the same value,
    indicating immediate merging opportunities.

    Heuristic Value = Number of potential merges.
    """
    merge_potential = 0
    for i in range(grid.size):
        for j in range(grid.size):
            value = grid.cells[i][j]
            if value != 0:
                # Check right neighbor
                if j + 1 < grid.size and grid.cells[i][j + 1] == value:
                    merge_potential += 1
                # Check bottom neighbor
                if i + 1 < grid.size and grid.cells[i + 1][j] == value:
                    merge_potential += 1
    return merge_potential


def corner_max_tile_heuristic(grid):
    """
    Evaluates the grid by checking if the maximum tile is located in one of the corners.

    This heuristic rewards grids where the highest tile is in a corner, a strategy
    that can help in organizing tiles for better merges.

    Heuristic Value = Max Tile Value (if in corner) or Negative Max Tile Value (if not).
    """
    max_tile = max(max(row) for row in grid.cells)
    corner_positions = [
        (0, 0),
        (0, grid.size - 1),
        (grid.size - 1, 0),
        (grid.size - 1, grid.size - 1)
    ]
    for x, y in corner_positions:
        if grid.cells[x][y] == max_tile:
            return max_tile
    return -max_tile  # Penalize if the max tile is not in a corner


def combined_heuristic(grid):
    """
    Evaluates the grid by combining multiple heuristics into a single score.

    The combined score is calculated as a weighted sum of the individual heuristic scores.
    Adjust the weights to prioritize different aspects of the game.

    Returns:
    - A numerical value representing the combined heuristic score.
    """
    # Calculate individual heuristic scores
    empty_cells_score = len(grid.retrieve_empty_cells())
    monotonicity_score = monotonicity_heuristic(grid)
    merge_potential_score = merge_potential_heuristic(grid)
    corner_max_tile_score = corner_max_tile_heuristic(grid)

    # Weights for each heuristic (adjust these weights based on experimentation)
    weights = {
        'empty_cells': 10,
        'monotonicity': 10,
        'merge_potential': 5,
        'corner_max_tile': 3,
    }

    # Combined heuristic score
    combined_score = (
        empty_cells_score * weights['empty_cells'] +
        monotonicity_score * weights['monotonicity'] +
        merge_potential_score * weights['merge_potential'] +
        corner_max_tile_score * weights['corner_max_tile']
    )

    return combined_score


def HeuristicStrategy(grid, heuristic_name):
    """
    Chooses the best action based on the specified heuristic.

    Parameters:
    - grid: The current game grid.
    - heuristic_name: The name of the heuristic to use for evaluation.

    Returns:
    - The action ('up', 'down', 'left', 'right') that results in the best heuristic score.

    This function uses a mapping of heuristic names to their corresponding functions
    for simplified selection.
    """
    actions = ['up', 'down', 'left', 'right']
    best_score = -float('inf')
    best_action = None

    # Mapping of heuristic names to functions
    heuristic_functions = {
        "empty-cells": empty_cell_heuristic,
        "snake": snake_heuristic,
        "monotonic": monotonicity_heuristic,
        "smoothness": smoothness_heuristic,
        "merge-potential": merge_potential_heuristic,
        "corner-max-tile": corner_max_tile_heuristic,
        "combined": combined_heuristic
    }

    heuristic_func = heuristic_functions.get(heuristic_name)
    if heuristic_func is None:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")

    for action in actions:
        # Clone the grid to simulate the move
        grid_copy = grid.clone()
        moved = grid_copy.move(action)

        if not moved:
            continue  # Skip if the move doesn't change the grid

        # Evaluate the new grid using the selected heuristic
        score = heuristic_func(grid_copy)

        if score > best_score:
            best_score = score
            best_action = action

    if best_action:
        return best_action
    else:
        # If no moves change the grid, return a random valid action
        return random.choice(actions)
    

def expectimax(grid, depth, heuristic_func):
    """
    Implements the Expectimax algorithm for grid evaluation.

    Parameters:
    - grid: The current game grid.
    - depth: The lookahead depth.
    - heuristic_func: The heuristic function to evaluate grid states.

    Returns:
    - The best heuristic score achievable from the current grid.
    """
    if depth == 0 or not grid.can_merge():
        return heuristic_func(grid)

    # Player's turn: Maximize score
    if depth % 2 == 1:
        best_score = -float('inf')
        for action in ['up', 'down', 'left', 'right']:
            grid_copy = grid.clone()
            moved = grid_copy.move(action)
            if moved:
                score = expectimax(grid_copy, depth - 1, heuristic_func)
                best_score = max(best_score, score)
        return best_score

    # Random tile placement (chance node): Average score
    else:
        empty_cells = grid.retrieve_empty_cells()
        total_score = 0
        for cell in empty_cells:
            for value in [2, 4]:
                grid_copy = grid.clone()
                grid_copy.place_tile(cell, value)
                score = expectimax(grid_copy, depth - 1, heuristic_func)
                probability = 0.9 if value == 2 else 0.1
                total_score += probability * score
        return total_score / len(empty_cells) if empty_cells else heuristic_func(grid)


def HeuristicStrategyWithLookahead(grid, heuristic_name, depth=2):
    actions = ['up', 'down', 'left', 'right']
    best_score = -float('inf')
    best_action = None

    heuristic_functions = {
        "empty-cells": empty_cell_heuristic,
        "snake": snake_heuristic,
        "monotonic": monotonicity_heuristic,
        "smoothness": smoothness_heuristic,
        "merge-potential": merge_potential_heuristic,
        "corner-max-tile": corner_max_tile_heuristic,
        "combined": combined_heuristic
    }

    heuristic_func = heuristic_functions.get(heuristic_name)
    if heuristic_func is None:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")

    for action in actions:
        grid_copy = grid.clone()
        moved = grid_copy.move(action)
        if not moved:
            continue

        score = expectimax(grid_copy, depth, heuristic_func)
        if score > best_score:
            best_score = score
            best_action = action

    return best_action if best_action else random.choice(actions)

# Policy Network Definition
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, activation_fn=nn.ReLU):
        super(PolicyNetwork, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(activation_fn())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.softmax(self.model(x), dim=-1)


# Policy Gradient Strategy
class PolicyGradientStrategy:
    def __init__(self, hidden_sizes=[128], learning_rate=1e-3, activation_fn=nn.ReLU, optimizer_cls=optim.Adam, gamma=0.99, entropy_coef=0.0):
        self.actions = ['up', 'down', 'left', 'right']
        self.policy_net = PolicyNetwork(
            input_size=16, output_size=4, hidden_sizes=hidden_sizes, activation_fn=activation_fn)
        self.optimizer = optimizer_cls(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # Discount factor
        self.entropy_coef = entropy_coef  # Coefficient for entropy regularization
        self.saved_log_probs = []
        self.rewards = []

    def preprocess_grid(self, grid):
        state = np.array(grid.cells, dtype=np.float32).flatten()
        state = np.log2(state + 1) / 16
        return torch.tensor(state, dtype=torch.float32)

    def select_action(self, grid):
        state = self.preprocess_grid(grid)
        action_probs = self.policy_net(state)
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return self.actions[action.item()]

    def train(self):
        R = 0
        policy_loss = []
        returns = []
        # Calculate the discounted rewards
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        baseline = returns.mean()
        for log_prob, R in zip(self.saved_log_probs, returns):
            advantage = R - baseline
            policy_loss.append(-log_prob * advantage)
        # Entropy regularization
        entropy_loss = -self.entropy_coef * torch.stack(self.saved_log_probs).mean()
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + entropy_loss
        loss.backward()
        self.optimizer.step()
        # Reset buffers
        self.saved_log_probs = []
        self.rewards = []

    def __call__(self, grid):
        return self.select_action(grid)