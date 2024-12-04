import random
import numpy as np
from utils import MCTSNode

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
        [2, 4, 8, 16],
        [256, 128, 64, 32],
        [512, 1024, 2048, 4096],
        [65536, 32768, 16384, 8192]
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


def homogeneity_heuristic(grid):
    """
    Evaluates the grid based on how homogeneous the values of neighboring tiles are.

    The heuristic rewards grids where neighboring tiles have similar values, encouraging smooth gradients that
    make merging easier in future moves.

    Heuristic Value = Higher value if neighboring tiles have similar values.
    """
    homogeneity_score = 0
    for i in range(grid.size):
        for j in range(grid.size):
            value = grid.cells[i][j]
            if value != 0:
                # Compare with right neighbor
                if j + 1 < grid.size and grid.cells[i][j + 1] != 0:
                    homogeneity_score -= abs(value - grid.cells[i][j + 1])
                # Compare with bottom neighbor
                if i + 1 < grid.size and grid.cells[i + 1][j] != 0:
                    homogeneity_score -= abs(value - grid.cells[i + 1][j])

    # A higher score is better, so we return the negative of the differences
    return -homogeneity_score


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
        'empty_cells': 20,
        'monotonicity': 40,
        'merge_potential': 40,
    }

    # Combined heuristic score
    combined_score = (
        empty_cells_score * weights['empty_cells'] +
        monotonicity_score * weights['monotonicity'] +
        merge_potential_score * weights['merge_potential']
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
        "combined": combined_heuristic,
        "advanced": enhanced_combined_heuristic
    }

    heuristic_func = heuristic_functions.get(heuristic_name)
    print("HEURISTIC FUNC", heuristic_func)
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
        self.optimizer = optimizer_cls(
            self.policy_net.parameters(), lr=learning_rate)
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
        entropy_loss = -self.entropy_coef * \
            torch.stack(self.saved_log_probs).mean()
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum() + entropy_loss
        loss.backward()
        self.optimizer.step()
        # Reset buffers
        self.saved_log_probs = []
        self.rewards = []

    def __call__(self, grid):
        return self.select_action(grid)


### Monte Carlo Tree Search (MCTS) ###


def enhanced_combined_heuristic(grid):
    """
    Evaluates the grid by combining multiple heuristics into a single score with an added penalty for losing.

    The combined score is calculated as a weighted sum of the individual heuristic scores. This function also includes
    a penalty if the game is lost, which helps to discourage moves that could lead to losing the game.

    Returns:
    - A numerical value representing the combined heuristic score, where a higher value is better.
    """
    # Calculate individual heuristic scores
    empty_cells_score = len(grid.retrieve_empty_cells())
    monotonicity_score = monotonicity_heuristic(grid)
    merge_potential_score = merge_potential_heuristic(grid)
    corner_max_tile_score = corner_max_tile_heuristic(grid)
    smoothness_score = smoothness_heuristic(grid)
    homogeneity_score = homogeneity_heuristic(grid)

    # Weights for each heuristic (adjust these weights based on experimentation)
    weights = {
        'monoticity': 15,
        'homogeneity': 10,
    }

    # Combined heuristic score
    combined_score = (
        homogeneity_score * weights['homogeneity']
        + monotonicity_score * weights['monoticity']
    )

    # Penalty for losing the game
    if not grid.can_merge() and not grid.retrieve_empty_cells():
        combined_score -= 10000  # Large penalty for a losing state

    return combined_score


def expectimax_1(grid, depth, heuristic_func):
    if depth == 0 or not grid.can_merge() and not grid.has_empty_cells():
        return heuristic_func(grid)

    # Player's turn: Maximize score
    if depth % 2 == 0:
        best_score = float('-inf')
        for action in ['up', 'down', 'left', 'right']:
            grid_copy = grid.clone()
            moved = grid_copy.move(action)
            if moved:
                # Simulate random tile insertion
                empty_cells = grid_copy.retrieve_empty_cells()
                total_score = 0
                for cell in empty_cells:
                    for value, probability in [(1, 0.9 / len(empty_cells)), (2, 0.1 / len(empty_cells))]:
                        grid_after_chance = grid_copy.clone()
                        grid_after_chance.place_tile(cell, value)
                        score = expectimax_1(
                            grid_after_chance, depth - 1, heuristic_func)
                        total_score += probability * score
                best_score = max(best_score, total_score)
        return best_score

    # Chance node: Expected value over all possible tile spawns
    else:
        empty_cells = grid.retrieve_empty_cells()
        total_score = 0
        for cell in empty_cells:
            for value, probability in [(1, 0.9 / len(empty_cells)), (2, 0.1 / len(empty_cells))]:
                grid_copy = grid.clone()
                grid_copy.place_tile(cell, value)
                score = expectimax_1(grid_copy, depth - 1, heuristic_func)
                total_score += probability * score
        return total_score


def expectimax_pruned(grid, depth, heuristic_func, max_actions=2, max_chance_events=4, transposition_table=None):
    if transposition_table is None:
        transposition_table = {}

    # Create a unique key for the current grid state
    grid_key = tuple(tuple(row) for row in grid.cells)
    if grid_key in transposition_table and depth <= transposition_table[grid_key]['depth']:
        return transposition_table[grid_key]['score']

    if depth == 0 or not grid.can_merge() and not grid.has_empty_cells():
        score = heuristic_func(grid)
        transposition_table[grid_key] = {'score': score, 'depth': depth}
        return score

    # Player's turn: Maximize score
    if depth % 2 == 0:
        best_score = float('-inf')

        # Evaluate and order actions
        actions = []
        for action in ['up', 'down', 'left', 'right']:
            grid_copy = grid.clone()
            moved = grid_copy.move(action)
            if moved:
                action_score = heuristic_func(grid_copy)
                actions.append((action_score, action, grid_copy))

        # Sort actions by heuristic score (descending)
        actions.sort(reverse=True, key=lambda x: x[0])

        # Limit the number of actions considered
        actions = actions[:max_actions]

        for _, action, grid_copy in actions:
            # Simulate random tile insertion
            empty_cells = grid_copy.retrieve_empty_cells()
            if not empty_cells:
                continue
            total_score = 0
            chance_events = []
            for cell in empty_cells:
                chance_events.append(
                    (cell, 1, 0.9 / len(empty_cells)))  # '2' tile
                chance_events.append(
                    (cell, 2, 0.1 / len(empty_cells)))  # '4' tile

            # Limit the number of chance events considered
            if len(chance_events) > max_chance_events:
                chance_events = random.sample(chance_events, max_chance_events)
                # Normalize probabilities
                total_probability = sum(
                    probability for _, _, probability in chance_events)
                chance_events = [(cell, value, probability / total_probability)
                                 for cell, value, probability in chance_events]

            for cell, value, probability in chance_events:
                grid_after_chance = grid_copy.clone()
                grid_after_chance.place_tile(cell, value)
                score = expectimax_pruned(
                    grid_after_chance, depth - 1, heuristic_func, max_actions, max_chance_events, transposition_table)
                total_score += probability * score
            best_score = max(best_score, total_score)
        transposition_table[grid_key] = {'score': best_score, 'depth': depth}
        return best_score

    # Chance node: Expected value over all possible tile spawns
    else:
        empty_cells = grid.retrieve_empty_cells()
        if not empty_cells:
            return heuristic_func(grid)
        total_score = 0
        chance_events = []
        for cell in empty_cells:
            chance_events.append((cell, 1, 0.9 / len(empty_cells)))  # '2' tile
            chance_events.append((cell, 2, 0.1 / len(empty_cells)))  # '4' tile

        # Limit the number of chance events considered
        if len(chance_events) > max_chance_events:
            chance_events = random.sample(chance_events, max_chance_events)
            # Normalize probabilities
            total_probability = sum(
                probability for _, _, probability in chance_events)
            chance_events = [(cell, value, probability / total_probability)
                             for cell, value, probability in chance_events]

        for cell, value, probability in chance_events:
            grid_copy = grid.clone()
            grid_copy.place_tile(cell, value)
            score = expectimax_pruned(
                grid_copy, depth - 1, heuristic_func, max_actions, max_chance_events, transposition_table)
            total_score += probability * score
        transposition_table[grid_key] = {'score': total_score, 'depth': depth}
        return total_score


def MCTSStrategy(grid, heuristic=None, iterations=50, exploration_weight=2, expectimax_depth=1):
    root = MCTSNode(grid, parent=None, action=None)
    for _ in range(iterations):
        node = root

        # Selection
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = node.best_child(exploration_weight)

        # Expansion
        if not node.is_terminal_node():
            action = node.untried_actions.pop()
            grid_copy = node.grid.clone()
            moved = grid_copy.move(action)
            if moved:
                grid_copy.random_cell()
                # Simulate random tile insertion (chance node)
                child = MCTSNode(grid_copy, parent=node, action=action)
                node.children.append(child)
                node = child  # Move to the new child node

        # Simulation using Expectimax
        simulation_grid = node.grid.clone()
        if heuristic:
            heuristic_func = enhanced_combined_heuristic
            # Use Expectimax to evaluate the expected utility
            score = expectimax_pruned(
                simulation_grid, expectimax_depth, heuristic_func)
        else:
            # If no heuristic is provided, use the current score
            score = simulation_grid.current_score

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    # Select the best action
    print("GRID", grid.print_grid())
    print("ROOT CHILDREN SCORES", [
        (child.action, child.total_score / child.visits) for child in root.children])
    best_action = max(
        root.children, key=lambda child: child.total_score / child.visits).action
    print("BEST ACTION", best_action)
    return best_action
