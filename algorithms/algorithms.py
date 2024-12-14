import random
from algorithms.heuristics import empty_cell_heuristic, snake_heuristic, monotonicity_heuristic, smoothness_heuristic, merge_potential_heuristic, corner_max_tile_heuristic, combined_heuristic
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def RandomStrategy(grid):
    """
    Randomly selects a valid action from 'up', 'down', 'left', 'right'.

    This strategy does not consider the state of the grid and is used as a baseline.
    """
    actions = ['up', 'down', 'left', 'right']
    return random.choice(actions)


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

        # evaluate the new grid using the selected heuristic
        score = heuristic_func(grid_copy)

        if score > best_score:
            best_score = score
            best_action = action

    if best_action:
        return best_action
    else:
        # if no moves change the grid, return a random valid action
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

    if depth % 2 == 1:
        best_score = -float('inf')
        for action in ['up', 'down', 'left', 'right']:
            grid_copy = grid.clone()
            moved = grid_copy.move(action)
            if moved:
                score = expectimax(grid_copy, depth - 1, heuristic_func)
                best_score = max(best_score, score)
        return best_score

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


def HeuristicStrategyWithLookahead(grid, heuristic_name, depth=3):
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
        self.gamma = gamma
        self.entropy_coef = entropy_coef
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
        self.saved_log_probs = []
        self.rewards = []

    def __call__(self, grid):
        return self.select_action(grid)

    def __name__(self):
        return "PolicyGradient"
