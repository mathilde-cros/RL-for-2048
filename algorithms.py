import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

def RandomStrategy(grid):
    print(grid.cells)
    # Here, you can implement your algorithm to choose the best action
    # For demonstration, we'll choose a random valid action
    # You have access to the grid cells via grid.cells
    actions = ['up', 'down', 'left', 'right']
    # Implement your logic to choose the best action
    # For now, return a random action
    return random.choice(actions)


def evaluate_grid(grid):
    '''Evaluate the grid and return a heuristic value.'''
    empty_cells = len(grid.retrieve_empty_cells())
    max_tile = max(max(row) for row in grid.cells)
    # You can adjust weights as needed
    heuristic_value = empty_cells + (max_tile * 4)
    return heuristic_value


def HeuristicStrategy(grid):
    actions = ['up', 'down', 'left', 'right']
    best_score = -float('inf')
    best_action = None

    for action in actions:
        # Clone the grid to simulate the move
        grid_copy = grid.clone()
        moved = grid_copy.move(action)

        if not moved:
            continue  # Skip if the move doesn't change the grid

        # Evaluate the new grid
        score = evaluate_grid(grid_copy)

        if score > best_score:
            best_score = score
            best_action = action

    if best_action:
        return best_action
    else:
        # If no moves change the grid, return a random valid action
        return random.choice(actions)

# Policy Network Definition
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


# Policy Gradient Strategy
class PolicyGradientStrategy:
    def __init__(self):
        self.actions = ['up', 'down', 'left', 'right']
        self.policy_net = PolicyNetwork(input_size=16, output_size=4)  # 4x4 grid = 16 input features
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)

    def preprocess_grid(self, grid):
        """Flatten the grid into a 1D tensor."""
        return torch.tensor(grid.cells, dtype=torch.float32).flatten()

    def select_action(self, grid):
        """Select an action using the policy network."""
        state = self.preprocess_grid(grid)
        action_probs = self.policy_net(state)
        action = np.random.choice(len(self.actions), p=action_probs.detach().numpy())
        return self.actions[action]

    def train(self, state, action, reward):
        """Update the policy network using the policy gradient."""
        self.optimizer.zero_grad()

        # Convert inputs to tensors
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.int64)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)

        # Forward pass
        action_probs = self.policy_net(state_tensor)
        action_log_probs = torch.log(action_probs[action_tensor])
        loss = -action_log_probs * reward_tensor  # Policy gradient loss

        # Backward pass
        loss.backward()
        self.optimizer.step()

    def __call__(self, grid):
        """PolicyGradientStrategy interface for the Game class."""
        return self.select_action(grid)