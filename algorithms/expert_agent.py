import random
import torch
from algorithms.mcts import neural_network_policy
from algorithms.train_policy_network import PolicyNetworkCNN


def load_expert_agent():
    model = PolicyNetworkCNN()
    model.load_state_dict(torch.load("./data/policy_network.pth"))
    model.eval()
    return model


def ExpertAgent(grid, model):
    """
    Expert agent strategy using a neural network policy.

    Args:
        grid (Grid): The current game grid.
        model (PolicyNetworkCNN): Trained neural network model.

    Returns:
        str: A valid action that moves the grid.
    """
    action = neural_network_policy(model, grid)

    # Check if the neural network's suggested action is valid
    if grid.can_move_action(action):
        return action
    else:
        # Retry until a valid action is found
        valid_actions = ['up', 'down', 'left', 'right']
        random.shuffle(valid_actions)  # Shuffle to avoid bias
        for act in valid_actions:
            if grid.can_move_action(act):
                return act

    # If no valid actions are available (shouldn't happen), return None
    return None
