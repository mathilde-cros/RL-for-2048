import random
import numpy as np
import torch
from algorithms.heuristics import get_heuristic_function
from utils.helpers import MCTSNode


### Monte Carlo Tree Search (MCTS) ###
def MCTSStrategy(grid, heuristic=None, iterations=50, simulation_depth=50, exploration_weight=0.5):
    root = MCTSNode(grid)

    for _ in range(iterations):
        node = root

        # Selection
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = node.best_child(exploration_weight)

        # Expansion
        if not node.is_terminal_node():
            action = node.untried_actions.pop()
            grid_copy = node.grid.clone()
            # Check if the move changes the grid
            moved = grid_copy.move(action)
            if moved:
                grid_copy.random_cell()
                child = MCTSNode(grid_copy, parent=node, action=action)
                node.children.append(child)
                node = child

        # Simulation
        simulation_grid = node.grid.clone()

        for _ in range(simulation_depth):
            if not simulation_grid.can_move():
                break
            action = random.choice(['up', 'down', 'left', 'right'])
            # Ensure the move changes the grid
            moved = simulation_grid.move(action)
            if moved and simulation_grid.can_move():
                simulation_grid.random_cell()

        score = simulation_grid.current_score

        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    # print for each child the number of visits
    # for child in root.children:
    #     print(f"Action: {child.action}, Visits: {child.visits}")

    # # print score for each child
    # for child in root.children:
    #     print(
    #         f"Action: {child.action}, Score: {child.total_score / child.visits}")
    # Choose the best action based on average score
    best_action = max(
        root.children, key=lambda child: child.total_score / child.visits
    ).action
    return best_action


def MCTSStrategyHeuristic(grid, heuristic, iterations=50, simulation_depth=50, exploration_weight=0.5):
    root = MCTSNode(grid)

    for _ in range(iterations):
        node = root

        # Selection
        while not node.is_terminal_node() and node.is_fully_expanded():
            node = node.best_child(exploration_weight)

        # Expansion
        if not node.is_terminal_node():
            action = node.untried_actions.pop()
            grid_copy = node.grid.clone()
            # Check if the move changes the grid
            moved = grid_copy.move(action)
            if moved:
                grid_copy.random_cell()
                child = MCTSNode(grid_copy, parent=node, action=action)
                node.children.append(child)
                node = child

        # Simulation
        simulation_grid = node.grid.clone()
        for _ in range(simulation_depth):
            if not simulation_grid.can_move():
                break
            # action = random.choice(['up', 'down', 'left', 'right'])

            # for each action, take it and evalute the grid with the heuristic
            best_action = None
            best_score = -np.inf
            for action in ['up', 'down', 'left', 'right']:
                grid_copy = simulation_grid.clone()
                moved = grid_copy.move(action)
                if moved:
                    if best_action is None:
                        best_action = action
                        heuristic_fct = get_heuristic_function(
                            heuristic)
                        best_score = heuristic_fct(grid_copy)
                    else:
                        heuristic_fct = get_heuristic_function(heuristic)
                        score = heuristic_fct(grid_copy)
                        if score > best_score:
                            best_action = action
                            best_score = score
                    grid_copy.random_cell()
            action = best_action
            # # Ensure the move changes the grid
            moved = simulation_grid.move(action)
            if moved and simulation_grid.can_move():
                simulation_grid.random_cell()

        score = simulation_grid.current_score

        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    # print for each child the number of visits
    # for child in root.children:
    #     print(f"Action: {child.action}, Visits: {child.visits}")

    # # print score for each child
    # for child in root.children:
    #     print(
    #         f"Action: {child.action}, Score: {child.total_score / child.visits}")
    # Choose the best action based on average score
    best_action = max(
        root.children, key=lambda child: child.total_score / child.visits
    ).action
    return best_action


def neural_network_policy(model, grid):
    """
    Use the trained neural network to predict the best action.

    Args:
        model (PolicyNetworkCNN): Trained neural network model
        grid (Grid): Current game grid

    Returns:
        str: Predicted best action ('up', 'down', 'left', or 'right')
    """
    # Prepare the grid as input for the model
    grid_array = np.array(grid.cells).reshape(1, 1, 4, 4).astype(np.float32)
    grid_tensor = torch.tensor(grid_array)
    # Get model predictions
    with torch.no_grad():
        outputs = model(grid_tensor)
        predicted_action_index = torch.argmax(outputs).item()

    # Map index to action
    action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
    return action_map[predicted_action_index]


def MCTSExpertStrategy(grid, model, heuristic=None, iterations=50, simulation_depth=100, exploration_weight=1):
    """
    Monte Carlo Tree Search strategy enhanced with neural network guidance

    Args:
        grid (Grid): Current game grid
        model (PolicyNetworkCNN): Trained neural network model
        heuristic (str, optional): Heuristic function to use
        iterations (int): Number of MCTS iterations
        simulation_depth (int): Depth of simulation rollouts
        exploration_weight (float): UCB exploration parameter

    Returns:
        str: Best action to take
    """
    root = MCTSNode(grid)

    for _ in range(iterations):
        node = root

        # Selection
        while not node.is_terminal_node() and node.is_fully_expanded():
            # Use neural network to guide child selection with some randomness
            if random.random() < 0.8:  # 80% neural network guidance
                try:
                    nn_action = neural_network_policy(model, node.grid)
                    children_actions = [
                        child.action for child in node.children]

                    if nn_action in children_actions:
                        node = next(
                            child for child in node.children if child.action == nn_action)
                    else:
                        node = node.best_child(exploration_weight)
                except:
                    node = node.best_child(exploration_weight)
            else:
                node = node.best_child(exploration_weight)

        # Expansion
        if not node.is_terminal_node():
            action = node.untried_actions.pop()
            grid_copy = node.grid.clone()
            moved = grid_copy.move(action)
            if moved:
                grid_copy.random_cell()
                child = MCTSNode(grid_copy, parent=node, action=action)
                node.children.append(child)
                node = child

        # Simulation
        simulation_grid = node.grid.clone()
        for _ in range(simulation_depth):
            if not simulation_grid.can_move():
                break

            # Use neural network for guided simulation with some randomness
            if random.random() < 0.2:  # 80% neural network guidance
                try:
                    action = neural_network_policy(model, simulation_grid)
                    print("NN action: ", action)
                except:
                    action = random.choice(['up', 'down', 'left', 'right'])
            else:
                action = random.choice(['up', 'down', 'left', 'right'])

            moved = simulation_grid.move(action)
            if moved and simulation_grid.can_move():
                simulation_grid.random_cell()

        # Backpropagation
        score = simulation_grid.current_score

        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    # Choose the best action based on average score
    best_action = max(
        root.children, key=lambda child: child.total_score / child.visits
    ).action
    return best_action
