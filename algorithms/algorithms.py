import random
from utils.helpers import MCTSNode
from algorithms.heuristics import empty_cell_heuristic, snake_heuristic, monotonicity_heuristic, smoothness_heuristic, merge_potential_heuristic, corner_max_tile_heuristic, combined_heuristic


def RandomStrategy(grid):
    """
    Randomly selects a valid action from 'up', 'down', 'left', 'right'.

    This strategy does not consider the state of the grid and is used as a baseline.
    """
    print(grid.cells)
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
