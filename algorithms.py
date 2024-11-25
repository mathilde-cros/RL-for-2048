import random


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
