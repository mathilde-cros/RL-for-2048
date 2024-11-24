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


def PolicyGradient(grid):
    # implement policy gradient algorithm without passing
