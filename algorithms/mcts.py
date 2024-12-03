import random
from algorithms.heuristics import combined_heuristic
from utils.helpers import MCTSNode
### Monte Carlo Tree Search (MCTS) ###


def MCTSStrategy(grid, heuristic, iterations=50, simulation_depth=200, exploration_weight=0.4):
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

        # Backpropagation
        if not heuristic:
            heuristic_fct = combined_heuristic
            score = heuristic_fct(simulation_grid)
        else:
            score = simulation_grid.current_score

        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent

    # print for each child the number of visits
    for child in root.children:
        print(f"Action: {child.action}, Visits: {child.visits}")

    # print score for each child
    for child in root.children:
        print(
            f"Action: {child.action}, Score: {child.total_score / child.visits}")
    # Choose the best action based on average score
    best_action = max(
        root.children, key=lambda child: child.total_score / child.visits
    ).action
    return best_action
