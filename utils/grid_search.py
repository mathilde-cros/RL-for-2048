import itertools
import numpy as np

from algorithms.algorithms import HeuristicStrategy, PolicyGradientStrategy, RandomStrategy
from algorithms.mcts import MCTSExpertStrategy, MCTSStrategy, MCTSStrategyHeuristic

import itertools
import numpy as np


from utils.logic import DummyPanel, Game, Grid


def manual_grid_search(grid_generator, heuristic, param_grid, games_per_param=5):
    """
    Perform a manual grid search over MCTS parameters by running multiple games for each parameter set.

    Args:
        grid_generator (callable): Function to generate a fresh initial grid state.
        heuristic (str): The heuristic to use in MCTSStrategyHeuristic.
        param_grid (dict): Dictionary containing parameter ranges to search.
                           Example: {'iterations': [50, 100], 'simulation_depth': [10, 50], 'exploration_weight': [1, 1.5]}
        games_per_param (int): Number of games to run for each parameter set.

    Returns:
        dict: The best parameter combination and corresponding average score.
    """
    best_params = None
    best_score = -np.inf
    results = []

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v))
                          for v in itertools.product(*values)]

    # Loop over all parameter combinations
    for params in param_combinations:
        scores = []
        print(f"Testing parameters: {params}")

        # Run multiple games for the current parameter set
        for game_idx in range(games_per_param):
            # Generate a fresh initial grid state
            grid = grid_generator()

            # Run MCTS with the current parameter combination
            action = MCTSStrategyHeuristic(
                grid,
                heuristic=heuristic,
                iterations=params['iterations'],
                simulation_depth=params['simulation_depth'],
                exploration_weight=params['exploration_weight']
            )

            # Simulate taking the action and calculate the final score
            grid_copy = grid.clone()
            while grid_copy.can_move():  # Play the game until it's over
                grid_copy.move(action)
                grid_copy.random_cell()
                action = MCTSStrategyHeuristic(
                    grid_copy,
                    heuristic=heuristic,
                    iterations=params['iterations'],
                    simulation_depth=params['simulation_depth'],
                    exploration_weight=params['exploration_weight']
                )
            scores.append(grid_copy.current_score)

        # Calculate average score for this parameter set
        avg_score = np.mean(scores)
        results.append((params, avg_score))

        # Update best parameters if this combination is better
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

        print(f"Scores: {scores}")
        print(f"Average Score: {avg_score}")

    # Print all results for analysis
    print("\nAll Results:")
    for params, score in results:
        print(f"Params: {params}, Average Score: {score}")

    return {'best_params': best_params, 'best_score': best_score}


def run_grid_search_mcts(args):

    param_grid = {
        'iterations': [50, 100, 200],
        'simulation_depth': [25, 50, 100],
        'exploration_weight': [0.5, 1]
    }

    best_params = None
    best_score = -np.inf
    results = []

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v))
                          for v in itertools.product(*values)]

    # Loop over all parameter combinations
    for params in param_combinations:
        print(f"Testing parameters: {params}")
        scores = []

        for i in range(5):
            print("Running game number", i+1, "with parameters", params)
            strategy = get_strategy_function(args.algo)
            grid = Grid(4)
            game_instance = Game(
                grid,
                DummyPanel(grid),
                strategy_function=strategy,
                delay=args.delay,
                use_gui=False,
                heuristic=args.heuristic,
                iterations=params['iterations'],
                simulation_depth=params['simulation_depth'],
                exploration_weight=params['exploration_weight'],
                grid_search=True
            )

            result = game_instance.start()
            scores.append(result)
            print("RESULT:", result)

        avg_score = np.mean(scores)
        results.append((params, avg_score))

        # best param and score
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    print("\nAll Results:", results)
    print("\nBest Parameters:", best_params)
    print("Best Score:", best_score)


def get_strategy_function(strategy_name):
    """Get the strategy function based on the name."""
    if strategy_name == 'random':
        return RandomStrategy
    elif strategy_name == 'heuristic':
        return HeuristicStrategy
    elif strategy_name == 'policy_gradient':
        return PolicyGradientStrategy
    elif strategy_name == 'mcts':
        return MCTSStrategy
    elif strategy_name == 'mcts_expert':
        return MCTSExpertStrategy
    elif strategy_name == 'mcts_heuristic':
        return MCTSStrategyHeuristic
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
