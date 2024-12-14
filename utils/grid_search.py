import itertools
import numpy as np

from algorithms.algorithms import HeuristicStrategy, PolicyGradientStrategy, RandomStrategy
from algorithms.mcts import MCTSExpertStrategy, MCTSStrategy, MCTSStrategyHeuristic
from utils.logic import DummyPanel, Game, Grid


def run_grid_search_mcts(args):
    """Perform grid search over hyperparameters for MCTS."""

    param_grid = {
        'iterations': [50,  200],
        'simulation_depth': [50, 100],
        'exploration_weight': [0.5, 1]
    }

    best_params = None
    best_score = -np.inf
    results = []

    # generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v))
                          for v in itertools.product(*values)]

    for params in param_combinations:
        print(f"Testing parameters: {params}")
        scores = []

        for i in range(3):
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
