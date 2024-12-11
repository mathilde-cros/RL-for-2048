import argparse
import numpy as np


description = "2048 Game CLI"
algo_list = ["random", "heuristic",
             "heuristic_lookahead", "policy_gradient", "mcts"]
heuristic_list = ["empty-cells", "snake",
                  "monotonic", "smoothness", "merge-potential", "corner-max-tile", "combined", "advanced"]


def get_args():
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--algo", choices=algo_list, required=True, help="algorithm"
    )

    parser.add_argument(
        "--delay", type=int, default=100, help="delay between moves"
    )

    parser.add_argument('--gui', action='store_true',
                        help='Enable GUI', default=False)
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs for simulation')

    parser.add_argument('--heuristic', choices=heuristic_list,
                        required=False, default=heuristic_list[0])

    parser.add_argument('--grid_search', action='store_true',
                        help='Perform grid search over hyperparameters', default=False)
    parser.add_argument('--runs_per_combo', type=int, default=3,
                        help='Number of runs per hyperparameter combination during grid search')

    parser.set_defaults(**get_default_args() or {})

    args = parser.parse_args()
    return args


def get_default_args() -> dict:
    pass


class MCTSNode:
    def __init__(self, grid, parent=None, action=None):
        self.grid = grid
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.total_score = 0

    def is_terminal_node(self):
        return not self.grid.can_move()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, exploration_weight=2):
        """Select the best child using UCT (Upper Confidence Bound for Trees)."""
        scores = [
            (
                child.total_score / child.visits +  # Exploitation
                exploration_weight * \
                np.sqrt(np.log(self.visits) / child.visits)  # Exploration
            )
            for child in self.children
        ]
        return self.children[np.argmax(scores)]

    @property
    def untried_actions(self):
        tried_actions = {child.action for child in self.children}
        return [action for action in ['up', 'down', 'left', 'right'] if action not in tried_actions and self.grid.clone().move(action)]
