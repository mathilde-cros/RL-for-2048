from utils.helpers import get_args
from utils.logic import Grid, GamePanel, Game, DummyPanel
from algorithms.mcts import MCTSStrategy
from algorithms.algorithms import RandomStrategy, HeuristicStrategyWithLookahead, PolicyGradientStrategy
import os
import itertools
from torch import nn, optim


class Game2048:
    def __init__(self, strategy, delay, size=4, use_gui=True, heuristic=None, strategy_params=None):
        self.size = size
        self.grid = Grid(size)
        self.use_gui = use_gui
        if use_gui:
            self.panel = GamePanel(self.grid)
        else:
            self.panel = DummyPanel(self.grid)
        self.delay = delay
        self.strategy = strategy
        self.heuristic = heuristic
        self.strategy_params = strategy_params
        self.strategy_instance = None

    def start(self):
        if isinstance(self.strategy, type):  # Check if it's a class, not an instance
            self.strategy_instance = self.strategy(**self.strategy_params)
        else:
            self.strategy_instance = self.strategy
        game_instance = Game(
            self.grid,
            self.panel,
            strategy_function=self.strategy_instance,
            delay=self.delay,
            use_gui=self.use_gui,
            heuristic=self.heuristic
        )
        result = game_instance.start()
        # Retrieve the updated strategy instance
        self.strategy_instance = game_instance.strategy_instance
        return result

# The gridsearch for the neural netwokr tuning in Policy Gradient algorithm


def grid_search(args):
    hidden_sizes_grid = [[64], [128], [256], [512], [64, 64], [128, 64], [256, 128], [128, 128], [256, 256], [512, 256], [256, 128, 64], [512, 256, 128], [256, 256, 256], [
        512, 512, 512], [256, 256, 128], [512, 512, 256], [512, 512, 512], [512, 512, 512, 512], [1024, 512, 256], [1024, 1024, 512], [1024, 1024, 1024], [1024, 1024, 1024, 1024]]
    learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    activation_functions = [nn.ReLU, nn.LeakyReLU, nn.ELU, nn.Tanh]
    optimizers = [optim.Adam, optim.SGD, optim.RMSprop]
    gammas = [0.99, 0.95, 0.9]
    entropy_coefs = [0.0, 0.01, 0.05]

    num_runs = args.runs_per_combo
    results = []

    best_config = None
    best_result_overall = 0

    hyperparameter_combinations = list(itertools.product(
        hidden_sizes_grid,
        learning_rates,
        activation_functions,
        optimizers,
        gammas,
        entropy_coefs
    ))

    for idx, (hidden_sizes, lr, activation_fn, optimizer_cls, gamma, entropy_coef) in enumerate(hyperparameter_combinations):
        print(
            f"Testing configuration {idx+1}/{len(hyperparameter_combinations)}:")
        print(f"  hidden_sizes={hidden_sizes}, learning_rate={lr}, activation_fn={activation_fn.__name__}, optimizer={optimizer_cls.__name__}, gamma={gamma}, entropy_coef={entropy_coef}")
        scores = []
        for i in range(num_runs):
            # print(f"  Run {i+1}/{num_runs}")
            strategy_params = {
                'hidden_sizes': hidden_sizes,
                'learning_rate': lr,
                'activation_fn': activation_fn,
                'optimizer_cls': optimizer_cls,
                'gamma': gamma,
                'entropy_coef': entropy_coef
            }
            game = Game2048(
                strategy=PolicyGradientStrategy,
                delay=args.delay,
                use_gui=False,
                strategy_params=strategy_params
            )
            result = game.start()
            # Access the strategy instance to train
            strategy_instance = game.strategy_instance
            if strategy_instance is not None:
                strategy_instance.train()
            else:
                print("Warning: strategy_instance is None.")
            scores.append(result)
        avg_score = sum(scores) / len(scores)
        print(f"  Average score: {avg_score}")
        results.append({
            'hidden_sizes': hidden_sizes,
            'learning_rate': lr,
            'activation_fn': activation_fn.__name__,
            'optimizer': optimizer_cls.__name__,
            'gamma': gamma,
            'entropy_coef': entropy_coef,
            'average_score': avg_score,
            'scores': scores
        })

        # Save results to a file
        os.makedirs('results', exist_ok=True)
        with open('results/grid_search_results.txt', 'w') as f:
            for res in results:
                # f.write(f"Configuration:\n")
                # f.write(f"  hidden_sizes={res['hidden_sizes']}\n")
                # f.write(f"  learning_rate={res['learning_rate']}\n")
                # f.write(f"  activation_fn={res['activation_fn']}\n")
                # f.write(f"  optimizer={res['optimizer']}\n")
                # f.write(f"  gamma={res['gamma']}\n")
                # f.write(f"  entropy_coef={res['entropy_coef']}\n")
                # f.write(f"  Average score: {res['average_score']}\n")
                # f.write(f"  Best Score reached: {max(res['scores'])}\n")
                # f.write(f"  Scores: {res['scores']}\n")
                # f.write("\n")

                if max(res['scores']) > best_result_overall:
                    best_result_overall = max(res['scores'])
                    best_config = res

            f.write(f"Best configuration:\n")
            f.write(f"  hidden_sizes={best_config['hidden_sizes']}\n")
            f.write(f"  learning_rate={best_config['learning_rate']}\n")
            f.write(f"  activation_fn={best_config['activation_fn']}\n")
            f.write(f"  optimizer={best_config['optimizer']}\n")
            f.write(f"  gamma={best_config['gamma']}\n")
            f.write(f"  entropy_coef={best_config['entropy_coef']}\n")
            f.write(f"  Average score: {best_config['average_score']}\n")
            f.write(f"  Best Score reached: {max(best_config['scores'])}\n")
            f.write(f"  Scores: {best_config['scores']}\n")
            f.write("\n")


if __name__ == "__main__":
    args = get_args()
    use_gui = args.gui
    if args.grid_search:
        grid_search(args)

    runs = args.runs

    if args.algo == "random":
        strategy = RandomStrategy
    elif args.algo == "heuristic":
        strategy = HeuristicStrategyWithLookahead
    elif args.algo == "mcts":
        strategy = MCTSStrategy

    elif args.algo == "policy_gradient":
        strategy = PolicyGradientStrategy()
    else:

        print(f"Unknown algorithm: {args.algo}")
        exit(1)

    heuristic = args.heuristic
    scores = []
    for i in range(runs):
        print(f"Running game number {i+1}")
        game = Game2048(strategy, delay=args.delay,
                        use_gui=use_gui, heuristic=heuristic)
        result = game.start()
        scores.append(result)
        print("RESULT:", result)
        if runs > 1:
            mean_score = sum(scores) / len(scores)
            print(f"Mean score over {runs} runs: {mean_score}")
            print(f"Best score reached over {runs} runs: {max(scores)}")
            save_path = f'./results/{args.algo}_{heuristic}_mean_score.txt' if args.algo == "heuristic" else f'./results/{args.algo}_mean_score.txt'
            with open(save_path, 'w') as f:
                f.write(f"Mean score over {runs} runs: {mean_score} \n")
                f.write(f"Best score reached over {runs} runs: {max(scores)}")
