from algorithms import RandomStrategy, HeuristicStrategy, PolicyGradientStrategy, HeuristicStrategyWithLookahead
from logic import Grid, GamePanel, Game, DummyPanel
from utils import get_args
import itertools
import os

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

    def start(self):
        if self.strategy == PolicyGradientStrategy:
            strategy_instance = self.strategy(**self.strategy_params)
        else:
            strategy_instance = self.strategy
        game2048 = Game(
            self.grid,
            self.panel,
            strategy_function=strategy_instance,
            delay=self.delay,
            use_gui=self.use_gui,
            heuristic=self.heuristic
        )
        result = game2048.start()
        return result
    
## The gridsearch for the neural netwokr tuning in Policy Gradient algorithm
def grid_search(args):
    hidden_sizes_grid = [[64], [128], [256], [128, 64], [256, 128], [512, 256], [512, 256, 128], [1024, 512, 256], [1024, 512, 256, 128]]
    learning_rates = [1e-2, 1e-3, 1e-4]
    num_runs = args.runs_per_combo

    results = []

    for hidden_sizes, lr in itertools.product(hidden_sizes_grid, learning_rates):
        print(f"Testing configuration: hidden_sizes={hidden_sizes}, learning_rate={lr}")
        scores = []
        for i in range(num_runs):
            print(f" Run {i+1}/{num_runs}")
            strategy_params = {
                'hidden_sizes': hidden_sizes,
                'learning_rate': lr
            }
            game = Game2048(
                strategy=PolicyGradientStrategy,
                delay=args.delay,
                use_gui=False,
                strategy_params=strategy_params
            )
            result = game.start()
            scores.append(result)
        avg_score = sum(scores) / len(scores)
        print(f" Average score: {avg_score}")
        results.append({
            'hidden_sizes': hidden_sizes,
            'learning_rate': lr,
            'average_score': avg_score,
            'scores': scores
        })

    # Save results to a file
    os.makedirs('results', exist_ok=True)
    with open('results/grid_search_results.txt', 'w') as f:
        for res in results:
            f.write(f"Configuration: hidden_sizes={res['hidden_sizes']}, learning_rate={res['learning_rate']}\n")
            f.write(f" Average score: {res['average_score']}\n")
            f.write(f" Scores: {res['scores']}\n")
            f.write("\n")


if __name__ == "__main__":
    args = get_args()
    use_gui = args.gui
    
    if args.grid_search:
        grid_search(args)
    else:
        runs = args.runs

        if args.algo == "random":
            strategy = RandomStrategy
        elif args.algo == "heuristic":
            strategy = HeuristicStrategyWithLookahead
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
 