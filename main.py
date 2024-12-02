from algorithms import RandomStrategy, HeuristicStrategy, PolicyGradientStrategy, HeuristicStrategyWithLookahead
from logic import Grid, GamePanel, Game, DummyPanel
from utils import get_args


class Game2048:
    def __init__(self, strategy, delay, size=4, use_gui=True, heuristic=None):
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

    def start(self):
        game2048 = Game(
            self.grid,
            self.panel,
            strategy_function=self.strategy,
            delay=self.delay,
            use_gui=self.use_gui,
            heuristic=self.heuristic
        )
        result = game2048.start()
        return result


if __name__ == "__main__":
    args = get_args()
    use_gui = args.gui
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
