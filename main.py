from algorithms import RandomStrategy
from logic import Grid, GamePanel, Game, DummyPanel
from utils import get_args


class Game2048:
    def __init__(self, strategy, delay, size=4, use_gui=True):
        self.size = size
        self.grid = Grid(size)
        self.use_gui = use_gui
        if use_gui:
            self.panel = GamePanel(self.grid)
        else:
            self.panel = DummyPanel(self.grid)
        self.delay = delay
        self.strategy = strategy

    def start(self):
        game2048 = Game(
            self.grid,
            self.panel,
            strategy_function=self.strategy,
            delay=self.delay,
            use_gui=self.use_gui
        )
        result = game2048.start()
        return result


if __name__ == "__main__":
    args = get_args()
    use_gui = args.gui
    runs = args.runs

    if args.algo == "random":
        scores = []
        for _ in range(runs):
            game = Game2048(RandomStrategy, delay=args.delay, use_gui=use_gui)
            result = game.start()
            scores.append(result)
            print("RESULT:", result)
        if runs > 1:
            mean_score = sum(scores) / len(scores)
            print(f"Mean score over {runs} runs: {mean_score}")
            with open('./results/random_strategy_mean_score.txt', 'w') as f:
                f.write(f"Mean score over {runs} runs: {mean_score}")