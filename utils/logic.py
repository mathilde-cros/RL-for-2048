import tkinter as tk
import tkinter.messagebox as messagebox
import copy

import sys
import random


class Grid:
    """The Grid class represents the 2048 game grid."""

    def __init__(self, n):
        self.size = n
        self.cells = self.generate_empty_grid()
        self.compressed = False
        self.merged = False
        self.moved = False
        self.current_score = 0

    def print_grid(self):
        print('-' * 40)
        for i in range(self.size):
            for j in range(self.size):
                if self.cells[i][j] == 0:
                    print('0\t', end='')
                else:
                    # Use bit shift to get actual value
                    print(f'{1 << self.cells[i][j]}\t', end='')
            print()
        print('-' * 40)

    def random_cell(self):
        cell = random.choice(self.retrieve_empty_cells())
        i, j = cell
        # exponent 1 represents 2 (since 1 << 1 == 2)
        self.cells[i][j] = 1 if random.random(
        ) < 0.9 else 2

    def retrieve_empty_cells(self):
        empty_cells = []
        for i in range(self.size):
            for j in range(self.size):
                if self.cells[i][j] == 0:
                    empty_cells.append((i, j))
        return empty_cells

    def generate_empty_grid(self):
        return [[0] * self.size for i in range(self.size)]

    def transpose(self):
        self.cells = [list(t) for t in zip(*self.cells)]

    def reverse(self):
        for i in range(self.size):
            start = 0
            end = self.size - 1
            while start < end:
                self.cells[i][start], self.cells[i][end] = \
                    self.cells[i][end], self.cells[i][start]
                start += 1
                end -= 1

    def clear_flags(self):
        self.compressed = False
        self.merged = False
        self.moved = False

    def left_compress(self):
        self.compressed = False
        new_grid = self.generate_empty_grid()
        for i in range(self.size):
            count = 0
            for j in range(self.size):
                if self.cells[i][j] != 0:
                    new_grid[i][count] = self.cells[i][j]
                    if count != j:
                        self.compressed = True
                    count += 1
        self.cells = new_grid

    def left_merge(self):
        self.merged = False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.cells[i][j] == self.cells[i][j + 1] and self.cells[i][j] != 0:
                    self.cells[i][j] += 1
                    self.cells[i][j + 1] = 0
                    self.current_score += 1 << self.cells[i][j]
                    self.merged = True

    def found_2048(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.cells[i][j] >= 11:
                    return True
        return False

    def place_tile(self, cell, value):
        i, j = cell
        self.cells[i][j] = value

    def has_empty_cells(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.cells[i][j] == 0:
                    return True
        return False

    def can_merge(self):
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.cells[i][j] == self.cells[i][j + 1]:
                    return True
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.cells[i][j] == self.cells[i + 1][j]:
                    return True
        return False

    def set_cells(self, cells):
        self.cells = cells

    def clone(self):
        """ Return a copy of the grid """
        new_grid = Grid(self.size)
        new_grid.cells = copy.deepcopy(self.cells)
        new_grid.current_score = self.current_score
        return new_grid

    def move(self, direction):
        '''Apply a move to the grid. Returns True if the grid has changed.'''
        self.clear_flags()
        if direction == 'up':
            self.up()
        elif direction == 'down':
            self.down()
        elif direction == 'left':
            self.left()
        elif direction == 'right':
            self.right()
        else:
            return False
        return self.moved

    def can_move(self):
        return self.has_empty_cells() or self.can_merge()

    def can_move_action(self, action):
        if action not in ['up', 'down', 'left', 'right']:
            return False
        cloned_grid = self.clone()
        moved = cloned_grid.move(action)

        return moved

    def up(self):
        self.transpose()
        self.left_compress()
        self.left_merge()
        self.moved = self.compressed or self.merged
        self.left_compress()
        self.transpose()

    def down(self):
        self.transpose()
        self.reverse()
        self.left_compress()
        self.left_merge()
        self.moved = self.compressed or self.merged
        self.left_compress()
        self.reverse()
        self.transpose()

    def left(self):
        self.left_compress()
        self.left_merge()
        self.moved = self.compressed or self.merged
        self.left_compress()

    def right(self):
        self.reverse()
        self.left_compress()
        self.left_merge()
        self.moved = self.compressed or self.merged
        self.left_compress()
        self.reverse()


class GamePanel:
    """The GamePanel class represents the GUI of the 2048 game."""
    CELL_PADDING = 10
    BACKGROUND_COLOR = '#92877d'
    EMPTY_CELL_COLOR = '#9e948a'
    CELL_BACKGROUND_COLOR_DICT = {
        '2': '#eee4da',
        '4': '#ede0c8',
        '8': '#f2b179',
        '16': '#f59563',
        '32': '#f67c5f',
        '64': '#f65e3b',
        '128': '#edcf72',
        '256': '#edcc61',
        '512': '#edc850',
        '1024': '#edc53f',
        '2048': '#edc22e',
        'beyond': '#3c3a32'
    }
    CELL_COLOR_DICT = {
        '2': '#776e65',
        '4': '#776e65',
        '8': '#f9f6f2',
        '16': '#f9f6f2',
        '32': '#f9f6f2',
        '64': '#f9f6f2',
        '128': '#f9f6f2',
        '256': '#f9f6f2',
        '512': '#f9f6f2',
        '1024': '#f9f6f2',
        '2048': '#f9f6f2',
        'beyond': '#f9f6f2'
    }
    FONT = ('Verdana', 24, 'bold')
    UP_KEYS = ('w', 'W', 'Up')
    LEFT_KEYS = ('a', 'A', 'Left')
    DOWN_KEYS = ('s', 'S', 'Down')
    RIGHT_KEYS = ('d', 'D', 'Right')

    def __init__(self, grid):
        self.grid = grid
        self.root = tk.Tk()
        if sys.platform == 'win32':
            self.root.iconbitmap('2048.ico')
        self.root.title('2048')
        self.root.resizable(False, False)

        self.best_score = self.load_best_score()

        self.score_frame = tk.Frame(self.root)
        self.score_frame.pack()

        self.current_score_label = tk.Label(
            self.score_frame, text=f"Score: {self.grid.current_score}", font=('Verdana', 16)
        )
        self.current_score_label.pack(side=tk.LEFT, padx=10)

        self.best_score_label = tk.Label(
            self.score_frame, text=f"Best: {self.best_score}", font=('Verdana', 16)
        )
        self.best_score_label.pack(side=tk.LEFT, padx=10)

        self.background = tk.Frame(self.root, bg=GamePanel.BACKGROUND_COLOR)
        self.cell_labels = []
        for i in range(self.grid.size):
            row_labels = []
            for j in range(self.grid.size):
                label = tk.Label(
                    self.background,
                    text='',
                    bg=GamePanel.EMPTY_CELL_COLOR,
                    justify=tk.CENTER,
                    font=GamePanel.FONT,
                    width=4,
                    height=2
                )
                label.grid(row=i, column=j, padx=10, pady=10)
                row_labels.append(label)
            self.cell_labels.append(row_labels)
        self.background.pack(side=tk.TOP)

    def load_best_score(self):
        try:
            with open('best_score.txt', 'r') as f:
                return int(f.read())
        except (FileNotFoundError, ValueError):
            return 0

    def save_best_score(self):
        with open('best_score.txt', 'w') as f:
            f.write(str(self.best_score))

    def paint(self):
        for i in range(self.grid.size):
            for j in range(self.grid.size):
                exponent = self.grid.cells[i][j]
                if exponent == 0:
                    self.cell_labels[i][j].configure(
                        text='',
                        bg=GamePanel.EMPTY_CELL_COLOR
                    )
                else:
                    actual_value = 1 << exponent
                    cell_text = str(actual_value)
                    if actual_value > 2048:
                        bg_color = GamePanel.CELL_BACKGROUND_COLOR_DICT.get(
                            'beyond')
                        fg_color = GamePanel.CELL_COLOR_DICT.get('beyond')
                    else:
                        bg_color = GamePanel.CELL_BACKGROUND_COLOR_DICT.get(
                            cell_text)
                        fg_color = GamePanel.CELL_COLOR_DICT.get(cell_text)
                    self.cell_labels[i][j].configure(
                        text=cell_text,
                        bg=bg_color,
                        fg=fg_color
                    )

        self.current_score_label.configure(
            text=f"Score: {self.grid.current_score}")

        if self.grid.current_score > self.best_score:
            self.best_score = self.grid.current_score
            self.best_score_label.configure(text=f"Best: {self.best_score}")
            self.save_best_score()

    def returnScore(self, score):
        return score


class DummyPanel:
    """The DummyPanel class is a dummy panel for the game."""

    def __init__(self, grid):
        self.grid = grid

    def paint(self):
        pass

    def returnScore(self, score):
        return score


class Game:
    """The Game class represents the 2048 game."""

    def __init__(self, grid, panel, strategy_function=None, delay=200, use_gui=True, heuristic=None, model=None, iterations=50, simulation_depth=100, exploration_weight=1, grid_search=False):
        self.grid = grid
        self.panel = panel
        self.start_cells_num = 2
        self.over = False
        self.won = False
        self.keep_playing = False
        self.strategy_function = strategy_function
        self.strategy_instance = strategy_function
        self.valid_actions = ['up', 'down', 'left', 'right']
        self.delay = delay
        self.use_gui = use_gui
        self.heuristic = heuristic
        self.previous_score = 0
        self.model = model
        self.iterations = iterations
        self.simulation_depth = simulation_depth
        self.exploration_weight = exploration_weight
        self.grid_search = grid_search
        self.number_moves = 0

    def is_game_terminated(self):
        return self.over or (self.won and (not self.keep_playing))

    def start(self):
        self.add_start_cells()
        self.panel.paint()
        if self.use_gui:
            # start the auto-play loop with GUI
            self.panel.root.after(self.delay, self.auto_play)
            self.panel.root.mainloop()
            return (self.grid.current_score, self.number_moves)
        else:
            # start the game loop without GUI
            while not self.is_game_terminated():
                self.auto_play()
            return (self.grid.current_score, self.number_moves)

    def add_start_cells(self):
        for _ in range(self.start_cells_num):
            self.grid.random_cell()

    def can_move(self):
        return self.grid.has_empty_cells() or self.grid.can_merge()

    def move(self, direction):
        moved = self.grid.move(direction)
        reward = 0

        score_increase = self.grid.current_score - self.previous_score
        reward += score_increase

        # if self.grid.found_2048():
        #     self.you_win()
        #     if not self.keep_playing:
        #         return
        self.panel.paint()

        if moved:
            self.grid.random_cell()

        self.panel.paint()
        if not self.can_move():
            self.over = True
            self.game_over()
            return

        self.previous_score = self.grid.current_score

        if hasattr(self.strategy_instance, 'rewards'):
            self.strategy_instance.rewards.append(reward)

    def auto_play(self):
        self.number_moves += 1
        if self.is_game_terminated():
            return self.grid.current_score

        if self.strategy_function:

            if self.grid_search:
                action = self.strategy_function(
                    self.grid, self.heuristic, self.iterations, self.simulation_depth, self.exploration_weight)
            elif self.strategy_function.__name__ == 'MCTSExpertStrategy':
                action = self.strategy_function(
                    self.grid, self.model, self.heuristic)
            elif self.strategy_function.__name__ == 'ExpertAgent':
                if not self.model:
                    return 'No model found'
                action = self.strategy_function(self.grid, self.model)
            elif self.strategy_function.__name__ == "MCTSStrategyHeuristic":
                action = self.strategy_function(self.grid, self.heuristic)
            elif self.heuristic and self.strategy_function.__code__.co_argcount >= 2:
                action = self.strategy_function(self.grid, self.heuristic)
            else:
                action = self.strategy_function(self.grid)
            if action not in self.valid_actions:
                print('Invalid action from strategy function.')
                return
        else:
            action = random.choice(self.valid_actions)
        self.move(action)

        if self.use_gui:
            self.panel.root.after(self.delay, self.auto_play)

    def you_win(self):
        if not self.won:
            self.won = True
            if self.use_gui:
                if messagebox.askyesno('2048', 'You Win!\nAre you going to continue the 2048 game?'):
                    self.keep_playing = True
            else:
                self.keep_playing = False

    def game_over(self):
        if self.use_gui:
            messagebox.showinfo('2048', 'Oops!\nGame over!')
        else:
            pass

    def up(self):
        self.grid.transpose()
        self.grid.left_compress()
        self.grid.left_merge()
        self.grid.moved = self.grid.compressed or self.grid.merged
        self.grid.left_compress()
        self.grid.transpose()

    def left(self):
        self.grid.left_compress()
        self.grid.left_merge()
        self.grid.moved = self.grid.compressed or self.grid.merged
        self.grid.left_compress()

    def down(self):
        self.grid.transpose()
        self.grid.reverse()
        self.grid.left_compress()
        self.grid.left_merge()
        self.grid.moved = self.grid.compressed or self.grid.merged
        self.grid.left_compress()
        self.grid.reverse()
        self.grid.transpose()

    def right(self):
        self.grid.reverse()
        self.grid.left_compress()
        self.grid.left_merge()
        self.grid.moved = self.grid.compressed or self.grid.merged
        self.grid.left_compress()
        self.grid.reverse()
