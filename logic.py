import tkinter as tk
import tkinter.messagebox as messagebox

import sys
import random


class Grid:
    '''The data structure representation of the 2048 game.
    '''

    def __init__(self, n):
        self.size = n
        self.cells = self.generate_empty_grid()
        self.compressed = False
        self.merged = False
        self.moved = False
        self.current_score = 0

    def random_cell(self):
        cell = random.choice(self.retrieve_empty_cells())
        i, j = cell
        # Exponent 1 represents 2 (since 1 << 1 == 2)
        self.cells[i][j] = 1 if random.random(
        ) < 0.9 else 2  # Exponents 1 or 2

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
                    self.cells[i][j] += 1  # Increment exponent
                    self.cells[i][j + 1] = 0
                    # Update score: 1 shifted left by the new exponent
                    self.current_score += 1 << self.cells[i][j]
                    self.merged = True

    def found_2048(self):
        for i in range(self.size):
            for j in range(self.size):
                # Exponent for 2048 (since 1 << 11 == 2048)
                if self.cells[i][j] >= 11:
                    return True
        return False

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


class GamePanel:
    '''The GUI view class of the 2048 game showing via tkinter.'''
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

        # Load the best score from a file
        self.best_score = self.load_best_score()

        # Create a frame for the score labels
        self.score_frame = tk.Frame(self.root)
        self.score_frame.pack()

        # Label for the current score
        self.current_score_label = tk.Label(
            self.score_frame, text=f"Score: {self.grid.current_score}", font=('Verdana', 16)
        )
        self.current_score_label.pack(side=tk.LEFT, padx=10)

        # Label for the best score
        self.best_score_label = tk.Label(
            self.score_frame, text=f"Best: {self.best_score}", font=('Verdana', 16)
        )
        self.best_score_label.pack(side=tk.LEFT, padx=10)

        # Background frame for the grid
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
                    actual_value = 1 << exponent  # Calculate the actual value
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

        # Update the current score label
        self.current_score_label.configure(
            text=f"Score: {self.grid.current_score}")

        # Update the best score if necessary
        if self.grid.current_score > self.best_score:
            self.best_score = self.grid.current_score
            self.best_score_label.configure(text=f"Best: {self.best_score}")
            self.save_best_score()

    def returnScore(self, score):
        return score


class DummyPanel:
    '''A dummy panel for headless mode without GUI.'''

    def __init__(self, grid):
        self.grid = grid

    def paint(self):
        pass

    def returnScore(self, score):
        return score


class Game:
    '''The main game class which is the controller of the whole game.'''

    def __init__(self, grid, panel, strategy_function=None, delay=200, use_gui=True):
        self.grid = grid
        self.panel = panel
        self.start_cells_num = 2
        self.over = False
        self.won = False
        self.keep_playing = False
        self.strategy_function = strategy_function
        self.valid_actions = ['up', 'down', 'left', 'right']
        self.delay = delay
        self.use_gui = use_gui

    def is_game_terminated(self):
        return self.over or (self.won and (not self.keep_playing))

    def start(self):
        self.add_start_cells()
        self.panel.paint()
        if self.use_gui:
            # Start the auto-play loop with GUI
            self.panel.root.after(self.delay, self.auto_play)
            self.panel.root.mainloop()
        else:
            # Start the game loop without GUI
            while not self.is_game_terminated():
                self.auto_play()
            return self.grid.current_score

    def add_start_cells(self):
        for _ in range(self.start_cells_num):
            self.grid.random_cell()

    def can_move(self):
        return self.grid.has_empty_cells() or self.grid.can_merge()

    def move(self, direction):
        self.grid.clear_flags()
        if direction == 'up':
            self.up()
        elif direction == 'down':
            self.down()
        elif direction == 'left':
            self.left()
        elif direction == 'right':
            self.right()
        else:
            pass

        self.panel.paint()
        if self.grid.found_2048():
            self.you_win()
            if not self.keep_playing:
                return

        if self.grid.moved:
            self.grid.random_cell()

        self.panel.paint()
        if not self.can_move():
            self.over = True
            self.game_over()
            return

    def auto_play(self):
        if self.is_game_terminated():
            return self.grid.current_score

        if self.strategy_function:
            action = self.strategy_function(self.grid)
            if action not in self.valid_actions:
                print('Invalid action from strategy function.')
                return
        else:
            action = random.choice(self.valid_actions)

        self.move(action)

        if self.use_gui:
            # Schedule the next move with GUI
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
