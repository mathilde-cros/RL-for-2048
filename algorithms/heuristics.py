

def empty_cell_heuristic(grid):
    """
    Evaluates the grid based on the number of empty cells and the maximum tile.

    Heuristic Value = (Number of Empty Cells) + (Maximum Tile Value * 4)

    This heuristic encourages moves that keep the grid emptier (providing more space
    for future moves) and increase the maximum tile on the grid.
    """
    empty_cells = len(grid.retrieve_empty_cells())
    max_tile = max(max(row) for row in grid.cells)
    heuristic_value = empty_cells + (max_tile * 4)
    return heuristic_value


def snake_heuristic(grid):
    """
    Evaluates the grid based on a 'snake' pattern weight matrix.

    The weight matrix is designed to guide the placement of high-value tiles in a snake-like
    pattern starting from one corner, typically to maximize merging opportunities.

    Heuristic Value = Sum of (Tile Value * Corresponding Weight)
    """
    weights = [
        [2, 4, 8, 16],
        [256, 128, 64, 32],
        [512, 1024, 2048, 4096],
        [65536, 32768, 16384, 8192]
    ]
    score = 0
    for i in range(grid.size):
        for j in range(grid.size):
            score += grid.cells[i][j] * weights[i][j]
    return score


def monotonicity_heuristic(grid):
    """
    Evaluates the grid based on how monotonic the rows and columns are.

    This heuristic measures the monotonicity (either increasing or decreasing) of the
    tile values in both rows and columns, encouraging smooth gradients.

    Heuristic Value = Higher negative value indicates better monotonicity.
    """
    totals = [0, 0, 0, 0]
    for i in range(grid.size):
        for j in range(grid.size - 1):
            current = grid.cells[i][j]
            next = grid.cells[i][j + 1]
            if current > next:
                totals[0] += next - current
            else:
                totals[1] += current - next
    for j in range(grid.size):
        for i in range(grid.size - 1):
            current = grid.cells[i][j]
            next = grid.cells[i + 1][j]
            if current > next:
                totals[2] += next - current
            else:
                totals[3] += current - next
    return max(totals[0], totals[1]) + max(totals[2], totals[3])


def smoothness_heuristic(grid):
    """
    Evaluates the grid based on how 'smooth' it is, favoring grids where neighboring
    tiles have similar values.

    This heuristic penalizes large differences between neighboring tiles, which can
    hinder merging opportunities.

    Heuristic Value = Negative sum of absolute differences between neighboring tiles.
    """
    smoothness = 0
    for i in range(grid.size):
        for j in range(grid.size):
            value = grid.cells[i][j]
            if value != 0:
                # Compare with right neighbor
                if j + 1 < grid.size and grid.cells[i][j + 1] != 0:
                    smoothness -= abs(value - grid.cells[i][j + 1])
                # Compare with bottom neighbor
                if i + 1 < grid.size and grid.cells[i + 1][j] != 0:
                    smoothness -= abs(value - grid.cells[i + 1][j])
    return smoothness


def merge_potential_heuristic(grid):
    """
    Evaluates the grid based on the number of potential merges available.

    This heuristic counts the number of adjacent tiles with the same value,
    indicating immediate merging opportunities.

    Heuristic Value = Number of potential merges.
    """
    merge_potential = 0
    for i in range(grid.size):
        for j in range(grid.size):
            value = grid.cells[i][j]
            if value != 0:
                # Check right neighbor
                if j + 1 < grid.size and grid.cells[i][j + 1] == value:
                    merge_potential += 1
                # Check bottom neighbor
                if i + 1 < grid.size and grid.cells[i + 1][j] == value:
                    merge_potential += 1
    return merge_potential


def corner_max_tile_heuristic(grid):
    """
    Evaluates the grid by checking if the maximum tile is located in one of the corners.

    This heuristic rewards grids where the highest tile is in a corner, a strategy
    that can help in organizing tiles for better merges.

    Heuristic Value = Max Tile Value (if in corner) or Negative Max Tile Value (if not).
    """
    max_tile = max(max(row) for row in grid.cells)
    corner_positions = [
        (0, 0),
        (0, grid.size - 1),
        (grid.size - 1, 0),
        (grid.size - 1, grid.size - 1)
    ]
    for x, y in corner_positions:
        if grid.cells[x][y] == max_tile:
            return max_tile
    return -max_tile  # Penalize if the max tile is not in a corner


def homogeneity_heuristic(grid):
    """
    Evaluates the grid based on how homogeneous the values of neighboring tiles are.

    The heuristic rewards grids where neighboring tiles have similar values, encouraging smooth gradients that
    make merging easier in future moves.

    Heuristic Value = Higher value if neighboring tiles have similar values.
    """
    homogeneity_score = 0
    for i in range(grid.size):
        for j in range(grid.size):
            value = grid.cells[i][j]
            if value != 0:
                # Compare with right neighbor
                if j + 1 < grid.size and grid.cells[i][j + 1] != 0:
                    homogeneity_score -= abs(value - grid.cells[i][j + 1])
                # Compare with bottom neighbor
                if i + 1 < grid.size and grid.cells[i + 1][j] != 0:
                    homogeneity_score -= abs(value - grid.cells[i + 1][j])

    # A higher score is better, so we return the negative of the differences
    return -homogeneity_score


def combined_heuristic(grid):
    """
    Evaluates the grid by combining multiple heuristics into a single score.

    The combined score is calculated as a weighted sum of the individual heuristic scores.
    Adjust the weights to prioritize different aspects of the game.

    Returns:
    - A numerical value representing the combined heuristic score.
    """
    # Calculate individual heuristic scores
    empty_cells_score = len(grid.retrieve_empty_cells())
    monotonicity_score = monotonicity_heuristic(grid)
    merge_potential_score = merge_potential_heuristic(grid)
    corner_max_tile_score = corner_max_tile_heuristic(grid)

    # Weights for each heuristic (adjust these weights based on experimentation)
    weights = {
        'empty_cells': 300,
        'monotonicity': 500,
        'merge_potential': 100,
        'corner_max_tile': 300
    }

    # Combined heuristic score
    combined_score = (
        empty_cells_score * weights['empty_cells'] +
        monotonicity_score * weights['monotonicity'] +
        merge_potential_score * weights['merge_potential']
        + corner_max_tile_score * weights['corner_max_tile']
    )

    return combined_score
