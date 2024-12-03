import numpy as np


def board_to_features(board_hex):
    """
    Converts a hexadecimal representation of a 2048 board to a 4x4 grid.

    Args:
        board_hex (str): Hexadecimal string representing the board state.

    Returns:
        np.ndarray: 4x4 grid of integers representing tile values.
    """
    board = int(board_hex, 16)
    features = []
    for _ in range(16):
        features.append(board & 0xF)
        board >>= 4
    features = features[::-1]
    grid = np.array(features).reshape(4, 4)
    return grid


def rotate_grid(grid, k):
    """
    Rotates a grid 90 degrees clockwise `k` times.

    Args:
        grid (np.ndarray): 4x4 grid of integers.
        k (int): Number of 90-degree rotations.

    Returns:
        np.ndarray: Rotated 4x4 grid.
    """
    return np.rot90(grid, k=k)


def adjust_action_rotation(action, k):
    """
    Adjusts an action based on the rotation applied to the grid.

    Args:
        action (int): Original action (0: Up, 1: Down, 2: Left, 3: Right).
        k (int): Number of 90-degree rotations.

    Returns:
        int: Adjusted action corresponding to the rotated grid.
    """
    rotation_mapping = {
        0: [0, 3, 1, 2],
        1: [1, 2, 0, 3],
        2: [2, 0, 3, 1],
        3: [3, 1, 2, 0],
    }
    return rotation_mapping[action][k % 4]


def flip_horizontal(grid):
    """
    Flips the grid horizontally.

    Args:
        grid (np.ndarray): 4x4 grid of integers.

    Returns:
        np.ndarray: Horizontally flipped grid.
    """
    return np.flipud(grid)


def flip_vertical(grid):
    """
    Flips the grid vertically.

    Args:
        grid (np.ndarray): 4x4 grid of integers.

    Returns:
        np.ndarray: Vertically flipped grid.
    """
    return np.fliplr(grid)


def transpose(grid):
    """
    Transposes the grid.

    Args:
        grid (np.ndarray): 4x4 grid of integers.

    Returns:
        np.ndarray: Transposed grid.
    """
    return np.transpose(grid)


def grid_to_tuple(grid):
    """
    Converts a 4x4 grid to a tuple representation.

    Args:
        grid (np.ndarray): 4x4 grid of integers.

    Returns:
        tuple: Tuple representation of the grid.
    """
    return tuple(grid.flatten())


def adjust_action(action, transformation):
    """
    Adjusts the action based on the transformation applied to the grid.

    Args:
        action (int): Original action (0: Up, 1: Down, 2: Left, 3: Right).
        transformation (str): Transformation type ('flip_horizontal', 
                              'flip_vertical', 'transpose').

    Returns:
        int: Adjusted action after applying the transformation.
    """
    if transformation == 'flip_horizontal':
        return 1 if action == 0 else 0 if action == 1 else action
    elif transformation == 'flip_vertical':
        return 3 if action == 2 else 2 if action == 3 else action
    elif transformation == 'transpose':
        if action == 0:
            return 2
        elif action == 1:
            return 3
        elif action == 2:
            return 0
        elif action == 3:
            return 1
    return action


def augment_data(data):
    """
    Augments the dataset with grid transformations and adjusts the actions 
    accordingly. Removes duplicate and inconsistent entries.

    Args:
        data (pd.DataFrame): DataFrame with columns 'board_state' and 'action'.

    Returns:
        dict: Unique data entries as a dictionary where keys are grid tuples 
              and values are actions.
    """
    data.dropna(inplace=True)
    augmented_data = []

    for _, row in data.iterrows():
        grid = board_to_features(row['board_state'])
        action = row['action']
        transformations = ['original', 'flip_horizontal',
                           'flip_vertical', 'transpose']

        for transform in transformations:
            if transform == 'original':
                grid_transformed = grid
                action_transformed = action
            elif transform == 'flip_horizontal':
                grid_transformed = flip_horizontal(grid)
                action_transformed = adjust_action(action, 'flip_horizontal')
            elif transform == 'flip_vertical':
                grid_transformed = flip_vertical(grid)
                action_transformed = adjust_action(action, 'flip_vertical')
            elif transform == 'transpose':
                grid_transformed = transpose(grid)
                action_transformed = adjust_action(action, 'transpose')

            augmented_data.append(
                {'grid': grid_transformed, 'action': action_transformed})

    unique_data = {}
    inconsistent_entries = 0

    for item in augmented_data:
        grid_key = grid_to_tuple(item['grid'])
        action = item['action']

        if grid_key in unique_data:
            if unique_data[grid_key] != action:
                inconsistent_entries += 1
                continue
        else:
            unique_data[grid_key] = action

    return unique_data
