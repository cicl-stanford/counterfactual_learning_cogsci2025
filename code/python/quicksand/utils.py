import os
import ast 
import copy
import random
import numpy as np
from collections import deque
from pathlib import Path
from inspect import signature
from typing import Type, Union
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import heapq

def bernoulli(p):
    return np.random.rand() < p

def beta(a, b):
    return np.random.beta(a, b)

def forkable(cls: Type) -> Type:
    """
    A class decorator that augments classes to automatically store the parameters passed to 
    the __init__ method of the class. It also adds a method 'fork' that can use 
    these parameters to create a new instance of the class, with optional overrides.

    Args:
        cls (Type): The class to decorate.

    Returns:
        Type: The decorated class with additional functionality.
    """
    orig_init = cls.__init__
    
    # Define a new __init__ that wraps the original
    def new_init(self, *args, **kwargs):
        # Call the original __init__
        orig_init(self, *args, **kwargs)

        # Binding arguments to parameter names
        bound_args = signature(orig_init).bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # Store init params excluding 'self'
        self.init_params = {k: copy.deepcopy(v) for k, v in bound_args.arguments.items() if k != 'self'}

    # Replace the original __init__ with the new one
    cls.__init__ = new_init

    def fork(self, **overrides):
        """
        Creates a new instance of the class using the stored init parameters,
        optionally overriding them with specified values.

        Args:
            **overrides: Keyword arguments that override the stored init parameters.

        Returns:
            An instance of the class initialized with the combined parameters.
        """
        # Combine init_params with overrides
        params = {**self.init_params, **overrides}
        return cls(**params)

    # Add fork to the class
    cls.fork = fork

    return cls


def flag_last(iterable):
    """
    Iterator adapter that flags the last item in an iterable.
    """
    it = iter(iterable)
    prev = next(it)
    for item in it:
        yield prev, False
        prev = item
    yield prev, True

def softmax(x):
    """
    Compute the softmax of a 1D array.
    """
    e_x = np.exp(x - np.max(x))  # Stability improvement: subtract max to avoid overflow
    return e_x / e_x.sum(axis=0)

def moving_average(values, n):
    """
    Calculate the moving average of a list with a specified window size.

    Args:
        values (list of float): The list of numbers.
        n (int): The number of elements in the window to average over.

    Returns:
        list of float: A list containing the moving averages.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if len(values) < n:
        return []  # Return an empty list if the window size is larger than the list size

    # Initialize the sum of the first window and create the result list
    window_sum = sum(values[:n])
    moving_averages = [window_sum / n]

    # Compute the moving average for each window
    for i in range(n, len(values)):
        window_sum += values[i] - values[i - n]
        moving_averages.append(window_sum / n)

    return moving_averages

def compute_experience_and_estimates(env):
    """
    Compute the experience and probability estimates based on the agent's world model.
    
    Args:
        env (Environment): An environment object containing the world model and agent information.
    
    Returns:
        tuple: A tuple containing two numpy arrays: 'experience' and 'model_estimates'.
               'experience' shows the total amount of beta-distribution observations for each cell.
               'model_estimates' contains the estimated probability of quicksand for each cell.
    """
    width = env.world_generator.environment_spec['metadata']['width']
    height = env.world_generator.environment_spec['metadata']['height']
    
    experience = np.zeros((width, height))
    model_estimates = np.zeros((width, height))
    
    for (x, y), value in env.agent.world_model.environment_spec['states'].items():
        if isinstance(value, tuple):
            alpha, beta = value
            total_observations = alpha + beta
            experience[x, y] = total_observations - 10  # Adjust by the initial prior counts
            model_estimates[x, y] = alpha / total_observations
            
    return experience, model_estimates

### sample start location ###
def get_tiles_at_distance(tiles, dist, goal):
    tiles_at_dist = []
    for tile in tiles:
        if abs(tile[0] - goal[0]) + abs(tile[1] - goal[1]) == dist:
            tiles_at_dist.append(tile)
    return tiles_at_dist

def sample_start_location(goal, dist, width, height, walls=[]):
    """Sample a starting location dist points away from the goal that is not a wall."""
    all_tiles = [(j, i) for i in range(height) for j in range(width) if (j, i) not in walls]
    tiles_at_dist = get_tiles_at_distance(all_tiles, dist, goal)
    # use path_exists to filter tiles_at_dist
    valid_tiles = [tile for tile in tiles_at_dist if path_exists(tile, goal, walls, width, height)]
    if valid_tiles:
        return random.choice(valid_tiles)
    else:
        raise ValueError(f"No valid tiles at distance {dist} from goal {goal}")

### Matrix functions ###
def normalize_ranks(diagonal):
    """
    Normalize the ranks of values in a diagonal array to be between 0 and 1.
    
    Args:
        diagonal (np.array): The diagonal array to normalize.
    
    Returns:
        np.array: Normalized ranks of the diagonal elements.
    """
    argsorted = np.argsort(diagonal)
    ranks = np.argsort(argsorted)
    normalized_ranks = ranks / ranks.max()
    return normalized_ranks

def get_neighbors(matrix, x, y):
    """
    Get the values of the four direct neighbors (up, down, left, right) of a cell in a matrix.
    
    Args:
        matrix (np.array): Input 2D matrix.
        x (int): X-coordinate of the cell.
        y (int): Y-coordinate of the cell.
    
    Returns:
        np.array: Values of the four neighbors, uses -25 for out-of-bounds locations.
    """
    neighbors = np.zeros(4) - 25  # Initialize with -25 for out-of-bound values
    if x > 0:
        neighbors[0] = matrix[x-1, y]  # Top
    if x < matrix.shape[0] - 1:
        neighbors[1] = matrix[x+1, y]  # Bottom
    if y > 0:
        neighbors[2] = matrix[x, y-1]  # Left
    if y < matrix.shape[1] - 1:
        neighbors[3] = matrix[x, y+1]  # Right
    return neighbors

def rank_matrix_diagonals(matrix, anti_diagonal=False):
    """
    Rank the diagonals or anti-diagonals of a matrix by normalizing their ranks between 0 and 1.
    
    Args:
        matrix (np.array): The input matrix to rank.
        anti_diagonal (bool): If True, rank the anti-diagonals instead of the main diagonals.
    
    Returns:
        np.array: A matrix with ranked diagonals or anti-diagonals.
    """
    n = matrix.shape[0]
    ranked_matrix = np.zeros_like(matrix, dtype=float)
    target_matrix = np.fliplr(matrix) if anti_diagonal else matrix

    for offset in range(-n + 1, n):
        diag = np.diag(target_matrix, k=offset)
        if len(diag) > 1:
            ranked_diag = np.diag(normalize_ranks(diag), k=offset)
            if anti_diagonal:
                ranked_diag = np.fliplr(ranked_diag)
            ranked_matrix += ranked_diag
    
    return ranked_matrix

def clamp_matrix_values(matrix, min_p, max_p):
    """
    Clamps the values in a matrix to the specified percentile limits.
    
    Args:
    - matrix (np.array): The input matrix to clamp.
    - min_p (float): The lower percentile limit.
    - max_p (float): The upper percentile limit.
    
    Returns:
    - np.array: The clamped matrix.
    """
    # Compute the percentile values
    min_val = np.percentile(matrix, min_p)
    max_val = np.percentile(matrix, max_p)
    
    # Clip the matrix values to these percentiles
    clamped_matrix = np.clip(matrix, min_val, max_val)
    return clamped_matrix

def compute_neighbor_difference(value_matrix):
    """
    Compute the difference between each cell's value and the average value of its neighbors.
    
    Args:
        value_matrix (np.array): A 2D array of values representing a grid.
    
    Returns:
        np.array: A 2D array where each cell's value is adjusted by the average of its neighbors.
    """
    difference_matrix = np.zeros_like(value_matrix)
    rows, cols = value_matrix.shape
    for x in range(rows):
        for y in range(cols):
            neighbors = [
                value_matrix[x + dx, y + dy]
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= x + dx < rows and 0 <= y + dy < cols
            ]
            difference_matrix[x, y] = value_matrix[x, y] - np.mean(neighbors)
    return difference_matrix

### Spec functions ###    
def convert_spec_from_json(spec):
    spec_parsed = copy.deepcopy(spec)
    spec_parsed['states'] = {ast.literal_eval(k): v for k,v in spec['states'].items()}
    return spec_parsed

def convert_stimuli_from_json(stimuli):
    stimuli_parsed = copy.deepcopy(stimuli)
    stimuli_parsed['worlds'] = [convert_spec_from_json(world) for world in stimuli['worlds']]
    return stimuli_parsed

def make_grid_from_spec(spec):
    """Make a grid from a given spec."""
    grid = np.zeros((spec['metadata']['height'], spec['metadata']['width']))
    for k, v in spec['states'].items():
        if v == 'wall':
            grid[k[1], k[0]] = 0
        else:
            grid[k[1], k[0]] = v
    return grid

def visualize_spec(spec, ax=None):
    """
    Visualizes a spec by plotting the grid of states with colors representing different cell types.
    
    Args:
        spec (dict): The spec to visualize.
        ax (matplotlib.axes.Axes, optional): The axes object where the grid will be plotted.
            If None, the current active axes will be used.
    
    Returns:
        matplotlib.image.AxesImage: The image object resulting from the imshow plotting.
    """
    grid = make_grid_from_spec(spec)
    if ax is None:
        ax = plt.gca()
        im = ax.imshow(grid, cmap='viridis')
    else: 
        im = ax.imshow(grid, cmap='viridis')
    ax.set_title(f'Spec: {spec["metadata"]["world_id"]}')

def get_vals(env):
    vals = np.zeros((env.world_generator.environment_spec['metadata']['width'], 
                     env.world_generator.environment_spec['metadata']['height']))
    for k, v in env.agent.planner.value_function.items():
        vals[k] = v
    return vals

def get_model(env):
    return copy.deepcopy(env.agent.world_model.environment_spec['states'])
    model = np.zeros((env.world_generator.environment_spec['metadata']['width'],
                      env.world_generator.environment_spec['metadata']['height']))
    for k,v in env.agent.world_model.environment_spec['states'].items():
        model[k] = v
    return model


### Visualization functions ###
def generate_color_matrix(matrix, mode='directional'):
    """
    Generate a color matrix for a given matrix. This function supports generating
    RGBA, HSVA, or directional RGB based on the mode specified.
    
    Args:
        matrix (np.array): The input matrix whose neighbor relationships are visualized.
        mode (str): The type of color matrix to generate. Options are 'rgba', 'hsva', 'directional'.
    
    Returns:
        np.array: A color matrix representing neighbor relationships.
    """
    n, m = matrix.shape
    color_matrix = np.zeros((n, m, 4 if mode != 'directional' else 3))  # Initialize appropriate shape

    for i in range(n):
        for j in range(m):
            neighbors = get_neighbors(matrix, i, j)
            softmax_values = softmax(neighbors)

            if mode == 'rgba':
                # RGBA color mapping using softmax directly
                color_matrix[i, j, :] = softmax_values  # Assign the softmax values as RGBA
            elif mode == 'hsva':
                # HSVA color mapping
                hue = softmax_values[0] * 360  # Hue from 0 to 360 degrees
                saturation = softmax_values[1]  # Saturation from 0 to 1
                value = softmax_values[2]  # Value from 0 to 1
                alpha = softmax_values[3]  # Alpha from 0 to 1
                rgb = mcolors.hsv_to_rgb([hue / 360.0, saturation, value])  # Convert to RGB
                rgba = np.concatenate((rgb, [alpha]))  # Create RGBA by appending alpha
                color_matrix[i, j, :] = rgba
            elif mode == 'directional':
                # Directional RGB using softmax to weight color directions
                color = np.zeros(3)  # RGB
                color += softmax_values[0] * np.array([0, 1, 0])  # Green for Up
                color += softmax_values[1] * np.array([1, 0, 0])  # Red for Down
                color += softmax_values[2] * np.array([1, 1, 0])  # Yellow for Left (R+G)
                color += softmax_values[3] * np.array([0, 0, 1])  # Blue for Right
                color_matrix[i, j] = color

    return color_matrix

def visualize_sandland(sandland, ax=None, color_map=None):
    """
    Visualizes the QuicksandLand grid with different colors representing different cell types
    such as walls, start, goal, and quicksand areas.

    Args:
        sandland (QuicksandLand): The QuicksandLand object containing the grid to visualize.
        ax (matplotlib.axes.Axes, optional): The axes object where the grid will be plotted.
            If None, the current active axes will be used.
        color_map (dict, optional): A dictionary mapping cell types to their respective colors.
            If None, a default set of colors is used.

    Returns:
        matplotlib.image.AxesImage: The image object resulting from the imshow plotting.
    """
    # Establish default color mapping if none is provided
    if color_map is None:
        color_map = {
            'normal_sand': '#ffe6cc',  # Light brown
            'quicksand': '#964b00',    # Dark brown
            'wall': 'grey',            # Grey
            'start': 'green',          # Green
            'goal': 'orange',          # Orange
        }
    
    # Generate a numerical grid for visualization based on cell types
    grid = np.zeros((sandland.height, sandland.width))
    for x, row in enumerate(sandland.grid):
        for y, cell in enumerate(row):
            if cell.is_wall:
                grid[x, y] = 2
            elif cell.is_start:
                grid[x, y] = 0
            elif cell.is_goal:
                grid[x, y] = 1
            elif cell.is_quicksand:
                grid[x, y] = 3
            else:
                grid[x, y] = 4

    # Set the color map based on the provided or default color mappings
    cmap = mcolors.ListedColormap([
        color_map['start'],
        color_map['goal'],
        color_map['wall'],
        color_map['quicksand'],
        color_map['normal_sand'],
    ])

    # Use the provided ax or the current active axes if ax is None
    if ax is None:
        ax = plt.gca()

    # Plot the grid
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
    ax.set_title('Quicksand Land Visualization')

    # Create a color bar with labels for the grid
    cbar = plt.colorbar(im, ax=ax, ticks=[0.4, 1.2, 2, 2.8, 3.6])
    cbar.ax.set_yticklabels(['Start', 'Goal', 'Wall', 'Quicksand', 'Sand'])

    return im

def relative_symlink(target: Union[Path, str], destination: Union[Path, str]):
    """Create a symlink pointing to ``target`` from ``destination``.
    
    Args:
        target: The target of the symlink (the file/directory that is pointed to).
        destination: The location of the symlink itself.
    """
    target = Path(target)
    destination = Path(destination)
    target_dir = destination.parent
    target_dir.mkdir(exist_ok=True, parents=True)
    relative_source = os.path.relpath(target, target_dir)

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    dir_fd = os.open(str(target_dir.absolute()), os.O_RDONLY)
    print(f"{relative_source} -> {destination.name} in {target_dir}")
    try:
        os.symlink(relative_source, destination.name, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)

def path_exists(start_pos, goal_pos, walls, grid_width, grid_height):
    def is_valid_move(x, y, grid_width, grid_height, walls, visited):
        return 0 <= x < grid_width and 0 <= y < grid_height and (x, y) not in walls and not visited[y][x]
    
    queue = deque([start_pos])
    visited = [[False for _ in range(grid_width)] for _ in range(grid_height)]
    visited[start_pos[1]][start_pos[0]] = True
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        x, y = queue.popleft()
        
        if (x, y) == goal_pos:
            return True
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if is_valid_move(nx, ny, grid_width, grid_height, walls, visited):
                visited[ny][nx] = True
                queue.append((nx, ny))
                
    return False

def dict_to_coords(coords):
    if type(coords) == dict:
        return (coords['y'], coords['x'])
    elif type(coords) == list and type(coords[0]) == dict:
        return [(coord['y'], coord['x']) for coord in coords]

def best_path(grid, start, goal, mode='min', walls=None):
    """Find the best path from start to goal in a grid using Dijkstra's algorithm.
    Walls are implemented by adding them to the visited set before starting the search.
    """
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    visited = np.full((rows, cols), False)
    if walls is not None:
        for wall in walls:
            visited[wall] = True

    if mode == 'min':
        initial_distance = np.inf
        comparison_operator = lambda a, b: a < b
        priority_queue = [(grid[start], start)]
    elif mode == 'max':
        initial_distance = -np.inf
        comparison_operator = lambda a, b: a > b
        priority_queue = [(-grid[start], start)]

    distance = np.full((rows, cols), initial_distance)
    distance[start] = grid[start]
    parent = np.full((rows, cols, 2), -1)  # To store the path

    while priority_queue:
        current_distance, current_pos = heapq.heappop(priority_queue)
        if mode == 'max':
            current_distance = -current_distance  # Convert back to positive for max mode

        if visited[current_pos]:
            continue

        visited[current_pos] = True

        if current_pos == goal:
            path = []
            while current_pos != start:
                path.append(current_pos)
                current_pos = tuple(parent[current_pos])
            path.append(start)
            path.reverse()
            return path

        random.shuffle(directions)  # Randomize the order of directions to break ties
        for direction in directions:
            neighbor = (current_pos[0] + direction[0], current_pos[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and not visited[neighbor]:
                new_distance = current_distance + grid[neighbor]
                if comparison_operator(new_distance, distance[neighbor]):
                    distance[neighbor] = new_distance
                    parent[neighbor] = current_pos
                    if mode == 'min':
                        heapq.heappush(priority_queue, (new_distance, neighbor))
                    elif mode == 'max':
                        heapq.heappush(priority_queue, (-new_distance, neighbor))  # Push negative to maintain max-heap

    return None  # Return None if no path is found

def get_value_of_path(path, grid):
    return sum(grid[tuple(pos)] for pos in path)

def get_path_quicksand_likelihood(path, grid):
    """Get the likelihood of encountering no quicksand along a path. Grid is a 2D array of quicksand probabilities."""
    return np.prod([1 - grid[tuple(pos)] for pos in path])

def parse_instance(instance):
    # instance = copy.deepcopy(instance)
    # instance = {ast.literal_eval(k): v for k,v in instance.items()}
    coords = list(instance.keys())
    grid = np.zeros((max([coord[1] for coord in coords]) + 1, max([coord[0] for coord in coords]) + 1))
    walls = []
    for coord, value in instance.items():
        # if value['tile_type'] == 'normal':
        grid[coord[1], coord[0]] = value['prob_quicksand']
        # elif
        if value['tile_type'] == 'wall':
            walls.append((coord[1], coord[0]))
    return grid, walls

def generate_walls(num_walls, width, height, start_loc, goal_loc):
    walls = []
    available_positions = [(x, y) for x in range(width) for y in range(height)
                           if (x, y) != (start_loc['x'], start_loc['y']) and (x, y) != (goal_loc['x'], goal_loc['y'])]
    random.shuffle(available_positions)
    while len(walls) < num_walls:
        if not available_positions:
            raise ValueError("Random wall generation failed: not enough available positions.")
        candidate_wall = available_positions.pop()
        candidate_walls = [(wall['x'], wall['y']) for wall in walls] + [candidate_wall]
        if path_exists((start_loc['x'], start_loc['y']), (goal_loc['x'], goal_loc['y']), candidate_walls, width, height):
            walls.append({'x' : candidate_wall[0], 'y': candidate_wall[1]})
    return walls

get_coords = lambda coords: (coords['y'], coords['x'])
grid_neg_log_likelihood = lambda grid: -np.log(1 - grid)
