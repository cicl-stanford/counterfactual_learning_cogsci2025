import uuid
import numpy as np

def generate_base_spec(width, height, world_id, update_per_timestep=False):
    """Generates a base specification dictionary for Quicksand environments."""
    return {
        "metadata": {
            "width": width,
            "height": height,
            "update_per_timestep": update_per_timestep,
            "geometric": False,
            "world_id": world_id
        },
        "states": {}
    }

def apply_random_walls(spec, grid, wall_probability=0.0, round_vals=True):
    """Apply random walls based on a given probability to the spec states."""
    for x in range(spec['metadata']['width']):
        for y in range(spec['metadata']['height']):
            spec['states'][(x, y)] = 'wall' if np.random.random() < wall_probability else round(grid[y, x], 2) if round_vals else grid[y, x]

def set_start_and_goal(spec, start_location, goal_location):
    """Set the start and goal locations in the spec."""
    spec['states'][start_location] = 'start'
    spec['states'][goal_location] = 'goal'
    
def spec_random(width=20, height=20, min_p=0, max_p=1, wall_probability=0.0):
    """Generate a random environment spec with a given width, height, min and max probabilities, and wall probability."""
    spec = generate_base_spec(width, height, str(uuid.uuid4()))
    grid = np.random.uniform(min_p, max_p, (height, width))
    apply_random_walls(spec, grid, wall_probability)
    return spec

def spec_beta(width=4, height=4, beta_val=0.5, wall_probability=0.0, round_vals=True):
    """Generate a random environment spec with a given width, height, beta value, and wall probability."""
    spec = generate_base_spec(width, height, str(uuid.uuid4()))
    grid = np.random.beta(beta_val, beta_val, (height, width))
    apply_random_walls(spec, grid, wall_probability, round_vals)
    return spec

def spec_bimodal_beta(width=4, height=4, alpha=2, beta=8, wall_probability=0.0):
    """Generate a random environment spec with a given width, height, alpha and beta values, and wall probability."""
    spec = generate_base_spec(width, height, str(uuid.uuid4()))
    bimodal_beta = lambda a, b: np.random.beta(a, b) if np.random.random() < 0.5 else np.random.beta(b, a)
    grid = np.array([[bimodal_beta(alpha, beta) for _ in range(width)] for _ in range(height)])
    apply_random_walls(spec, grid, wall_probability)
    return spec

def spec_bernoulli_modes(width=4, height=4, modes=[0.2, 0.8], wall_probability=0.0):
    """Generate a random environment spec with a given width, height, modes, and wall probability."""
    spec = generate_base_spec(width, height, str(uuid.uuid4()))
    grid = np.random.choice(modes, (height, width))
    apply_random_walls(spec, grid, wall_probability)
    return spec

def spec_0(min_p=0, max_p=1):
    """Generate a basic environment spec with customizable min and max probabilities."""
    width, height = 10, 10
    spec = generate_base_spec(width, height, 'spec_0')
    grid = np.full((height, width), min_p)
    grid[1:9, 1:9] = np.full((8, 8), max_p - (1 - max_p))
    blocked_areas = [(1, 4), (2, 4), (3, 4), (3, 5), (3, 6), (4, 6), (4, 7), (4, 8), (5, 8), (6, 8), (7, 8)]
    for i in blocked_areas:
        grid[i] = min_p
    apply_random_walls(spec, grid)
    return spec

def spec_1(width=20, height=20, min_p=0):
    """Generate an environment spec with even sinusoidal patterns."""
    spec = generate_base_spec(width, height, 'spec_1')
    c = np.pi / 4
    grid = np.array([[np.sin(c * x - 2) + np.sin(c * y - 2) for x in range(width)] for y in range(height)])
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    grid = grid * (1 - min_p) + min_p
    apply_random_walls(spec, grid)
    return spec

def spec_2(width=20, height=20, min_p=0):
    """Generate an environment spec with even sinusoidal patterns, setting a part of the environment's quicksand at max."""
    spec = generate_base_spec(width, height, 'spec_2')
    c = np.pi / 4
    grid = np.array([[max(np.sin(c * x - 2) + np.sin(c * y - 2), 2 * (((width - x) + y) > width)) for x in range(width)] for y in range(height)])
    grid = np.power(grid, 0.3)
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    grid = grid * (1 - min_p) + min_p
    apply_random_walls(spec, grid)
    return spec

def spec_3(width=20, height=20, min_p=0, max_p=1, wavyness=6):
    """Generate an environment spec with a cosine wave pattern that is larger in the center of the gridworld."""
    spec = generate_base_spec(width, height, 'spec_3')
    grid = np.zeros((height, width))
    for x, v in enumerate(np.linspace(0, wavyness, width)):
        for y, w in enumerate(np.linspace(0, wavyness, height)):
            grid[y, x] = np.cos(np.abs(v - wavyness / 2)**1.5) + np.cos(np.abs(w - wavyness / 2)**1.5)
    grid = (grid - grid.min()) / (grid.max() - grid.min())
    grid = grid * (max_p - min_p) + min_p
    apply_random_walls(spec, grid)
    return spec