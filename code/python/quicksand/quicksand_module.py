import numpy as np
import random
from utils import *

class QuicksandActions:
    # note: (x, y) = (0, 0) is the top-left corner
    UP = (0, -1)  # No change in x, decrease in y
    DOWN = (0, 1)  # No change in x, increase in y
    LEFT = (-1, 0)  # Decrease in x, no change in y
    RIGHT = (1, 0)  # Increase in x, no change in y
    action_space = [UP, DOWN, LEFT, RIGHT]
    
    @staticmethod
    def describe_action(action):
        if action == QuicksandActions.UP:
            return "Move up (dx=0, dy=-1)"
        elif action == QuicksandActions.DOWN:
            return "Move down (dx=0, dy=1)"
        elif action == QuicksandActions.LEFT:
            return "Move left (dx=-1, dy=0)"
        elif action == QuicksandActions.RIGHT:
            return "Move right (dx=1, dy=0)"
        else:
            return "Unknown action"

class QuicksandElement:
    def __init__(self, location, prob_quicksand, 
                is_quicksand=False, is_wall=False, is_start=False, is_goal=False):
        self.location = location
        self.prob_quicksand = prob_quicksand
        self.is_quicksand = is_quicksand
        self.is_wall = is_wall
        self.is_start = is_start
        self.is_goal = is_goal
        
    def __str__(self):
        if self.is_wall:
            return f'Wall at {self.location}'
        elif self.is_start:
            return f'Start at {self.location}'
        elif self.is_goal:
            return f'Goal at {self.location}'
        else:
            return f'Quicksand={str(self.is_quicksand)} at {self.location} with prob {self.prob_quicksand}'

class QuicksandLand:
    def __init__(self, grid, width, height, update_per_timestep, start_location, goal_location):
        self.grid = grid
        self.width = width
        self.height = height
        self.update_per_timestep = update_per_timestep
        self.start_location = start_location
        self.goal_location = goal_location
        self.timestep = 0
        self.in_stall = False

    def get_grid_element(self, state):
        x, y = state
        return self.grid[y][x]  # Accessing by row-major [y][x] after conversion from (x, y)

    def inbounds(self, state):
        x, y = state
        return 0 <= x < self.width and 0 <= y < self.height
    
    def get_new_state(self, state, action):
        x, y = state
        dx, dy = action
        new_x, new_y = x + dx, y + dy
        return new_x, new_y
    
    def transition(self, current_state, action):
        info = {'stalled': False}
        reward = -1
        task_finished = False

        next_state = self.get_new_state(current_state, action)
        if not self.inbounds(next_state):
            next_state = current_state
            reward = -4
        next_cell = self.get_grid_element(next_state)

        if next_cell.is_wall:
            reward = -4
            next_state = current_state
        elif self.get_grid_element(current_state).is_quicksand:
            info['stalled'] = True
            if self.update_per_timestep:
                next_state = current_state
            else:
                self.in_stall = True
                reward = -2

        self.progress_world()
        if next_cell.is_goal:
            reward = 0
            task_finished = True

        return next_state, reward, self.timestep, task_finished, info
    
    def progress_world(self):
        if self.update_per_timestep:
            for row in self.grid:
                for cell in row:
                    if not cell.is_wall:
                        cell.is_quicksand = np.random.random() < cell.prob_quicksand
            self.timestep += 1
        else:
            if self.in_stall:
                self.timestep += 2
                self.in_stall = False
            else:
                self.timestep += 1

@forkable
class QuicksandGenerator:
    def __init__(self, environment_spec=None):
        if environment_spec is None:
            self.environment_spec = self.init_default_spec()
        else:
            self.environment_spec = environment_spec
        self.action_space = QuicksandActions.action_space
        self.state_space = list(self.environment_spec['states'].keys())

    def init_default_spec(self):
        spec = {
            "metadata": {"width": 15, "height": 15, "update_per_timestep": False},
            "states": {}
        }        
        for x in range(spec['metadata']['width']):
            for y in range(spec['metadata']['height']):
                # by default, no walls
                spec['states'][(x, y)] = 'wall' if bernoulli(0.0) else beta(4, 1)
    
        return spec
    
    def get_base_spec(self):
        base_spec = {
            "metadata": self.environment_spec['metadata'],
            "states": {}
        }
        for state, _ in self.environment_spec['states'].items():
            base_spec['states'][state] = None
        return base_spec

    def create_grid(self, states, width, height, start_location, goal_location, wall_locations=None, confusion_rate=0.0):
        if wall_locations is None:
            wall_locations = []

        grid = [[None] * width for _ in range(height)]
        for (x, y), cell_type in states.items():
            if (x, y) == start_location:
                grid[y][x] = QuicksandElement((x, y), None, False, False, True, False)
            elif (x, y) == goal_location:
                grid[y][x] = QuicksandElement((x, y), None, False, False, False, True)
            elif (x, y) in wall_locations:
                grid[y][x] = QuicksandElement((x, y), None, False, True, False, False)
            elif isinstance(cell_type, str) and cell_type == 'wall':
                grid[y][x] = QuicksandElement((x, y), None, False, True, False, False)
            else:
                # a 2-length tuple with scalar elements
                if isinstance(cell_type, tuple) and len(cell_type) == 2 and all(map(np.isscalar, cell_type)):
                    alpha, beta = cell_type
                    p = np.random.beta(alpha, beta)
                elif np.isscalar(cell_type):
                    p = cell_type
                else:
                    raise ValueError(f"Invalid cell type at {(x, y)}: {cell_type}")

                is_quicksand = np.random.rand() < p
                is_quicksand = not is_quicksand if np.random.rand() < confusion_rate else is_quicksand
                prob_quicksand = 0 if self.environment_spec['metadata']['update_per_timestep'] else p
                grid[y][x] = QuicksandElement((x, y), prob_quicksand, is_quicksand, False, False, False)

        return grid
    
    def generate_world(self, start_location, goal_location, wall_locations=None, confusion_rate=0.0):
        width = self.environment_spec['metadata']['width']
        height = self.environment_spec['metadata']['height']
        update_per_timestep = self.environment_spec['metadata']['update_per_timestep']
        grid = self.create_grid(self.environment_spec['states'], width, height, start_location, goal_location, wall_locations, confusion_rate)
        return QuicksandLand(grid, width, height, update_per_timestep, start_location, goal_location)