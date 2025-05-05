# This notebook computes the preregistered variables from https://osf.io/tzha7.
import os
import ast
import json
import heapq
import random
import numpy as np
import pandas as pd

# ---------- Utility Functions ----------

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
    parent = np.full((rows, cols, 2), -1) # To store the path

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

def decode_json(json_str):
    """Safely decode JSON strings in a dataframe column."""
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return {
                ast.literal_eval(k) if isinstance(k, str) and k.startswith("(") else k: v
                for k, v in parsed.items()
            }
        elif isinstance(parsed, list):
            return [decode_json(json.dumps(item)) if isinstance(item, dict) else item for item in parsed]
        else:
            return parsed
    except (ValueError, json.JSONDecodeError):
        try:
            return ast.literal_eval(json_str)
        except:
            return json_str

def process_json_columns(df, json_columns):
    """Decodes JSON columns in the given DataFrame."""
    for col in json_columns:
        df[col] = df[col].apply(lambda x: decode_json(x) if isinstance(x, str) else x)


# ---------- Derived Variables ----------

def compute_day(quicksand_df):
    """Computes the 'day' variable based on trial index."""
    return (
        quicksand_df
        .groupby(['game_id', 'world_id', 'trial_type'])
        .trial_index
        .rank(method='first') - 1
    )

def compute_hazard_count(row, quicksand_df):
    """Computes hazard counts for trials."""
    
    if row.trial_type == 'quicksand-planner':
        return sum(
            row.quicksand_instance_info[tuple(loc)]['prob_quicksand'] == 0.8
            for loc in row.path_to_goal
        )
    elif row.trial_type == 'quicksand-simulate':
        linked_trial = quicksand_df.loc[quicksand_df.trial_id == row.navigation_trial_id].iloc[0]
        return sum(
            linked_trial.quicksand_instance_info[tuple(loc)]['prob_quicksand'] == 0.8
            for loc in row.path_to_goal
        )
    else:
        return np.nan

def compute_exam_trial_correct_tiles(world_df):
    """Computes the number of correct tiles selected in the exam trial."""
    prob_safe_map = {'safe': 0.0, 'unsafe': 0.8}
    return world_df.exam_response.apply(
        lambda response: sum(
            tile['prob_quicksand'] == prob_safe_map[tile['rating']]
            for tile in response.values()
        )
    )

def compute_exam_trial_hazard_count(world_df, quicksand_df):
    """Simulates a rational agent using participants' exam responses."""
    prob_safe_map = {'safe': 0.0, 'unsafe': 0.8}

    def parse_exam_response(response):
        coords = list(response.keys())
        grid = np.zeros((
            max([coord[1] for coord in coords]) + 1, 
            max([coord[0] for coord in coords]) + 1))
        for coord, value in response.items():
            grid[coord[1], coord[0]] = prob_safe_map[value['rating']]
        return grid

    hazard_counts = []
    for _, row in world_df.iterrows():
        world_id = row.world_id
        exam_grid = parse_exam_response(row.exam_response)
        navigation_trials = quicksand_df.loc[
            (quicksand_df.world_id == world_id) &
            (quicksand_df.trial_type == 'quicksand-planner')
        ]

        simulated_num_hazards = []
        for _, nav_trial in navigation_trials.iterrows():
            exam_best_path = best_path(
                exam_grid,
                dict_to_coords(nav_trial.start_position),
                dict_to_coords(nav_trial.goal_position),
                walls=dict_to_coords(nav_trial.wall_positions)
            )
            simulated_num_hazards.append(
                sum(row.states[(loc[1], loc[0])] == 0.8 for loc in exam_best_path[1:-1])
            )

        hazard_counts.append(np.mean(simulated_num_hazards))

    return hazard_counts

# ---------- Main Processing ----------

if __name__ == "__main__":
    print("Computing derived variables...")
    python_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(python_dir, '..', '..'))
    s1_dir = os.path.join(project_dir, 'data', 's1_quicksand')
    s2_dir = os.path.join(project_dir, 'data', 's2_quicksand')

    # s1 variables
    s1_quicksand_df = pd.read_csv(os.path.join(s1_dir, 'trial_data.csv'))
    process_json_columns(s1_quicksand_df, ['path_to_goal', 'quicksand_instance_info'])
    
    s1_quicksand_df['navigation_trial_id'] = s1_quicksand_df.instance_id.map({
        value: key for key, value in s1_quicksand_df[s1_quicksand_df.trial_type == 'quicksand-planner']
        .set_index('trial_id').instance_id.items()
    })

    s1_quicksand_df['hazard_count'] = s1_quicksand_df.apply(
        lambda row: compute_hazard_count(row, s1_quicksand_df), axis=1
    )
    s1_quicksand_df['day'] = compute_day(s1_quicksand_df)
    s1_quicksand_df.drop(columns=['navigation_trial_id'], inplace=True)

    s1_quicksand_df.to_csv(os.path.join(s1_dir, 'trial_data.csv'), index=False)

    # s2 variables
    s2_quicksand_df = pd.read_csv(os.path.join(s2_dir, 'trial_data.csv'))
    s2_world_df = pd.read_csv(os.path.join(s2_dir, 'world_data.csv'))
    process_json_columns(s2_quicksand_df, ['start_position', 'goal_position', 'wall_positions', 'path_to_goal', 'quicksand_instance_info'])
    process_json_columns(s2_world_df, ['exam_response', 'states'])

    s2_quicksand_df['navigation_trial_id'] = s2_quicksand_df.instance_id.map({
        value: key for key, value in s2_quicksand_df[s2_quicksand_df.trial_type == 'quicksand-planner']
        .set_index('trial_id').instance_id.items()
    })
    s2_quicksand_df['hazard_count'] = s2_quicksand_df.apply(
        lambda row: compute_hazard_count(row, s2_quicksand_df), axis=1
    )
    s2_quicksand_df['day'] = compute_day(s2_quicksand_df)
    s2_world_df['exam_trial_correct_tiles'] = compute_exam_trial_correct_tiles(s2_world_df)
    s2_world_df['exam_trial_hazard_count'] = compute_exam_trial_hazard_count(s2_world_df, s2_quicksand_df)
    s2_quicksand_df.drop(columns=['navigation_trial_id'], inplace=True)

    s2_quicksand_df.to_csv(os.path.join(s2_dir, 'trial_data.csv'), index=False)
    s2_world_df.to_csv(os.path.join(s2_dir, 'world_data.csv'), index=False)

    print("Finished computing derived variables.")