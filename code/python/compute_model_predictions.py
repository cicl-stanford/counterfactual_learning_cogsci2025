import os
import sys
import pandas as pd
import numpy as np
from compute_derived_variables import process_json_columns
from joblib import Parallel, delayed

# Import quicksand modules
python_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(python_dir, 'quicksand'))

from quicksand.environment import Environment
from quicksand.agent import *
from quicksand.planners import *
from quicksand.quicksand_module import *
from quicksand.utils import *

def setup_environment(world_spec, prob_simulation_noise=0.6, prob_observation_noise=0.75, mixture_weight=0.9, softmax_temp=0.01, decay_rate=0.95):
    quicksand_generator = QuicksandGenerator(environment_spec=world_spec)
    env = Environment(quicksand_generator, BayesianModelLearningAgent)
    env.initialize_agent(
        NegLogLikelihoodSoftmaxPlanner, 
        agent_params={
            'prior': 0.5,
            'prob_simulation_noise': prob_simulation_noise,
            'prob_observation_noise': prob_observation_noise,
            'mixture_weight': mixture_weight,
            'decay_rate': decay_rate
        }, 
        planner_params={
            'softmax_temp': softmax_temp
        }
    )
    return env

def safe_best_path(grid, start, goal, walls=None):
    path = best_path(grid, start, goal, walls=walls)
    if path is None:
        return None
    return [(int(x), int(y)) for x, y in path]


def compute_model_predictions(quicksand_df, world_df, softmax_temp=0.01, n_jobs=-1):
    planner_df = quicksand_df[quicksand_df.trial_type == 'quicksand-planner']
    planner_df = planner_df.sort_values(by=['world_id', 'day']).reset_index(drop=True)

    hazard_count = lambda row, path: sum(
        row.quicksand_instance_info[(loc[1], loc[0])]['prob_quicksand'] == 0.8
        for loc in path
    )
    
    def update_env_states_from_grid(envr, grid, to_safe=True):
        height, width = grid.shape

        def transform_value(val):
            return 1.0 if to_safe and val == 0.8 else float(val)

        new_states = {(x, y): transform_value(grid[y, x]) for y in range(height) for x in range(width)}
        envr.agent.planner.world_model.environment_spec['states'].update(new_states)

    def get_env_path(envr, row, walls, update_model=True):
        start = (row.start_position['x'], row.start_position['y'])
        goal = (row.goal_position['x'], row.goal_position['y'])
        path = envr.run_episode(start, goal, wall_locations=walls, greedy=False, update_model=update_model)
        return [(tile[1][1], tile[1][0]) for tile in path] + [(goal[1], goal[0])]

    def process_world(world_id, block):
        world_spec = world_df[world_df.world_id == world_id].iloc[0].to_dict()
        world_spec['metadata'] = {
            'width': world_spec['world_width'],
            'height': world_spec['world_height'],
            'update_per_timestep': False,
            'geometric': False,
            'world_id': world_spec['world_id']
        }

        env = setup_environment(world_spec, prob_simulation_noise=0.5, prob_observation_noise=1, mixture_weight=1, softmax_temp=softmax_temp, decay_rate=1)
        env_oracle = setup_environment(world_spec, prob_simulation_noise=0.5, prob_observation_noise=1, mixture_weight=1, softmax_temp=softmax_temp, decay_rate=1)
        env_rand = setup_environment(world_spec, prob_simulation_noise=0.5, prob_observation_noise=1, mixture_weight=1, softmax_temp=softmax_temp, decay_rate=1)

        records = []
        for _, row in block.iterrows():
            grid, walls = parse_instance(row.quicksand_instance_info)
            walls = dict_to_coords(row.wall_positions)

            noise_range = 0.001
            prior_grid = np.full(grid.shape, 0.5) + np.random.random(grid.shape) * noise_range * 2 - noise_range

            update_env_states_from_grid(env_oracle, grid, to_safe=True)
            update_env_states_from_grid(env_rand, prior_grid, to_safe=False)

            ideal_path_to_goal = get_env_path(env, row, walls)
            best_path_to_goal = get_env_path(env_oracle, row, walls, update_model=False)
            rand_path_to_goal = get_env_path(env_rand, row, walls, update_model=False)

            ideal_path_set = set(ideal_path_to_goal) if ideal_path_to_goal else None
            best_path_set = set(best_path_to_goal) if best_path_to_goal else None
            rand_path_set = set(rand_path_to_goal) if rand_path_to_goal else None

            ideal_hazard_count = hazard_count(row, ideal_path_set) if ideal_path_set else None
            best_hazard_count = hazard_count(row, best_path_set) if best_path_set else None
            rand_hazard_count = hazard_count(row, rand_path_set) if rand_path_set else None

            records.extend([
                {'condition': 'optimal learner', 'trial_id': row.trial_id, 'path_to_goal': ideal_path_to_goal, 'hazard_count': ideal_hazard_count},
                {'condition': 'oracle', 'trial_id': row.trial_id, 'path_to_goal': best_path_to_goal, 'hazard_count': best_hazard_count},
                {'condition': 'random prior', 'trial_id': row.trial_id, 'path_to_goal': rand_path_to_goal, 'hazard_count': rand_hazard_count},
            ])
        return records

    # Parallel processing per world_id
    grouped = list(planner_df.groupby('world_id'))
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_world)(world_id, block) for world_id, block in grouped
    )

    # Flatten list of lists
    model_records = [record for sublist in results for record in sublist]

    return pd.DataFrame(model_records)

if __name__ == "__main__":
    print("Computing model predictions...")

    # Set up directory paths
    python_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(python_dir, '..', '..'))
    s1_dir = os.path.join(project_dir, 'data', 's1_quicksand')
    s2_dir = os.path.join(project_dir, 'data', 's2_quicksand')

    # Process s1
    s1_quicksand_df = pd.read_csv(os.path.join(s1_dir, 'trial_data.csv'))
    process_json_columns(s1_quicksand_df, ['start_position', 'goal_position', 'wall_positions', 'path_to_goal', 'quicksand_instance_info'])
    s1_world_df = pd.read_csv(os.path.join(s1_dir, 'world_data.csv'))
    process_json_columns(s1_world_df, ['states'])
    s1_model_df = compute_model_predictions(s1_quicksand_df, s1_world_df)
    s1_model_df.to_csv(os.path.join(s1_dir, 'model_predictions.csv'), index=False)
    print(f"s1 model predictions saved to {os.path.join(s1_dir, 'model_predictions.csv')}")

    # Process s2
    s2_quicksand_df = pd.read_csv(os.path.join(s2_dir, 'trial_data.csv'))
    process_json_columns(s2_quicksand_df, ['start_position', 'goal_position', 'wall_positions', 'path_to_goal', 'quicksand_instance_info'])
    s2_world_df = pd.read_csv(os.path.join(s2_dir, 'world_data.csv'))
    process_json_columns(s2_world_df, ['states'])
    s2_model_df = compute_model_predictions(s2_quicksand_df, s2_world_df)
    s2_model_df.to_csv(os.path.join(s2_dir, 'model_predictions.csv'), index=False)
    print(f"s2 model predictions saved to {os.path.join(s2_dir, 'model_predictions.csv')}")

    print("Finished computing model predictions.")