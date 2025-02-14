import os
import ast
import json
import uuid
import numpy as np
import pandas as pd
from functools import reduce
from osf_data_handler import OSFDataHandler

criteria = {
    'project': 'counterfactual_learning',
    'experiment': 's1_quicksand',
    'iteration_name': 'pilot5',
}
osf_handler = OSFDataHandler('32xkv')
df = osf_handler.load_filtered_csvs(criteria)

session_rename_map = {
    'gameID': 'game_id',
    'prolificID': 'prolific_id',
    'studyID': 'prolific_study_id',
    'sessionID': 'prolific_session_id',
    'condition': 'condition',
    'iteration': 'iteration',
    'dev_mode': 'dev_mode',
    'project': 'project',
    'experiment': 'experiment',
    'startPlanInstructionTS': 'start_plan_instruction_ts',
    'startPlanPracticeTS': 'start_plan_practice_ts',
    'startSimInstructionTS': 'start_sim_instruction_ts',
    'startSimPracticeTS': 'start_sim_practice_ts',
    'startExperimentTS': 'start_experiment_ts',
    'endExperimentTS': 'end_experiment_ts',
    'participantYears': 'age',
    'participantGender': 'gender',
    'participantRace': 'race',
    'participantEthnicity': 'ethnicity',
    'participantComments': 'feedback',
    'TechnicalDifficultiesFreeResp': 'technical_difficulties',
    'comprehensionAttempts': 'comprehension_attempts',
    'participantEffort': 'judged_effort',
    'judgedDifficulty': 'judged_difficulty',
    'inputDevice': 'input_device',
    'width': 'browser_width',
    'height': 'browser_height',
    'browser': 'browser',
    'mobile': 'is_mobile_device'
}
session_order = [
    "game_id", "prolific_id", "prolific_study_id", "prolific_session_id",
    "project", "experiment", "condition", "iteration", "dev_mode",
    "browser", "browser_width", "browser_height", "is_mobile_device",
    "start_plan_instruction_ts", "start_plan_practice_ts",
    "start_sim_instruction_ts", "start_sim_practice_ts",
    "start_experiment_ts", "end_experiment_ts", "experiment_duration_ms",
    "comprehension_attempts",
    "age", "gender", "race", "ethnicity",
    "judged_difficulty", "judged_effort", "input_device",
    "feedback", "technical_difficulties"
]
session_df = []
for game_id, group in df.groupby('gameID'):
    if not (group.trial_type == 'survey').any():
        continue
    S = group[group.trial_type == 'survey']
    session = S.iloc[1][~S.iloc[1].isna()]
    session_data = {session_rename_map[k]: v for k, v in session.items() if k in session_rename_map}
    session_data['experiment_duration_ms'] = session['time_elapsed']
    survey_data = json.loads(S[~S.response.isna()].response.values[0])
    survey_data = {session_rename_map[k]: v for k, v in survey_data.items() if k in session_rename_map}
    browser = group[group.trial_type == 'browser-check'].iloc[0][['width', 'height', 'browser', 'mobile']]
    browser = {session_rename_map[k]: v for k, v in browser.items() if k in session_rename_map}
    session_df.append({**session_data, **survey_data, **browser})
session_df = pd.DataFrame(session_df)[session_order].reset_index(drop=True)
print(session_df.condition.value_counts(), '\n')
print(session_df.iloc[0])
world_df = []
for world_id, group in df.groupby('worldID'):
    world = {
        'world_id': world_id,
        'game_id': group.gameID.iloc[0],
        'condition': group.condition.iloc[0],
        'world_duration_ms': group.sequence_end.max() - group.sequence_start.min(),
        'learning_duration_ms': group.sequence_end.min() - group.sequence_start.min(),
        'evaluation_duration_ms': group.sequence_end.max() - group.sequence_start.max()
    }
    spec = json.loads(group[group.trial_type == 'quicksand-setup'].iloc[0].spec)
    world.update({
        'world_spec_id': spec['metadata']['world_id'],
        'world_width': spec['metadata']['width'],
        'world_height': spec['metadata']['height'],
        'states': json.dumps(spec['states']),
        'num_observation_trials': len(spec['learning_phase']['planner_trials']),
        'num_simulation_trials': len(spec['learning_phase']['simulate_trials']),
        'num_evaluation_trials': len(spec['evaluation_phase']['navigation_trials'])
    })
    world_df.append(world)
world_df = pd.DataFrame(world_df).reset_index(drop=True)
print(world_df.shape, '\n')
print(world_df.iloc[0])
quicksand_rename_map = {
    'trial_id': 'trial_id',
    'gameID': 'game_id',
    'worldID': 'world_id',
    'instance_uuid': 'instance_id',
    'condition': 'condition',
    'trial_type': 'trial_type',
    'trial_phase': 'trial_phase',
    'trial_index': 'trial_index',
    'internal_node_id': 'internal_node_id',
    'start_position': 'start_position',
    'goal_position': 'goal_position',
    'wall_positions': 'wall_positions',
    'dist': 'distance_from_goal',
    'time_elapsed': 'trial_duration_ms',
    'planning_time_ms': 'planning_duration_ms',
    'observe_time_ms': 'observe_duration_ms',
    'path': 'path_to_goal',
    'click_events': 'click_events',
    'bonus': 'bonus',
    'quicksand_info': 'quicksand_instance_info',
    'environment_instance': 'environment_instance',
}
quicksand_order = [
    'trial_id', 'instance_id', 'world_id', 'game_id', 
    'condition', 'trial_type', 'trial_phase', 
    'trial_index', 'internal_node_id',
    'start_position', 'goal_position', 'wall_positions', 'distance_from_goal', 'bonus',
    'trial_duration_ms', 'planning_duration_ms', 'observe_duration_ms',
    'path_to_goal', 'click_events', 'quicksand_instance_info', 'environment_instance',
]
df['trial_id'] = [str(uuid.uuid4()) for i in range(len(df))]
quicksand_df = df.loc[
    df.trial_type.isin(['quicksand-planner', 'quicksand-simulate', 'quicksand-eval-navigation']) &\
    ~df.worldID.isna(),
    quicksand_rename_map.keys()
]
quicksand_df = quicksand_df.rename(columns=quicksand_rename_map)[quicksand_order]
def decode_json(json_str):
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return {ast.literal_eval(k) if isinstance(k, str) and k.startswith("(") else k: v \
                    for k, v in parsed.items()}
        elif isinstance(parsed, list):
            return [decode_json(json.dumps(item)) if isinstance(item, dict) else item for item in parsed]
        else:
            return parsed
    except (ValueError, json.JSONDecodeError):
        return json_str
def process_json_columns(df, json_columns):
    for col in json_columns:
        df[col] = df[col].apply(lambda x: decode_json(x) if isinstance(x, str) else x)
def compute_hazard_count(row):
    if row.trial_type == 'quicksand-planner':
        return sum(
            row.quicksand_instance_info[tuple(loc)]['prob_quicksand'] == 0.8
            for loc in row.path_to_goal
        )
    elif row.trial_type == 'quicksand-simulate':
        return sum(quicksand_df.loc[quicksand_df.trial_id == row.navigation_trial_id].iloc[0].quicksand_instance_info[
            tuple(loc)]['prob_quicksand'] == 0.8 for loc in row.path_to_goal)
    else: 
        return np.nan
world_df_json = ['states']
process_json_columns(world_df, world_df_json)
quicksand_df_json = ['start_position', 'goal_position', 'wall_positions', 
                     'path_to_goal', 'click_events', 'quicksand_instance_info', 'environment_instance']
process_json_columns(quicksand_df, quicksand_df_json)
quicksand_df['navigation_trial_id'] = quicksand_df.instance_id.map({
    value: key for key, value in \
    quicksand_df[quicksand_df.trial_type == 'quicksand-planner'].set_index('trial_id').instance_id.items()})
quicksand_df['hazard_count'] = quicksand_df.apply(compute_hazard_count, axis=1)
quicksand_df['day'] = (
    quicksand_df
    .groupby(['game_id', 'world_id', 'trial_type'])
    .trial_index
    .rank(method='first') - 1
)
print(quicksand_df.shape, '\n')
print(quicksand_df.iloc[0])
import sys
python_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(python_root)
sys.path.append(os.path.abspath('../quicksand'))
from environment import Environment
from agent import *
from planners import *
from quicksand_module import *
from utils import *
from quicksand.utils import *
from visualize_participant_trials import draw_grid, draw_ground_truth_grid
best_path
def compute_simulated_null_performance(row):
    if row.trial_type == 'quicksand-planner':
        return np.nan
    elif row.trial_type == 'quicksand-simulate':
        plan_row = quicksand_df.loc[quicksand_df.trial_id == row.navigation_trial_id].iloc[0]
        grid, walls = parse_instance(plan_row.quicksand_instance_info)
        noise_range = 0.01
        prior_grid = np.full(grid.shape, 0.5) + np.random.random(grid.shape) * noise_range * 2 - noise_range
        if row.condition == 'counterfactual':
            for loc in row.path_to_goal: 
                prior_grid[loc[1], loc[0]] = int(plan_row.quicksand_instance_info[tuple(loc)]['is_quicksand'])
        rational_path_to_goal = best_path(prior_grid, 
                                          dict_to_coords(row.start_position), 
                                          dict_to_coords(row.goal_position),
                                          walls=[dict_to_coords(coord) for coord in row.wall_positions])
        null_unsafe_tiles_conditioning = sum(
            plan_row.quicksand_instance_info[(loc[1], loc[0])]['prob_quicksand'] == 0.8 for loc in rational_path_to_goal)
        return null_unsafe_tiles_conditioning
    else: 
        return np.nan
quicksand_df['simulation_hazard_count_flat_priors'] = quicksand_df.apply(
    compute_simulated_null_performance, axis=1)
(session_df.experiment_duration_ms / (1000*60)).mean()
for i, row in session_df.iterrows():
    print(row.prolific_id, row.condition)
    print(row.technical_difficulties)
    print(row.feedback, '\n')
session_df['total_bonus'] = (
    session_df
    .game_id
    .map(
        quicksand_df
        .groupby('game_id')
        .bonus
        .sum()
    )
    .map('{:.2f}'.format))
print(session_df[['prolific_id', 'total_bonus']].to_csv(index=False))
session_df['total_bonus'].astype(float).mean()
project_dir = os.path.abspath('../../..')
save_dir = os.path.join(project_dir, 'data', criteria['experiment'], criteria['iteration_name'])
save_data = True
if save_data:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    session_df.to_csv(os.path.join(save_dir, 'session_data.csv'), index=True)
    world_df.to_csv(os.path.join(save_dir, 'world_data.csv'), index=True)
    quicksand_df.to_csv(os.path.join(save_dir, 'trial_data.csv'), index=True)
    print(f'saved data to {save_dir}...')
else:
    session_df = pd.read_csv(os.path.join(save_dir, 'session_data.csv'))
    world_df = pd.read_csv(os.path.join(save_dir, 'world_data.csv'))
    quicksand_df = pd.read_csv(os.path.join(save_dir, 'trial_data.csv'))
