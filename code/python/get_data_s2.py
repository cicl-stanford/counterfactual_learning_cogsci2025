
import os
import json
import uuid
import pandas as pd
from functools import reduce
from osf_data_handler import OSFDataHandler
criteria = {
    'project': 'counterfactual_learning',
    'experiment': 's2_quicksand',
    'iteration_name': 'fullstudy',
}
osf_handler = OSFDataHandler('uy7cw')
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
print(session_df.iloc[-1])
world_df = []
for world_id, group in df.groupby('worldID'):
    exam_trial = group[group.trial_type == 'quicksand-eval'].iloc[0]
    learning_duration = group.sequence_end.min() - group.sequence_start.min()
    world = {
        'world_id': world_id,
        'game_id': group.gameID.iloc[0],
        'condition': group.condition.iloc[0],
        'world_duration_ms': learning_duration + exam_trial.eval_time_ms,
        'learning_duration_ms': learning_duration,
    }
    spec = json.loads(group[group.trial_type == 'quicksand-setup'].iloc[0].spec)
    world.update({
        'world_spec_id': spec['metadata']['world_id'],
        'world_width': spec['metadata']['width'],
        'world_height': spec['metadata']['height'],
        'states': json.dumps(spec['states']),
        'num_observation_trials': len(spec['learning_phase']['planner_trials']),
        'num_simulation_trials': len(spec['learning_phase']['simulate_trials']),
        'has_evaluation_trial': spec['evaluation_phase']['evaluation_trial']
    })
    world.update({
        'exam_duration_ms': exam_trial.eval_time_ms,
        'exam_first_click': exam_trial.exam_first_click,        
        'exam_response': exam_trial.quicksand_info,
        'exam_click_events': exam_trial.click_events,        
    })
    world_df.append(world)
world_df = pd.DataFrame(world_df).reset_index(drop=True)
print(world_df.shape, '\n')
print(world_df.iloc[-1])
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
print(quicksand_df.shape, '\n')
print(quicksand_df.iloc[0])
(session_df.experiment_duration_ms / (1000*60)).mean()
for i, row in session_df.iterrows():
    print(row.prolific_id, row.condition)
    print(row.technical_difficulties)
    print(row.feedback, '\n')
session_df['total_bonus'] = session_df.game_id.map(quicksand_df.groupby('game_id').bonus.sum()).map('{:.2f}'.format)
print(session_df[['prolific_id', 'total_bonus']].to_csv(index=False))
session_df['total_bonus'].astype(float).mean()
project_dir = os.path.abspath('../../..')
save_dir = os.path.join(project_dir, 'data', criteria['experiment'], criteria['iteration_name'])
save_data = True
if save_data:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    session_df.to_csv(os.path.join(save_dir, 'session_data.csv'), index=False)
    world_df.to_csv(os.path.join(save_dir, 'world_data.csv'), index=False)
    quicksand_df.to_csv(os.path.join(save_dir, 'trial_data.csv'), index=False)
    print(f'saved data to {save_dir}...')
else:
    session_df = pd.read_csv(os.path.join(save_dir, 'session_data.csv'))
    world_df = pd.read_csv(os.path.join(save_dir, 'world_data.csv'))
    quicksand_df = pd.read_csv(os.path.join(save_dir, 'trial_data.csv'))