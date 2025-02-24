experiments = {
    "s1": {
        "osf_id": "32xkv",
        "criteria": {
            "project": "counterfactual_learning",
            "experiment": "s1_quicksand",
            "iteration_name": "pilot5"
        },
        "exam_trial": False
    },
    "s2": {
        "osf_id": "uy7cw",
        "criteria": {
            "project": "counterfactual_learning",
            "experiment": "s2_quicksand",
            "iteration_name": "fullstudy"
        },
        "exam_trial": True
    }
}

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
    'time_elapsed': 'experiment_duration_ms',
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
    "start_plan_instruction_ts", "start_plan_practice_ts", "start_sim_instruction_ts","start_sim_practice_ts", "start_experiment_ts", "end_experiment_ts", "experiment_duration_ms",
    "comprehension_attempts",
    "age", "gender", "race", "ethnicity", 
    "judged_difficulty", "judged_effort", "input_device",
    "feedback", "technical_difficulties"
]

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
    'environment_instance': 'environment_instance'
}

quicksand_order = [
    'trial_id', 'instance_id', 'world_id', 'game_id',
    'condition', 'trial_type', 'trial_phase', 'trial_index', 'internal_node_id',
    'start_position', 'goal_position', 'wall_positions', 'distance_from_goal',
    'bonus',
    'trial_duration_ms', 'planning_duration_ms', 'observe_duration_ms',
    'path_to_goal', 'click_events', 'quicksand_instance_info', 'environment_instance'
]

world_df_json = ['states']
quicksand_df_json = [
    'start_position',
    'goal_position',
    'wall_positions',
    'path_to_goal',
    'click_events',
    'quicksand_instance_info',
    'environment_instance'
]
