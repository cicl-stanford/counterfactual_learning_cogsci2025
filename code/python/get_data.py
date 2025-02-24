import os
import json
import uuid
import pandas as pd

# custom modules
import config
from osf_data_handler import OSFDataHandler

def create_quicksand_dataframe(dataframe):
    """Processes quicksand trial-level data from raw OSF dataframe."""
    # Generate unique trial IDs
    dataframe['trial_id'] = [str(uuid.uuid4()) for _ in range(len(dataframe))]

    # Filter quicksand trials
    filtered_trials = dataframe.loc[
        dataframe.trial_type.isin([
            'quicksand-planner', 
            'quicksand-simulate', 
        ]) & ~dataframe.worldID.isna(),
        config.quicksand_rename_map.keys()
    ]

    quicksand_df = (
        filtered_trials
        .rename(columns=config.quicksand_rename_map)[config.quicksand_order]
        .reset_index(drop=True)
    )

    return quicksand_df

def create_block_dataframe(dataframe, experiment_config):
    """Processes block-level data from raw OSF dataframe."""
    block_records = []
    include_exam_trial = experiment_config.get("exam_trial", False)

    for block_id, group in dataframe.groupby("worldID"):
        learning_duration = group.sequence_end.min() - group.sequence_start.min()
        exam_duration = 0

        if include_exam_trial:
            exam_trial = group[group.trial_type == "quicksand-eval"].iloc[0]
            exam_duration = exam_trial.eval_time_ms

        block_data = {
            "world_id": block_id,
            "game_id": group.gameID.iloc[0],
            "condition": group.condition.iloc[0],
            "world_duration_ms": learning_duration + exam_duration,
            "learning_duration_ms": learning_duration
        }


        spec = json.loads(group[group.trial_type == "quicksand-setup"].iloc[0].spec)
        block_data.update({
            "world_spec_id": spec["metadata"]["world_id"],
            "world_width": spec["metadata"]["width"],
            "world_height": spec["metadata"]["height"],
            "states": json.dumps(spec["states"]),
            "num_observation_trials": len(spec["learning_phase"]["planner_trials"]),
            "num_simulation_trials": len(spec["learning_phase"]["simulate_trials"]),
            "has_evaluation_trial": spec["evaluation_phase"].get("evaluation_trial", False)
        })

        if include_exam_trial:
            block_data.update({
                "exam_duration_ms": exam_trial.eval_time_ms,
                "exam_first_click": exam_trial.exam_first_click,
                "exam_response": exam_trial.quicksand_info,
                "exam_click_events": exam_trial.click_events
            })

        block_records.append(block_data)

    block_df = pd.DataFrame(block_records).reset_index(drop=True)
    return block_df

def create_session_dataframe(dataframe, quicksand_df):
    """Processes session-level data from raw OSF dataframe."""
    session_records = []

    for _, participant in dataframe.groupby('gameID'):
        # Check if survey data exists
        if not (participant.trial_type == 'survey').any():
            print(f"Participant {participant.gameID.iloc[0]} has no survey data!")
            continue

        survey_trials = participant[participant.trial_type == 'survey']
        session_info = survey_trials.iloc[1].dropna()
        survey_info = json.loads(
            survey_trials[~survey_trials.response.isna()].response.values[0]
        )
        browser_info = participant[participant.trial_type == 'browser-check'].iloc[0][
            ['width', 'height', 'browser', 'mobile']
        ]

        combined_session_data = {
            **session_info,
            **survey_info,
            **browser_info
        }

        remapped_session_data = {
            config.session_rename_map.get(key, key): value
            for key, value in combined_session_data.items()
            if key in config.session_rename_map
        }

        session_records.append(remapped_session_data)

    session_df = pd.DataFrame(session_records)[config.session_order].reset_index(drop=True)
    session_df['total_bonus'] = session_df.game_id.map(
        quicksand_df.groupby('game_id').bonus.sum()
        ).map('{:.2f}'.format)

    return session_df

def save_processed_data(session_df, block_df, quicksand_df, experiment_config):
    project_dir = os.path.abspath('../../')
    save_dir = os.path.join(
        project_dir, 'data',
        experiment_config['criteria']['experiment'],
        experiment_config['criteria']['iteration_name']
    )

    os.makedirs(save_dir, exist_ok=True)

    session_df.to_csv(os.path.join(save_dir, 'session_data.csv'), index=False)
    block_df.to_csv(os.path.join(save_dir, 'world_data.csv'), index=False)
    quicksand_df.to_csv(os.path.join(save_dir, 'trial_data.csv'), index=False)

    print(f"Data saved to {save_dir}")

def process_experiment_data(study_key):
    print(f"Processing data for {study_key}...")

    settings = config.experiments[study_key]
    osf_handler = OSFDataHandler(settings['osf_id'])
    osf_data = osf_handler.load_filtered_csvs(settings['criteria'])

    quicksand_df = create_quicksand_dataframe(osf_data)
    block_df = create_block_dataframe(osf_data, settings)
    session_df = create_session_dataframe(osf_data, quicksand_df)

    save_processed_data(session_df, block_df, quicksand_df, settings)

if __name__ == "__main__":
    for study_key in config.experiments.keys():
        process_experiment_data(study_key)
