
import os
import re
import pandas as pd
import config

def load_dataframes(experiment_config):
    """Load pre-processed session, block, and quicksand data for a given experiment."""
    project_dir = os.path.abspath('../..')
    data_dir = os.path.join(
        project_dir, 'data',
        experiment_config['criteria']['experiment'],
        experiment_config['criteria']['iteration_name']
    )

    session_df = pd.read_csv(os.path.join(data_dir, 'session_data.csv'))
    world_df = pd.read_csv(os.path.join(data_dir, 'world_data.csv'))
    quicksand_df = pd.read_csv(os.path.join(data_dir, 'trial_data.csv'))

    # Add experiment duration in minutes
    session_df['experiment_duration_min'] = session_df.experiment_duration_ms / (1000 * 60)

    return session_df, world_df, quicksand_df

def demographics_categorical(df, feature):
    """Summarize categorical demographic information."""
    return {
        category.lower(): sum(df[feature] == category)
        for category in df[feature].dropna().unique()
    }

def demographics_ordinal(df, feature):
    """Summarize ordinal demographic information."""
    return {
        'mean': round(df[feature].mean(), 1),
        'median': int(df[feature].median()),
        'min': int(df[feature].min()),
        'max': int(df[feature].max()),
    }

def demographics_ratio(df, feature):
    """Summarize ratio-scale demographic information."""
    return {
        'mean': round(df[feature].mean(), 1),
        'median': round(df[feature].median(), 1),
        'min': round(df[feature].min(), 1),
        'max': round(df[feature].max(), 1),
    }

def summarize_methods_info(session_df, experiment_config):
    """Generate methods information for the LaTeX table."""
    return {
        'basePay': experiment_config.get('base_pay', '3'),
        'nParticipants': session_df.game_id.nunique(),
        'participantSex': demographics_categorical(session_df, 'gender'),
        'participantAge': demographics_ordinal(session_df, 'age'),
        'participantRace': demographics_categorical(session_df, 'race'),
        'participantEthnicity': demographics_categorical(session_df, 'ethnicity'),
        'participantDifficulty': demographics_ordinal(session_df, 'judged_difficulty'),
        'participantEffort': demographics_ordinal(session_df, 'judged_effort'),
        'participantDevice': demographics_categorical(session_df, 'input_device'),
        'participantBonus': demographics_ratio(session_df, 'total_bonus'),
        'participantCompletionTime': demographics_ratio(session_df, 'experiment_duration_min'),
        'participantComprehensionAttempts': demographics_ordinal(session_df, 'comprehension_attempts')
    }

def write_latex_commands(data, prefix='', file_handle=None):
    """Recursively write LaTeX commands for all metrics."""
    for key, value in data.items():
        # Convert snake_case to CamelCase
        key = re.sub(r'[-/\s]+(.)', lambda x: x.group(1).upper(), key)
        if isinstance(value, dict):
            write_latex_commands(value, prefix + key, file_handle)
        else:
            file_handle.write(f"\\newcommand{{\\{prefix}{key}}}{{{value}}}\n")

def save_methods_info_to_latex(methods_info, save_path):
    """Save methods information as LaTeX commands in a .tex file."""
    with open(save_path, 'w') as f:
        write_latex_commands(methods_info, f=f)
    print(f"Methods info saved to {save_path}")

def process_experiment(study_key):
    """Main processing loop for a single experiment."""
    print(f"Processing methods info for {study_key}...")

    settings = config.experiments[study_key]
    session_df, world_df, quicksand_df = load_dataframes(settings)

    methods_info = summarize_methods_info(session_df, settings)

    # Save the LaTeX-formatted methods info
    project_dir = os.path.abspath('../..')
    data_dir = os.path.join(
        project_dir, 'data',
        settings['criteria']['experiment'],
        settings['criteria']['iteration_name']
    )
    save_path = os.path.join(data_dir, 'methods_vars.tex')
    save_methods_info_to_latex(methods_info, save_path)


if __name__ == "__main__":
    for study_key in config.experiments.keys():
        process_experiment(study_key)
