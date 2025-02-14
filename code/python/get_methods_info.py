#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import re
import numpy as np
import pandas as pd


# In[2]:


criteria = {
    'project': 'counterfactual_learning',
    'experiment': 's2_quicksand',
    'iteration_name': 'fullstudy',
}

project_dir = os.path.abspath('../../..')
data_dir = os.path.join(project_dir, 'data', criteria['experiment'], criteria['iteration_name'])

session_df = pd.read_csv(os.path.join(data_dir, 'session_data.csv'))
world_df = pd.read_csv(os.path.join(data_dir, 'world_data.csv'))
quicksand_df = pd.read_csv(os.path.join(data_dir, 'trial_data.csv'))

session_df['experiment_duration_min'] = session_df.experiment_duration_ms / (1000 * 60)


# In[3]:


demographics_categorical = lambda df, feature: {
    category.lower(): sum(df[feature] == category) for category in df[feature].unique()
}
demographics_ordinal = lambda df, feature: {
        'mean': round(df[feature].mean(), 1),
        'median': int(df[feature].median()),
        'min': df[feature].min(),
        'max': df[feature].max(),
}

demographics_ratio = lambda df, feature: {
        'mean': round(df[feature].mean(), 1),
        'median': round(df[feature].median(), 1),
        'min': round(df[feature].min(), 1),
        'max': round(df[feature].max(), 1),
}


# In[4]:


methods_info = {
    'basePay': '3',
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
methods_info


# In[5]:


def to_camel_case(key):
    key = re.sub(r'[-/\s]+(.)', lambda x: x.group(1).upper(), key)  # Convert spaces, dashes, and slashes to camelCase
    return key

def write_latex_commands(d, prefix='', f=None):
    for key, value in d.items():
        key = to_camel_case(key)
        if isinstance(value, dict):
            write_latex_commands(value, prefix + key, f)
        else:
            f.write(f"\\newcommand{{\\{prefix}{key}}}{{{value}}}\n")


# In[6]:


with open(os.path.join(data_dir, 'methods_vars.tex'), 'w') as f:
    write_latex_commands(methods_info, f=f)

