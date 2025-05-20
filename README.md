# Learning to Plan from Actual and Counterfactual Experiences

**Justin Yang, Tobias Gerstenberg**

Presented at the 47th Annual Meeting of the Cognitive Science Society (2025; San Fransisco, CA).

[Link to paper](https://github.com/cicl-stanford/counterfactual_learning_cogsci2025/blob/main/counterfactual_learning_cogsci2025.pdf)

```
@inproceedings{yang2025counterfactuallearning,
  title = {Learning to Plan from Actual and Counterfactual Experiences},
  booktitle = {Proceedings of the 47th {Annual} {Conference} of the {Cognitive} {Science} {Society}},
  author = {Yang, Justin and Gerstenberg, Tobias},
  year = {2025},
}
```

**Contents:**

* [Overview](#overview)
* [Experiment pre-registration & demo](#experiment-pre-registration--demo)
* [Repository structure](#repository-structure)
* [Set up](#set-up)
* [Experiments](#experiments)
* [Empirical analyses](#empirical-analyses)
* [Models](#models)
* [CRediT author statement](#credit-author-statement)



## Overview

Our ability to plan and make effective decisions depends on an accurate mental model of the environment. 
While learning is often thought to depend on external observations, people can also improve their understanding by reasoning about past experiences. 
In this work, we examine whether counterfactual simulation enhances learning in environments where planning is straightforward but encoding new information is challenging. 
Across two studies, participants navigated gridworlds, learning to avoid hazardous tiles. Some participants engaged in counterfactual simulation, constructing alternative plans after observing navigation outcomes. Others learned purely from experience. 
While counterfactual paths contained fewer hazards than initial ones, we found reliable evidence across both studies that counterfactual simulation conferred no measurable advantage in either navigation performance or explicit environment learning. 
These findings shed new light on the scope of learning by thinking—suggesting that the mechanism by which counterfactual reasoning enhances learning might not be by encouraging deeper encoding of past experiences. 


## Experiment pre-registration & demo

The experiment reported in these results was pre-registered on the [Open Science Framework](https://help.osf.io/article/158-create-a-preregistration).
Our pre-registration can be found [here](https://osf.io/tzha7).

A demo of the experiment can be found [here](https://justintheyang.github.io/experiment_demos/counterfactual_learning/index.html).


## Repository structure

```
├── code
│   ├── experiments
│   ├── python
│       ├── quicksand
│   └── R
├── data/
├── figures
│   ├── results
│   └── schemas
```
* `/code`: code for experiments, as well as data processing and analyses reported in the paper.
    * `/experiments`: this folder contains web code to run demos of the experiment itself. 
    * `/python`: this folder contains python code for preprocessing  data and constructing measured variables.
        * `/quicksand`: this folder contains python code for the three baseline models used to contextualize performance results. 
    * `/R`: this folder contains R code for all results reported in the paper.

* `/data`: contains all behavioral and model data used in our analyses. Data can be created by following the set up scripts.

* `/figures`: folder containing all figures (and accompanying files) presented in the paper.
    * `/results`: results figures generated during analysis.
    * `/schemas`: overview figures illustrating the task.

## Set up

The project uses Python 3 (tested on 3.10). Also, R should be installed and added to the PATH. We recommend using conda to set up the analysis environment:
```
conda env create -f environment.yml
conda activate counterfactual_learning
```

With the python environment set up, the follwing command will download and preprocess all data, and then run all analyses: 
```
bash run_project.sh
```

## Experiments

<p align="center" style="font-size: smaller">
  <img width="80%" src="https://github.com/cicl-stanford/counterfactual_learning_cogsci2025/blob/main/figures/schemas/fig_experiment_cogsci.png"></img><br/>
  Experiment overview
</p>

Our experiment examined how people use counterfactual simulation to learn a better mental model of the environment. 
A detailed explanation of the experimental design procedure is documented in the [study preregistration](https://osf.io/tzha7/?view_only=989096283435445fa1d72d472ceafc9f).

Participants navigated a series of $8x3$ grid worlds whose cells were either sand or quicksand. 
Some cells were walls, blocking access to that tile. 
They were asked to make a path from start to goal, avoiding as much quicksand as possible. 
The participants had to plan a full sequence from the start location, without any knowledge of whether a tile was quicksand. 
They instead learned how frequently a tile was quicksand through repeated interactions in the environment. 

Demos for each experiment are available [here](https://justintheyang.github.io/experiment_demos/counterfactual_learning/index.html).

### Study 1
Participants completed 15 *experience* trials in a block. 

<p align="center" style="font-size: smaller">
  <img width="75%" src="https://github.com/cicl-stanford/counterfactual_learning_cogsci2025/blob/main/code/experiments/s1_quicksand/assets/instructions/figs/6_make_a_plan.gif?raw=true"></img><br/>
  Example experience trial.
</p>

They were assigned to one of three between-subjects conditions:
- In the *experience* condition, this is all they did. 
- In the *hypothetical + experience* condition, after each experience trial they completed a hypothetical trial:

<p align="center" style="font-size: smaller">
  <img width="75%" src="https://github.com/cicl-stanford/counterfactual_learning_cogsci2025/blob/main/code/experiments/s1_quicksand/assets/instructions/figs/7h_hypothetical_plan.gif?raw=true"></img><br/>
  Example hypothetical trial.
</p>

- In the *counterfactual + experience* condition, after each experience trial they completed a counterfactual trial:
<p align="center" style="font-size: smaller">
  <img width="75%" src="https://github.com/cicl-stanford/counterfactual_learning_cogsci2025/blob/main/code/experiments/s1_quicksand/assets/instructions/figs/7c_counterfactual_plan.gif?raw=true"></img><br/>
  Example hypothetical trial.
</p>

Code for this experiment can be found in `code/experiments/s1_quicksand`.

### Study 2
Study 2 was nearly identical to the first, except that participants completed an *exam* trial after each each block, where they indicated whether they thought a tile was safe or unsafe:
<p align="center" style="font-size: smaller">
  <img width="75%" src="https://github.com/cicl-stanford/counterfactual_learning_cogsci2025/blob/main/code/experiments/s2_quicksand/assets/instructions/figs/8_exam_safe.gif?raw=true"></img><br/>
  Example exam trial.
</p>

Participants were also assigned to either the *experience* or *counterfactual + experience* condition.

Code for this experiment can be found in `code/experiments/s2_quicksand`.

## Empirical analyses
To reproduce the empirical analyses found in the paper, you can run
```
bash run_project.sh
```

Here is an overview of what each analysis file does:
* `/python`
    * `config.py`: contains metadata and global variables useful for setting up the clean datasets.
    * `get_data.py`: converts the raw participant data stored using [`jspsych-datapipe`](https://pipe.jspsych.org/) into clean dataframes.
    * `get_methods_info.py`: generates `*.tex` files containing variables used for methods reporting in the manuscript. 
    * `compute_derived_variables.py`: computes the derived variables defined in the [preregistration](https://osf.io/tzha7/?view_only=989096283435445fa1d72d472ceafc9f).
    * `compute_model_predictions.py`: computes model predictions for the three baseline models used in Figure 3 of the paper.
    * `osf_data_handler.py`: interfaces with the OSF API to pull data from the experiment. 
* `/R`
    * `cogsci_results.Rmd`: runs all bayesian regressions and generates result figures using the tidy dataframes from the above python scripts.

For each experiment, the data is separated into three CSV files:
- `session_data.csv`: contains session-level data, such as participant browser information and demographics.
- `world_data.csv`: contains block-level data, such as participant exam trial responses and ground truth environment probabilities. 
- `trial_data.csv`: contains trial-level data (i.e., one row for each experience, counterfactual, or hypothetical navigation trial), such as the path taken to navigate to the goal. 
- `model_predictions.csv`: contains model predictions (one row for each models' prediction on a navigation trial, or 3 rows for each trial id), such as the predicted path taken to navigate to the goal. 

A detailed description of each column can be found in the preregistration. 



## CRediT author statement

*What is a [CRediT author statement](https://www.elsevier.com/authors/policies-and-guidelines/credit-author-statement)?*

- **Justin Yang:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization, Supervision, Project administration
- **Tobias Gerstenberg:** Conceptualization, Methodology, Writing - Review & Editing, Supervision, Project administration, Funding acquisition

