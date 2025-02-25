# Learning to Plan from Observed and Counterfactual Experiences

**Justin Yang, Tobias Gerstenberg**

Presented at the 47th Annual Meeting of the Cognitive Science Society (2025; San Fransisco, CA).

[Link to paper](https://github.com/cicl-stanford/counterfactual_learning_cogsci2025/blob/master/writeup/counterfactual_learning_cogsci2025.pdf)

```
@inproceedings{yang2025counterfactuallearning,
  title = {Learning to Plan from Observed and Counterfactual Experiences},
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
Our pre-registration can be found [here](https://anonymous.4open.science/r/cogsci2025-0348/README.md).

A demo of the experiment can be found [here](https://justintheyang.github.io/experiment_demos/counterfactual_learning/index.html).


## Repository structure

```
├── code
│   ├── experiments
│   ├── python
│   └── R
├── data/
├── figures
│   ├── results
│   └── schemas
```
* `/code`: code for experiments, as well as data processing and analyses reported in the paper.
    * `/experiments`: this folder contains web code to run demos of the experiment itself. 
    * `/python`: this folder contains python code for preprocessing  data and constructing measured variables.
    * `/R`: this folder contains R code for all results reported in the paper.

* `/data`: contains all behavioral and model data used in our analyses. Data can be created by following the set up scripts.

* `/figures`: folder containing all figures (and accompanying files) presented in the paper.
    * `/results`: results figures generated during analysis.
    * `/schemas`: overview figures illustrating the task.

## Set up
TODO: here we talk about how to run everything to get all results. 

## Experiments
TODO: here we talk about the experiment (a high level overview)

## Empirical analyses
TODO: here we outline the python and r files
"...we include all analyses in the reported in the paper" 

* `/python`
    * `get_data.py`: ...
    * `config.py`: ...
* `/R`
    * `...`


## CRediT author statement

*What is a [CRediT author statement](https://www.elsevier.com/authors/policies-and-guidelines/credit-author-statement)?*

- **Justin Yang:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data Curation, Writing - Original Draft, Writing - Review & Editing, Visualization, Supervision, Project administration
- **Tobias Gerstenberg:** Conceptualization, Methodology, Writing - Review & Editing, Supervision, Project administration, Funding acquisition

