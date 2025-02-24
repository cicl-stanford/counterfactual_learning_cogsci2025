# Experiments readme

## Quicksand experiment technical planning
For these studies, a few different components/phases go into forming the overall experiment. At a high level, the participant is learning how to traverse a gridworld, where each tile has a certain probability of being quicksand. They do not know which tiles are quicksand until they walk over them, and on each trial the quicksand will be reset according to the underlying tile probabilities. Their objective is to get from start to goal location as fast as possible, and to do so they much choose paths that are _not_ likely to be quicksand, as quicksand slows them down. 

In the **learning** phase, the participant plans a path to get from start to goal location by using their mouse to click and drag the path they want the agent to take. When ready, the participant submits their choice and watches the agent traverse that path. When the agent traverses a tile with _quicksand_, the agent will move more slowly through that path. After each time the participant submits this **observation** trial:
- in the **counterfactual** learning condition, they will see the same environment, with the path that was traversed revealed. They will then be asked to imagine starting from various other locations and asked to make a path from that location to the goal. After they submit their choice, the agent will _not_ traverse their path.
- in the **hypothetical** learning condition, they will see a _blank_ environment (i.e., the path traversed is no longer shown) and asked to imagine starting from various locations. Simiilar to the counterfactual learning condition, they will make a path from that starting location, and similarly they will _not_ recieve feedback on the agent's performance.

After completing several sets of these trials (observation + mental simulation), they will then be tested on their understanding of the environment in the **evaluation** phase. In the **evaluation** phase, participantswill be given two kinds of exams: 
- to probe quicksand navigation **performance**, we will ask them to make paths from various start to goal locations.
- to probe how well they learned the underlying quicksand **probability distribution**, we will present them with several quicksand environments where the state of all (or some) of the tiles are revealed, and they will make a choice about which gridworld is the most likely to have occured.

I think the end goal is to have a jsPsych experiment with the plugins {quicksand-traverse, quicksand-simulate, quicksand-eval-path, quicksand-eval-sandland}. It makes most sense to use nested timelines, roughly looking like this:

```
var learning_phase = {
  timeline: [
    {
      type: jsPsychQuicksandTraverse,
      gridworld_spec: jsPsych.timelineVariable('gridworld_spec'),
    },
    {
      timeline: [
        {
          type: jsPsychQuicksandSimulate,
          condition: jsPsych.timelineVariable('condition'), // counterfactual or hypothetical
          gridworld_spec: function() { jsPsych.data.get().filter({trial_type: 'jsPsychQuicksandTraverse'}).last().select('gridworld_spec').values[0] },
          path: function() { jsPsych.data.get().filter({trial_type: 'jsPsychQuicksandTraverse'}).last().select('path').values[0] },
          start_location: jsPsych.timelineVariable('start_location'),
          goal_location: jsPsych.timelineVariable('goal_location'),
        },  
      ],
      timeline_variables: [
        {start_location: '...', goal_location: '...'},
        {start_location: '...', goal_location: '...'},
        {start_location: '...', goal_location: '...'},
      ]
    }
  ], 
  timeline_variables: [
    {gridworld_spec: '...'},
    {gridworld_spec: '...'},
    {gridworld_spec: '...'},
  ]
}

var evaluation_phase = {
  timeline: [
    {
      timeline: [
        {
          type: jsPsychQuicksandEvalPath,
          gridworld_spec: jsPsych.timelineVariable('gridworld_spec'),
          start_location: jsPsych.timelineVariable('start_location'),
          goal_location: jsPsych.timelineVariable('goal_location'),
        },
      ],
      timeline_variables: [
        {gridworld_spec: '...', start_location: '...', goal_location: '...'},
        {gridworld_spec: '...', start_location: '...', goal_location: '...'},
        {gridworld_spec: '...', start_location: '...', goal_location: '...'},
      ]
    },
    {
      timeline: [
        {
          type: jsPsychQuicksandEvalSandland,
          gridworld_specs: jsPsych.timelineVariable('gridworld_specs'), // list of gridworld specs
        },
      ],
      timeline_variables: [
        {gridworld_specs: ['...', '...', '...']},
        {gridworld_specs: ['...', '...', '...']},
        {gridworld_specs: ['...', '...', '...']},
      ]
    }
  ]
}
```
