import { getJsPsych } from "./jspsychSingleton.mjs";

const jsPsych = await getJsPsych();

export function getFromLastTrial(trialType, selector) {
  return jsPsych.data.get().filter({ trial_type: trialType }).last().select(selector).values[0];
}

export function countNonTimelineTrials(obj) {
  let count = 0;

  function traverse(obj, multiplier = 1) {
    if (Array.isArray(obj)) {
      obj.forEach((item) => traverse(item, multiplier));
    } else if (obj !== null && typeof obj === "object") {
      if (obj.timeline) {
        const innerMultiplier = obj.timeline_variables ? obj.timeline_variables.length : 1;
        traverse(obj.timeline, multiplier * innerMultiplier);
      } else {
        count += multiplier;
      }
    }
  }

  traverse(obj);
  return count;
}

/* #region jsPsych Trial Templates */

function makeTrialTemplates() {
  function preloadTrial(images) {
    return {
      type: jsPsychPreload,
      images: images,
    };
  }

  function instructionsTrial(pages, allow_keys = true) {
    return {
      type: jsPsychInstructions,
      pages: pages,
      show_clickable_nav: true,
      allow_keys: allow_keys,
      allow_backward: true,
    };
  }

  function promptTrial(prompt, delay = 0) {
    return {
      type: jsPsychHtmlButtonResponse,
      stimulus: `<p id="prompt">${prompt}</p>`,
      choices: ["Start"],
      enable_button_after: delay,
    };
  }

  function htmlButtonResponseTrial(stimulus, choices) {
    return {
      type: jsPsychHtmlButtonResponse,
      stimulus: stimulus,
      choices: choices,
      margin_vertical: "20px",
    };
  }

  function fullScreenTrial() {
    return {
      type: jsPsychFullscreen,
      fullscreen_mode: true,
    };
  }

  function browserCheckTrial() {
    return {
      type: jsPsychBrowserCheck,
    };
  }

  function preSurveyMessage() {
    return {
      type: jsPsychHtmlButtonResponse,
      stimulus: () => {
        const bonus = _.sum(jsPsych.data.get().filter({ trial_type: 'quicksand-planner' }).select('bonus').values);
        return `<p>You've completed the experiment! You will gain a bonus payment of $${Number(bonus.toFixed(2))}.\
        </br>On the next page, you'll be shown a brief set of questions about how the experiment went.\
        </br>Once you submit your answers, you'll be redirected back to Prolific and credited for participation.</p>`
      },
      choices: ["Continue"],
      margin_vertical: "20px",  
    }
  };

  function goodbyeTrial(completion_url) {
    return {
      type: jsPsychInstructions,
      pages: () => {
        return [
          `<p>Thanks for participating in our experiment!</p>\
          <p>Please click the <em>Submit</em> button to complete the study.</p>
          <p>Once you click <em>Submit</em>, you will be redirected to Prolific and receive credit for your participation.</p>`,
        ];
      },
      show_clickable_nav: true,
      allow_backward: false,
      button_label_next: "< Submit",
      on_finish: () => {
        window.onbeforeunload = null;
        document.onfullscreenchange = null;
        window.open(completion_url, "_self");
      },
    };
  }

  function dataPipeTrial() {
    return {
      type: jsPsychPipe,
      action: "save",
      experiment_id: gs.study_metadata.datapipe_experiment_id,
      filename: `${gs.study_metadata.project}-${gs.study_metadata.experiment}-${gs.study_metadata.iteration_name}-${gs.study_metadata.condition}-${gs.session_data.gameID}.csv`,
      data_string: () => jsPsych.data.get().ignore("environment").csv(),
    };
  }

  const trialTemplates = {};
  trialTemplates.promptTrial = promptTrial;
  trialTemplates.preloadTrial = preloadTrial;
  trialTemplates.instructionsTrial = instructionsTrial;
  trialTemplates.fullScreenTrial = fullScreenTrial;
  trialTemplates.browserCheckTrial = browserCheckTrial;
  trialTemplates.goodbyeTrial = goodbyeTrial;
  trialTemplates.htmlButtonResponseTrial = htmlButtonResponseTrial;
  trialTemplates.dataPipeTrial = dataPipeTrial;
  trialTemplates.preSurveyMessage = preSurveyMessage;

  return trialTemplates;
}

export const trialTemplates = makeTrialTemplates();

/* #endregion */

/* #region exit survey */

export const exitSurvey = {
  type: jsPsychSurvey,
  pages: [
    [
      {
        type: "html",
        prompt: "Please answer the following questions:",
      },
      {
        type: "multi-choice",
        name: "participantGender",
        prompt: "What is your gender?",
        options: ["Male", "Female", "Non-binary", "Other"],
        columns: 0,
        required: true,
      },
      {
        type: "text",
        name: "participantYears",
        prompt: "How many years old are you?",
        placeholder: "18",
        textbox_columns: 5,
        required: true,
      },
      {
        type: "multi-choice",
        name: "participantRace",
        prompt: "What is your race?",
        options: [
          "White",
          "Black/African American",
          "American Indian/Alaska Native",
          "Asian",
          "Native Hawaiian/Pacific Islander",
          "Multiracial/Mixed",
          "Other",
        ],
        columns: 0,
        required: true,
      },
      {
        type: "multi-choice",
        name: "participantEthnicity",
        prompt: "What is your ethnicity?",
        options: ["Hispanic", "Non-Hispanic"],
        columns: 0,
        required: true,
      },
      {
        type: "multi-choice",
        name: "inputDevice",
        prompt:
          "Which of the following devices did you use to complete this study?",
        options: ["Mouse", "Trackpad", "Touch Screen", "Stylus", "Other"],
        columns: 0,
        required: true,
      },
      {
        type: "likert",
        name: "judgedDifficulty",
        prompt: "How difficult did you find this study?",
        likert_scale_min_label: "Very Easy",
        likert_scale_max_label: "Very Hard",
        likert_scale_values: [
          { value: 1 },
          { value: 2 },
          { value: 3 },
          { value: 4 },
          { value: 5 },
        ],
        required: true,
      },
      {
        type: "likert",
        name: "participantEffort",
        prompt:
          "How much effort did you put into the game? Your response will not effect your final compensation.",
        likert_scale_min_label: "Low Effort",
        likert_scale_max_label: "High Effort",
        likert_scale_values: [
          { value: 1 },
          { value: 2 },
          { value: 3 },
          { value: 4 },
          { value: 5 },
        ],
        required: true,
      },
      {
        type: "text",
        name: "participantComments",
        prompt:
          "What factors influenced how you decided to respond? Do you have any other comments or feedback to share with us about your experience?",
        placeholder: "I had a lot of fun!",
        textbox_rows: 4,
        required: false,
      },
      {
        type: "text",
        name: "TechnicalDifficultiesFreeResp",
        prompt:
          "If you encountered any technical difficulties, please briefly describe the issue.",
        placeholder: "I did not encounter any technical difficulities.",
        textbox_rows: 4,
        required: false,
      },
    ],
  ],
  on_start: function () {
    gs.session_data.endExperimentTS = Date.now();
  },
  on_finish: function () {
    jsPsych.data.write(gs.session_data);
  }
};

/* #endregion */

/* #region jsPsych Quicksand Setup */

function manhattanDistance(pos1, pos2) {
  return Math.abs(pos1[0] - pos2[0]) + Math.abs(pos1[1] - pos2[1]);
}

export function getTilesAtDistance(path, dist, goal) {
  return path.filter((tile) => manhattanDistance(tile, goal) === dist);
}

function groupSimulateTrials(simulateTrials, numPlannerTrials) {
  const simulateTrialsPerPlanner = Math.floor(
    simulateTrials.length / numPlannerTrials
  );
  const groupedSimulateTrials = [];
  for (let i = 0; i < numPlannerTrials; i++) {
    groupedSimulateTrials.push(
      simulateTrials.slice(
        i * simulateTrialsPerPlanner,
        (i + 1) * simulateTrialsPerPlanner
      )
    );
  }
  return groupedSimulateTrials;
}

function addPropertyToTrials(timeline, property, value) {
  timeline.forEach(trial => {
    if (trial.timeline) {
      // If the trial contains a nested timeline, recursively add value to property
      addPropertyToTrials(trial.timeline, property, value);
    } else {
      // Otherwise, add the UUID to this trial
      const original_on_finish = trial.on_finish;
      trial.on_finish = function(data) {
        if (original_on_finish) {
          original_on_finish(data);
        }
        data[property] = value;
      };
    }
  });
}

export function createQuicksandProcedure(config, condition) {
  const quicksandProcedure = {
    timeline: [],
  };

  _.shuffle(config.worlds).forEach((world, i) => {
    const worldID = UUID();
    const worldTimeline = {
      timeline: [],
      on_timeline_start: () => {
        if (gs.game_info.limit_steps) { gs.game_info.total_steps = world.metadata.width + world.metadata.height; }
        gs.background_color.randColor = convertToHSL({ 
          h: gs.background_color.default.h + randSign() * Math.random() * gs.background_color.hue_range, 
          s: gs.background_color.default.s + randSign() * Math.random() * gs.background_color.saturation_range,
          l: gs.background_color.default.l + randSign() * Math.random() * gs.background_color.lightness_range
        });
        document.body.style.backgroundColor = gs.background_color.randColor;
        document.getElementById("jspsych-progressbar-container").style.backgroundColor = gs.background_color.randColor;
      },
    };

    // Learning phase
    const plannerTrials = _.shuffle(world.learning_phase.planner_trials);
    const plannerInstances = _.shuffle(world.learning_phase.planner_instances);
    plannerTrials.forEach((trial, index) => { trial.instance = plannerInstances[index] });
    const simulateTrials = _.shuffle(world.learning_phase.simulate_trials);
    let learningTrials;
    if (condition === "observation") {
      learningTrials = plannerTrials.map((planner) => ({ plannerTrial: planner, simulateTrials: [] }));
    } else {
      const groupedSimulateTrials = groupSimulateTrials(simulateTrials, plannerTrials.length);
      learningTrials = _.zip(plannerTrials, groupedSimulateTrials).map(
        ([plan, simulate]) => ({ plannerTrial: plan, simulateTrials: simulate })
      );
    }

    const newWorldPrompt = trialTemplates.promptTrial(
      `${i === 0 ? "Welcome! You have landed in a new desert." : "Great work! You have landed in yet another new desert. Use this time to take a break before proceeding to the next desert."}</br>\
      In this desert, some locations are more likely to be quicksand than others. ${i === 0 ? '' : 'The locations likely to be quicksand are different from the previous desert.'} </br></br>\
      Quicksand will slow you down, so avoid it if you can.</br>\
      Plan your path to ensure the agent reaches the goal as quickly as possible.`,
      i === 0 ? 1000 : 10000
    );
    const learningPhase = {
      timeline: [newWorldPrompt, { type: jsPsychQuicksandSetup, spec: world }],
    };

    learningTrials.forEach((learningTrial, j) => {
      const learningTimeline = {
        timeline: [
          {
            type: jsPsychQuicksandPlanner,
            feedback: true,
            start_position: jsPsych.timelineVariable("start_position"),
            goal_position: jsPsych.timelineVariable("goal_position"),
            wall_positions: jsPsych.timelineVariable("wall_positions"),
            environment_instance: jsPsych.timelineVariable("instance"),
            environment: () => { return getFromLastTrial("quicksand-setup", "environment") },
            sequence_uuid: () => { return getFromLastTrial("quicksand-setup", "sequence_uuid") },
            on_start: () => {
              $('#jspsych-progressbar-container').find('span:first').html(`Desert ${i + 1} of ${config.worlds.length} | Day ${j + 1} of ${learningTrials.length}`);
            }
          },
          {
            timeline: [
              {
                type: jsPsychQuicksandSimulate,
                start_position: function () {
                  const dist = jsPsych.timelineVariable("dist");
                  const goal = getFromLastTrial("quicksand-planner", "goal_position");
                  if (dist !== null) {
                    if (condition === "counterfactual") {
                      const lastPath = getFromLastTrial("quicksand-planner", "path");
                      const tilesAtDist = getTilesAtDistance(lastPath, dist, [goal.x, goal.y]);
                      if (tilesAtDist.length > 0) {
                        const sampledTile = tilesAtDist[Math.floor(Math.random() * tilesAtDist.length)];
                        return { x: sampledTile[0], y: sampledTile[1] };
                      } else {
                        console.error(`Error: no tiles at distance ${dist} from goal ${goal}`);
                      }
                    } else if (condition === "hypothetical") {
                      const wallPositions = getFromLastTrial("quicksand-planner", "wall_positions");
                      const possibleTiles = [];
                      for (let i = 0; i < world.metadata.height; i++) {
                        for (let j = 0; j < world.metadata.width; j++) {
                          if (!wallPositions.some((wall) => wall.x === j && wall.y === i)) {
                            possibleTiles.push([j, i]);
                          }
                        }
                      }
                      const tilesAtDist = getTilesAtDistance(possibleTiles, dist, [goal.x, goal.y]);
                      const validTiles = tilesAtDist.filter(tile =>
                        pathExists(tile, [goal.x, goal.y], wallPositions, world.metadata.width, world.metadata.height)
                      );
                      
                      const sampledTile = validTiles[Math.floor(Math.random() * validTiles.length)];
                      return { x: sampledTile[0], y: sampledTile[1] };
                    }
                  } else {
                    return jsPsych.timelineVariable("start_position");
                  }
                },
                goal_position: () => { return getFromLastTrial("quicksand-planner", "goal_position"); },
                wall_positions: () => { return getFromLastTrial("quicksand-planner", "wall_positions"); },
                condition: condition,
                dist: jsPsych.timelineVariable("dist"),
                environment: () => { return getFromLastTrial("quicksand-planner", "environment"); },
                sequence_uuid: () => { return getFromLastTrial("quicksand-setup", "sequence_uuid"); },
                instance_uuid: () => { return getFromLastTrial("quicksand-planner", "instance_uuid"); },
                on_start: () => {
                  if (condition === "hypothetical") {
                    $('#jspsych-progressbar-container').find('span:first').html(`Desert ${i + 1} of ${config.worlds.length} | Thought experiment`);
                  }
                }    
              },
            ],
            timeline_variables: learningTrial.simulateTrials.map((trial) => ({
              start_position: trial.start_position,
              goal_position: trial.goal_position,
              wall_positions: trial.wall_positions,
              dist: trial.dist,
            })),
            conditional_function: () => {
              return condition !== "observation";
            },
          },
        ],
        timeline_variables: [learningTrial.plannerTrial],
      };

      learningPhase.timeline.push(learningTimeline);
    });

    learningPhase.timeline.push({
      type: jsPsychQuicksandDestroy,
      environment: () => {
        return getFromLastTrial("quicksand-setup", "environment");
      },
      sequence_uuid: () => {
        return getFromLastTrial("quicksand-setup", "sequence_uuid");
      },
      post_trial_gap: 500,
    });
    worldTimeline.timeline.push(learningPhase);

    const hasNavigationTrials = world.evaluation_phase.navigation_trials && world.evaluation_phase.navigation_trials.length > 0
    const hasProbabilityTrials = world.evaluation_phase.probability_trials && world.evaluation_phase.probability_trials.length > 0
    const hasInstanceTrials = world.evaluation_phase.instance_trials && world.evaluation_phase.instance_trials.length > 0
    if (hasNavigationTrials || hasProbabilityTrials || hasInstanceTrials) {
      // Evaluation phase
      const evaluationPhase = {
        timeline: [],
      };

      const evaluationPrompt = trialTemplates.promptTrial(
        "Great job! Now that you've gained some experience in this desert, we want to test your understanding."
      );
      evaluationPhase.timeline.push(evaluationPrompt);

      if (hasNavigationTrials) {
        const no_wall_trial = _.cloneDeep(
          world.evaluation_phase.navigation_trials[0]
        );
        no_wall_trial.wall_positions = [];
        const navigationTrials = _.shuffle(
          world.evaluation_phase.navigation_trials
        );
        navigationTrials.unshift(no_wall_trial);

        const navigationPrompt = trialTemplates.promptTrial(
          "In this section, you will plan a path so that the agent reaches the goal as quickly as possible.</br>\
          There may be walls that will block your way, so be mindful of them while planning your path."
        );

        evaluationPhase.timeline.push(navigationPrompt);
        evaluationPhase.timeline.push({
          type: jsPsychQuicksandSetup,
          spec: world,
          on_finish: (data) => {
            data.sequence_uuid = undefined;
          }
        });
        evaluationPhase.timeline.push({
          timeline: [
            {
              type: jsPsychQuicksandEvalNavigation,
              start_position: jsPsych.timelineVariable("start_position"),
              goal_position: jsPsych.timelineVariable("goal_position"),
              wall_positions: jsPsych.timelineVariable("wall_positions"),
              environment: function () {
                return getFromLastTrial("quicksand-setup", "environment");
              },
            },
          ],
          timeline_variables: navigationTrials.map((trial) => ({
            start_position: trial.start_position,
            goal_position: trial.goal_position,
            wall_positions: trial.wall_positions,
          })),
        });
        evaluationPhase.timeline.push({
          type: jsPsychQuicksandDestroy,
          environment: function () {
            return getFromLastTrial("quicksand-setup", "environment");
          },
          sequence_uuid: function () {
            return getFromLastTrial("quicksand-setup", "sequence_uuid");
          },
          post_trial_gap: 500,
        });
      }

      if (hasProbabilityTrials) {
        const probabilityPrompt = trialTemplates.promptTrial(
          "In this section, you will see two possible deserts where information about the quicksand is fully revealed.</br>\
          Each tile indicates the likelihood of it being quicksand: lighter tiles are less likely to be quicksand, and darker tiles are more likely.</br>\
          Use the slider to indicate which desert you think matches the one you experienced."
        );

        evaluationPhase.timeline.push(probabilityPrompt);
        evaluationPhase.timeline.push({
          timeline: [
            {
              type: jsPsychQuicksandEvalProbability,
              spec1: jsPsych.timelineVariable("spec1"),
              spec2: jsPsych.timelineVariable("spec2"),
            },
          ],
          timeline_variables: world.evaluation_phase.probability_trials.map(
            (trial) => ({
              spec1: trial.spec1,
              spec2: trial.spec2,
            })
          ),
        });
      }

      if (hasInstanceTrials) {
        const instancePrompt = trialTemplates.promptTrial(
          "In this section, you will see two instances of the desert.</br>\
          Each tile is either sand or quicksand, and all tiles are fully revealed to you.</br>\
          Use the slider to indicate which instance you think matches the one you experienced."
        );

        evaluationPhase.timeline.push(instancePrompt);
        evaluationPhase.timeline.push({
          timeline: [
            {
              type: jsPsychQuicksandEvalInstance,
              spec1: jsPsych.timelineVariable("spec1"),
              spec2: jsPsych.timelineVariable("spec2"),
            },
          ],
          timeline_variables: world.evaluation_phase.instance_trials.map(
            (trial) => ({
              spec1: trial.spec1,
              spec2: trial.spec2,
            })
          ),
        });
      }
      worldTimeline.timeline.push(evaluationPhase);
    }

    addPropertyToTrials(worldTimeline.timeline, 'worldID', worldID);
    quicksandProcedure.timeline.push(worldTimeline);
  });

  return quicksandProcedure;
}
/* #endregion */