import { practice_world, planner_practice, hypothetical_practice, counterfactual_practice } from "./practice_specs.mjs";
import { consent } from "./consent.mjs";
import { trialTemplates, getFromLastTrial, getTilesAtDistance } from "../../js/jspsych_folder/jspsych_utils.mjs";

export function setupInstructions(allow_keys) {
  const instructions = {}
  instructions.landingInstructions = trialTemplates.instructionsTrial(landingInstructions, allow_keys);
  instructions.consent = consent;
  instructions.preload = trialTemplates.preloadTrial(preloadFiles);
  instructions.planningInstructions = trialTemplates.instructionsTrial(planningInstructions, allow_keys);
  instructions.planningInstructions.on_start = () => { gs.session_data.startPlanInstructionTS = Date.now() };
  instructions.planningPractice = planningPractice;
  instructions.planningPractice.on_start = () => { gs.session_data.startPlanPracticeTS = Date.now() };
  instructions.planningPracticeConclusion = trialTemplates.htmlButtonResponseTrial(planningPracticeConclusion, ["Continue"]);
  instructions.planningComprehensionLoop = planningComprehensionLoop;

  if (gs.study_metadata.condition === "hypothetical") {
    instructions.simulationInstructions = trialTemplates.instructionsTrial(hypotheticalInstructions, allow_keys);
    instructions.simulationPractice = hypotheticalPractice;
    instructions.simulationPracticeConclusion = trialTemplates.htmlButtonResponseTrial(simulationPracticeConclusion, ["Continue"]);
    instructions.simulationComprehensionLoop = hypotheticalComprehensionLoop;
  } else if (gs.study_metadata.condition === "counterfactual") {
    instructions.simulationInstructions = trialTemplates.instructionsTrial(counterfactualInstructions, allow_keys);
    instructions.simulationPractice = counterfactualPractice;
    instructions.simulationPracticeConclusion = trialTemplates.htmlButtonResponseTrial(simulationPracticeConclusion, ["Continue"]);
    instructions.simulationComprehensionLoop = counterfactualComprehensionLoop;
  }

  if (gs.study_metadata.condition !== "observation") {
    instructions.simulationInstructions.on_start = () => { gs.session_data.startSimInstructionTS = Date.now() };
    instructions.simulationPractice.on_start = () => { gs.session_data.startSimPracticeTS = Date.now() };
  }

  instructions.examInstructions = trialTemplates.instructionsTrial(examInstructions, allow_keys);
  instructions.navigationInstructions = trialTemplates.instructionsTrial(navigationInstructions, allow_keys);
  // instructions.probabilityInstructions = trialTemplates.instructionsTrial(probabilityInstructions, allow_keys);
  // instructions.instanceInstructions = trialTemplates.instructionsTrial(instanceInstructions, allow_keys);

  instructions.instructionsConclusion = trialTemplates.htmlButtonResponseTrial(instructionsConclusion, ["Continue"]);
  instructions.instructionsConclusion.on_finish = () => { gs.session_data.startExperimentTS = Date.now() };

  return instructions;
}

const preloadFiles = [
  "assets/instructions/figs/1_agent_movement.gif",
  "assets/instructions/figs/2_agent_slowed_quicksand.gif",
  "assets/instructions/figs/3_blocking_walls.gif",
  "assets/instructions/figs/4_probabilistic_quicksand.png",
  "assets/instructions/figs/5_agent_reveal_tiles.gif",
  "assets/instructions/figs/6_make_a_plan.gif",
  "assets/instructions/figs/7h_hypothetical_plan.gif",
  "assets/instructions/figs/7c_counterfactual_plan.gif",
  // "assets/instructions/figs/8_eval_wall_navigation.gif",
];

const time = "20"
const landingInstructions = [
  `<p> Hello! Thank you for participating in this research.</p> \
  <p> We expect the study to last about ${time} minutes, including the time it takes to read these instructions.</p> \
  <i><p> Note: We recommend completing the study in Chrome. It has not been tested in other browsers.</p></i>`,
];

/* #region Planning Trial Instructions */

const start_text = `<span style="color: ${convertToHSL(gs.tile.colors.start)}; text-shadow: -0.5px 0 black, 0 0.5px black, 0.5px 0 black, 0 -0.5px black;">start</span>`
const goal_text = `<span style="color: ${convertToHSL(gs.tile.colors.goal)}; text-shadow: -0.5px 0 black, 0 0.5px black, 0.5px 0 black, 0 -0.5px black;">goal</span>`
//For example, in the grid on top below, the darker a tile is, the more likely it is to be quicksand
const planningInstructions = [
  `<p>In this study, your task is to help an agent navigate some deserts filled with quicksand!</p> \
  <p>You will make plans for the agent to go from start to goal location as quickly as possible.</p>`,
  
  `<p>Each desert is made up of a grid of tiles.</p> \
  <p>Your main task is to guide an agent from a ${start_text} location to a ${goal_text}.</p> \
  <img width="450" src="assets/instructions/figs/1_agent_movement.gif">`,
  
  `<p><b>Quicksand</b> is a hazard in this desert. It slows down your progress considerably.</p> \
  <p>Each tile in the desert represents a different spot in the terrain, with some being sand and others quicksand.</p> \
  <img width="450" src="assets/instructions/figs/2_agent_slowed_quicksand.gif">`,
  
  `<p>In addition to quicksand, some of the tiles may be <b>blocked</b> on certain days.</p> \
  <p>Roadblocks block the agent's path, so the agent must avoid them in order to reach the goal.</p> \
  <img width="450" src="assets/instructions/figs/3_blocking_walls.gif">`,

  `<p>The weather changes the terrain on each day. So just because there was quicksand on one day doesn't guarantee that the same tile will be quicksand on the next!</p> \
  <p>However, there are regular weather patterns that make some tiles quicksand more often than others.</p> \
  <p>For this study, some tiles will always be safe, while others are frequently quicksand. Specifically, the <b>light</b> tiles here are always safe, while the <b>brown</b> tiles are quicksand 80% of the time.</p> \
  <img width="1200" src="assets/instructions/figs/4_probabilistic_quicksand.png">`,

  `<p>You can't see which tiles are quicksand until the agent steps on them.</p> \
  <p>As the agent follows a path to the ${goal_text}, whether each tile is quicksand or not will be revealed.</p> \
  <img width="450" src="assets/instructions/figs/5_agent_reveal_tiles.gif">`,
    
  `<p>Your goal is to guide the agent to reach the ${goal_text} as quickly as possible. You will do this by planning a route from the agent's ${start_text} location to the ${goal_text}.</p> \
  <p>Click on tiles to add them to the path. Click on a tile in the path to remove all tiles after that point.</p> \
  <img width="500" src="assets/instructions/figs/6_make_a_plan.gif">\
  <p>When you finish making the plan, you will watch the agent follow the path you charted, and see where it gets slowed by quicksand.</p>`,
  
  `<p>To efficiently guide the agent, you must learn which tiles are likely to be quicksand and avoid them, while not making too many twists and turns which will make the agent's path too long!</p> \
  <p>You will be rewarded a bonus for making paths that avoid the quicksand. On each trial you can earn up to $0.04 by avoiding all quicksand, and will lose $0.01 bonus for each quicksand tile the agent encounters. So across all trials, you can earn a maximum up of $2.40!</p>`,

  `<p>Let\'s do an example together.</p> \
  <p>On the next screen, you\'ll see a grid. Click the tiles to plan a path to the ${goal_text} while avoiding quicksand.</p>`
];

const planningPractice = {
  timeline: [
    { type: jsPsychQuicksandSetup, spec: practice_world },
    {
      type: jsPsychQuicksandPlanner,
      start_position: planner_practice.start_position,
      goal_position: planner_practice.goal_position,
      wall_positions: planner_practice.wall_positions,
      environment: () => { return getFromLastTrial("quicksand-setup", "environment")},
      sequence_uuid: () => { return getFromLastTrial("quicksand-setup", "sequence_uuid")},
    },
    {
      type: jsPsychQuicksandDestroy,
      environment: () => { return getFromLastTrial("quicksand-setup", "environment") },
      sequence_uuid: () => { return getFromLastTrial("quicksand-setup", "sequence_uuid") },
      post_trial_gap: 500,
    }  
  ]
}

const planningPracticeConclusion = [
  `<p>Great job!</p> \
   <p>On the next screen, you\'ll be asked a few questions confirming that everything is crystal clear.</p>\
   <p>Please do your best. You will not be able to proceed to the next part of the experiment until you answer all questions correctly.</p>`
];

const planningComprehensionLoop = {
  timeline: [{
    type: jsPsychSurveyMultiChoice,
    preamble: "<strong>Comprehension Check</strong>",
    questions: [
      {
        prompt: "True or False: If a tile is quicksand on one day, it will definitely be quicksand on the next day.",
        name: "quicksandGeneration",
        options: [
          "True, if I see a tile as quicksand once, it will always be quicksand.",
          "False, the weather changes the terrain each day, so a tile that was quicksand one day may not be quicksand the next day."],
        required: true,
      },
      {
        prompt: "True or False: Some tiles tend to be quicksand more often then others.",
        name: "quicksandPatterns",
        options: [
          "True, some tiles are more likely to be quicksand than others.",
          "False, all tiles are equally likely to be quicksand."],
        required: true,
      },
      {
        prompt: "True or False: You are trying to spend as little time as possible making a plan for the agent.",
        name: "timeEfficiency",
        options: [
          "True, I need to make the plan as quickly as possible.",
          "False, I can take as little or as much time as I want."],
        required: true,
      },
      {
        prompt: "True or False: You are trying to make the agent take the fastest path to the goal.",
        name: "pathEfficiency",
        options: [
          "True, I want the agent to reach the goal as quickly as possible.",
          "False, it doesn't matter how long it takes for the agent to reach the goal."],
        required: true,
      },
      {
        prompt: "True or False: Quicksand slows the agent down.",
        name: "quicksandAvoidance",
        options: [
          "True, quicksand slows the agent down.",
          "False, quicksand doesn't affect the agent's speed."],
        required: true,
      },
      {
        prompt: "True or False: Some of the tiles are <b>never</b> quicksand.",
        name: "safeTiles",
        options: [
          "True, some of the tiles are never quicksand, so if a tile has quicksand you can be certain it is not safe.",
          "False, all tiles have at least a small chance of being quicksand."],
        required: true,
      }
    ]
  }],
  loop_function: (data) => {
    const responses = data.values()[0].response;
    if (responses.quicksandGeneration === "False, the weather changes the terrain each day, so a tile that was quicksand one day may not be quicksand the next day." &&
        responses.quicksandPatterns === "True, some tiles are more likely to be quicksand than others." &&
        responses.timeEfficiency === "False, I can take as little or as much time as I want." &&
        responses.pathEfficiency === "True, I want the agent to reach the goal as quickly as possible." &&
        responses.quicksandAvoidance === "True, quicksand slows the agent down." &&
        responses.safeTiles === "True, some of the tiles are never quicksand, so if a tile has quicksand you can be certain it is not safe.") {
      return false;
    } else {
      gs.session_data.comprehensionAttempts += 1;
      alert("One or more of your answers was incorrect. Please try again.");
      return true;
    }
  }
}

/* #endregion */

/* #region Simulation Trial Instructions */

const hypotheticalInstructions = [
  `<p>Great work! After observing the agent follow your path, you will engage in a thought experiment and consider what to do if the agent started at other points!</p>\
  <p>Supposing it is a new day in the desert, you will create a path to the goal that you believe would minimize the time it takes to reach the goal.</p>
  <img width="550" src="assets/instructions/figs/7h_hypothetical_plan.gif">`,

  `<p>On the next page, you will practice doing this: making a plan, observing the outcome, and thinking about efficient plans starting at other places.</p>`
];

const hypotheticalPractice = {
  timeline: [
    { type: jsPsychQuicksandSetup, spec: practice_world },
    {
      type: jsPsychQuicksandPlanner,
      start_position: planner_practice.start_position,
      goal_position: planner_practice.goal_position,
      wall_positions: planner_practice.wall_positions,
      environment: () => { return getFromLastTrial("quicksand-setup", "environment")},
      sequence_uuid: () => { return getFromLastTrial("quicksand-setup", "sequence_uuid")},
    },
    {
      type: jsPsychQuicksandSimulate,
      start_position: () => {
        const dist = hypothetical_practice.dist;
        const goal = planner_practice.goal_position;
        const allTiles = [];
        for (let i = 0; i < practice_world.metadata.height; i++) {
          for (let j = 0; j < practice_world.metadata.width; j++) {
            allTiles.push([j, i]);
          }
        }
        const tilesAtDist = getTilesAtDistance(allTiles, dist, [goal.x, goal.y,]);
        const sampledTile = tilesAtDist[Math.floor(Math.random() * tilesAtDist.length)];
        return { x: sampledTile[0], y: sampledTile[1] };
      },
      goal_position: () => { return getFromLastTrial("quicksand-planner", "goal_position") },
      wall_positions: () => { return getFromLastTrial("quicksand-planner", "wall_positions") },
      dist: hypothetical_practice.dist,
      condition: gs.study_metadata.condition,
      environment: () => { return getFromLastTrial("quicksand-setup", "environment")},
      sequence_uuid: () => { return getFromLastTrial("quicksand-setup", "sequence_uuid")}, 
      instance_uuid: () => { return getFromLastTrial("quicksand-planner", "instance_uuid")},
    },
    {
      type: jsPsychQuicksandDestroy,
      environment: () => { return getFromLastTrial("quicksand-setup", "environment") },
      sequence_uuid: () => { return getFromLastTrial("quicksand-setup", "sequence_uuid") },
      post_trial_gap: 500,
    }  
  ]
}

const hypotheticalComprehensionLoop = {
  timeline: [{
    type: jsPsychSurveyMultiChoice,
    preamble: "<strong>Comprehension Check</strong>",
    questions: [
      {
        prompt: "True or False: If a tile in the path I observed was quicksand, it will also be quicksand in this thought experiment.",
        name: "quicksandGeneration",
        options: [
          "True, if I saw a tile as quicksand, it will also be quicksand in the thought experiment.",
          "False, the weather changes the terrain each day, so the tile may not be quicksand because we are imagining a new day in the thought experiment."],
        required: true,
      }
    ],
  }],
  loop_function: (data) => {
    const responses = data.values()[0].response;
    if (responses.quicksandGeneration === "False, the weather changes the terrain each day, so the tile may not be quicksand because we are imagining a new day in the thought experiment.") {
      return false;
    } else {
      gs.session_data.comprehensionAttempts += 1;
      alert("Your answers was incorrect. Please try again.");
      return true;
    }
  }
}

const counterfactualInstructions = [
  `<p>Great work! After observing the agent follow your path, you will take some time to reflect on the plan you made and consider better alternatives!</p>\
  <p>Starting in the path you originally planned, you will create a path to the goal that you believe would have been more effective given what the agent encountered.</p>
  <img width="550" src="assets/instructions/figs/7c_counterfactual_plan.gif">`,

  // `<p>You are welcome to use the tiles in the original path if you think it will lead to a better outcome. However, you should not simply retrace the same path.</p>
  // <img width="450" src="assets/instructions/figs/6c2_counterfactual_retrace.png">`,

  `<p>On the next page, you will practice doing this: making a plan, observing the outcome, and making an alternative route that you believe would have been better for the agent to follow!</p>`
];

const counterfactualPractice = {
  timeline: [
    { type: jsPsychQuicksandSetup, spec: practice_world },
    {
      type: jsPsychQuicksandPlanner,
      start_position: planner_practice.start_position,
      goal_position: planner_practice.goal_position,
      wall_positions: planner_practice.wall_positions,
      environment: () => { return getFromLastTrial("quicksand-setup", "environment")},
      sequence_uuid: () => { return getFromLastTrial("quicksand-setup", "sequence_uuid")},
    },
    {
      type: jsPsychQuicksandSimulate,
      start_position: () => {
        const dist = counterfactual_practice.dist;
        const goal = counterfactual_practice.goal_position;
        const lastPath = getFromLastTrial("quicksand-planner", "path");
        const tilesAtDist = getTilesAtDistance(lastPath, dist, [goal.x, goal.y,]);
        const sampledTile = tilesAtDist[Math.floor(Math.random() * tilesAtDist.length)];
        return { x: sampledTile[0], y: sampledTile[1] };
      },
      goal_position: () => { return getFromLastTrial("quicksand-planner", "goal_position") },
      wall_positions: () => { return getFromLastTrial("quicksand-planner", "wall_positions") },
      dist: counterfactual_practice.dist,
      condition: gs.study_metadata.condition,
      environment: () => { return getFromLastTrial("quicksand-setup", "environment")},
      sequence_uuid: () => { return getFromLastTrial("quicksand-setup", "sequence_uuid")},
      instance_uuid: () => { return getFromLastTrial("quicksand-planner", "instance_uuid")},
    },
    {
      type: jsPsychQuicksandDestroy,
      environment: () => { return getFromLastTrial("quicksand-setup", "environment") },
      sequence_uuid: () => { return getFromLastTrial("quicksand-setup", "sequence_uuid") },
      post_trial_gap: 500,
    }
  ]
}

const counterfactualComprehensionLoop = {
  timeline: [{
    type: jsPsychSurveyMultiChoice,
    preamble: "<strong>Comprehension Check</strong>",
    questions: [
      {
        prompt: "True or False: If a tile in the path I observed was quicksand, it will also be quicksand in this reflection.",
        name: "quicksandGeneration",
        options: [
          "True, if I saw a tile as quicksand, it will also be quicksand in this reflection.",
          "False, the tile we just observed might not be quicksand in this reflection."],
        required: true,
      }
    ],
  }],
  loop_function: (data) => {
    const responses = data.values()[0].response;
    if (responses.quicksandGeneration === "True, if I saw a tile as quicksand, it will also be quicksand in this reflection.") {
      return false;
    } else {
      gs.session_data.comprehensionAttempts += 1;
      alert("Your answers was incorrect. Please try again.");
      return true;
    }
  }

}

const simulationPracticeConclusion = [
  `<p>Great job!</p> \
   <p>On the next screen, you\'ll be asked a few questions confirming that everything is crystal clear.</p>\
   <p>Please do your best. You will not be able to proceed to the next part of the experiment until you answer all questions correctly.</p>`
];

/* #endregion */

// NOT USED IN THIS STUDY
/* #region Exam Trial Instructions */

const examInstructions = [
  `<p>There's one last thing you need to know.</p> \
  <p>After you've gained some experience helping the agent navigate the desert, we will test your understanding of which tiles are likely to be quicksand.</p>`,
];

const navigationInstructions = [
  `<p>Like before, you will plan paths to help the agent reach the goal as quickly as possible.</p>\
  <p>This time, there may be roadblocks that will block your way, so be mindful of them while planning your path.</p> \
  <img width="550" src="assets/instructions/figs/8_eval_wall_navigation.gif">`
];

const probabilityInstructions = [
  `<p>Additionally, you will see two possible deserts and asked to determine which one you were in. For both these deserts, how often each tile is quicksand is fully revealed. Lighter tiles are less likely to be quicksand, and darker tiles are more likely. You will use the slider to indicate which desert you think matches the one you were in.</p> \
  <img width="450" src="assets/instructions/two_shaded_grids.png">`
];

const instanceInstructions = [
  `<p>Finally, you will see two specific days from two possible deserts, and asked to determine which one you were in. For both deserts, and all tiles are fully revealed to you, and will either be sand or quicksand. Use the slider to indicate which grid you think came from the desert you were in.</p> \
  <img width="450" src="assets/instructions/two_grids_sand_quicksand.png">`
];

/* #endregion */

const instructionsConclusion = [
  `<p>Great job!</p> \
  <p>You've learned everything you need to know about to help guide the agent to the goal.</p> \
  <p>Please click the continue button to get started.</p>`
]
