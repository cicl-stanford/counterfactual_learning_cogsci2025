import { practice_world, planner_practice, counterfactual_practice } from "./practice_specs.mjs";
import { consent } from "./consent.mjs";
import { trialTemplates, getFromLastTrial } from "../../js/jspsych_folder/jspsych_utils.mjs";

export function setupInstructions(allow_keys) {
  const instructions = {}
  instructions.landingInstructions = trialTemplates.instructionsTrial(landingInstructions, allow_keys);
  instructions.consent = consent;
  instructions.preload = trialTemplates.preloadTrial(preloadFiles);
  instructions.planningComprehensionLoop = planningComprehensionLoop;
  instructions.planningPractice = planningPractice;
  instructions.counterfactualComprehensionLoop = counterfactualComprehensionLoop;
  instructions.counterfactualPractice = counterfactualPractice;
  instructions.examInstructions = examInstructions;
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
  "assets/instructions/figs/7c_counterfactual_plan.gif",
];

const time = "15"
const landingInstructions = [
  `<p> Hello! Thank you for participating in this research.</p> \
  <p> We expect the study to last about ${time} minutes, including the time it takes to read these instructions.</p> \
  <i><p> Note: We recommend completing the study in Chrome. It has not been tested in other browsers.</p></i>`,
];

/* #region Planning Trial Instructions */

const start_text = `<span style="color: ${convertToHSL(gs.tile.colors.start)}; text-shadow: -0.5px 0 black, 0 0.5px black, 0.5px 0 black, 0 -0.5px black;">start</span>`
const goal_text = `<span style="color: ${convertToHSL(gs.tile.colors.goal)}; text-shadow: -0.5px 0 black, 0 0.5px black, 0.5px 0 black, 0 -0.5px black;">goal</span>`
//For example, in the grid on top below, the darker a tile is, the more likely it is to be quicksand
const planningInstructions = trialTemplates.instructionsTrial([
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
  <p>For this study, some tiles will <b>always</b> be safe, while others are frequently quicksand. Specifically, the <b>light</b> tiles here are always safe, while the <b>brown</b> tiles are quicksand 80% of the time.</p> \
  <img width="1200" src="assets/instructions/figs/4_probabilistic_quicksand.png">`,

  `<p>You can't see which tiles are quicksand until the agent steps on them.</p> \
  <p>As the agent follows a path to the ${goal_text}, whether each tile is quicksand or not will be revealed.</p> \
  <img width="450" src="assets/instructions/figs/5_agent_reveal_tiles.gif">`,
    
  `<p>Your goal is to plan an efficient route from the agent's ${start_text} location to the ${goal_text}.</p> \
  <p>Click on tiles to add them to the path. Click on a tile in the path to remove all tiles after that point.</p> \
  <img width="500" src="assets/instructions/figs/6_make_a_plan.gif">\
  <p>When you finish making the plan, you will watch the agent follow the path you charted, and see where it gets slowed by quicksand.</p>`,
  
  `<p>To efficiently guide the agent, you must avoid the tiles that are likely to be quicksand, while not making the agent's path too long!</p> \
  <p>You will be rewarded a bonus for making paths that avoid quicksand. On each trial you can earn up to $0.04 by avoiding all quicksand, and will lose $0.01 of bonus for each quicksand tile the agent encounters. So across all trials, you can earn a maximum up of $1.20!</p>
  <p>So the get the maximum bonus, you should aim to <b>minimize the number of quicksand tiles the agent encounters</b>!</p>
  `,

  `<p>On the next screen, you\'ll be asked a few questions confirming that everything is crystal clear.</p>\
  <p>Please do your best. You will not be able to proceed to the next part of the experiment until you answer all questions correctly.</p>`], true); // allow_keys for now always true

planningInstructions.on_start = () => { gs.session_data.startPlanInstructionTS = Date.now() };

const planningComprehensionFailedTrial = {
  timeline: [
    {
      type: jsPsychHtmlButtonResponse,
      stimulus: [
        `<p>Unfortunately, you did not answer all the questions correctly.</p> \
        <p>Please read the instructions carefully and try again.</p>`],
      choices: ["Try Again"],
      margin_vertical: "20px"
    }
  ],
  conditional_function: () => {
    const responses = getFromLastTrial("survey-multi-choice", "response");
    if (responses.quicksandGeneration === "False, the weather changes the terrain each day, so a tile that was quicksand one day may not be quicksand the next day." &&
        responses.quicksandPatterns === "True, some tiles are more likely to be quicksand than others." &&
        responses.quicksandAvoidance === "True, quicksand slows the agent down, so I should avoid tiles that are likely to be quicksand whenever possible." &&
        responses.safeTiles === "True, some of the tiles are never quicksand, so if a tile has quicksand you can be certain it is not safe.") {
      return false;
    } else {
      gs.session_data.comprehensionAttempts += 1;
      return true;
    }
  }
};

const planningComprehensionLoop = {
  timeline: [
    planningInstructions,
    {
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
          prompt: "True or False: Quicksand slows the agent down.",
          name: "quicksandAvoidance",
          options: [
            "True, quicksand slows the agent down, so I should avoid tiles that are likely to be quicksand whenever possible.",
            "False, quicksand doesn't affect the agent's speed, so I shouldn't consider it when planning."],
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
    }, 
    planningComprehensionFailedTrial
  ],
  loop_function: (data) => {
    const responses = data.values()[1].response;
    if (responses.quicksandGeneration === "False, the weather changes the terrain each day, so a tile that was quicksand one day may not be quicksand the next day." &&
        responses.quicksandPatterns === "True, some tiles are more likely to be quicksand than others." &&
        responses.quicksandAvoidance === "True, quicksand slows the agent down, so I should avoid tiles that are likely to be quicksand whenever possible." &&
        responses.safeTiles === "True, some of the tiles are never quicksand, so if a tile has quicksand you can be certain it is not safe.") {
      return false;
    } else {
      gs.session_data.comprehensionAttempts += 1;
      return true;
    }
  }
}

const planningPracticeIntroduction = trialTemplates.instructionsTrial([
  `<p>Great job!</p> \
  <p>Now let\'s do an example together.</p> \
  <p>On the next screen, you\'ll see a grid. Click the tiles to plan a path to the ${goal_text} while avoiding quicksand.</p>`], true); // allow_keys for now always true

const planningPractice = {
  timeline: [
    planningPracticeIntroduction,
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
  ],
  on_start: () => { gs.session_data.startPlanPracticeTS = Date.now() }
}

/* #endregion */

/* #region Simulation Trial Instructions */

const counterfactualInstructions = trialTemplates.instructionsTrial([
  `<p>Great work! After observing the agent follow your path, you will take some time to reflect on the plan you made and consider better alternatives!</p>\
  <p>Starting in the path you originally planned, you will create a path to the goal that you believe would have been more effective given what the agent encountered.</p>
  <img width="550" src="assets/instructions/figs/7c_counterfactual_plan.gif">`,

  // `<p>You are welcome to use the tiles in the original path if you think it will lead to a better outcome. However, you should not simply retrace the same path.</p>
  // <img width="450" src="assets/instructions/figs/6c2_counterfactual_retrace.png">`,

  `<p>Great job!</p> \
   <p>On the next screen, you\'ll be asked a few questions confirming that everything is crystal clear.</p>\
   <p>Please do your best. You will not be able to proceed to the next part of the experiment until you answer all questions correctly.</p>`
], true);

counterfactualInstructions.on_start = () => { gs.session_data.startSimInstructionTS = Date.now() };

const counterfactualComprehensionFailedTrial = {
  timeline: [
    {
      type: jsPsychHtmlButtonResponse,
      stimulus: [
        `<p>Unfortunately, you did not answer all the questions correctly.</p> \
        <p>Please read the instructions carefully and try again.</p>`],
      choices: ["Try Again"],
      margin_vertical: "20px"
    }
  ],
  conditional_function: () => {
    const responses = getFromLastTrial("survey-multi-choice", "response");
    if (responses.quicksandGeneration === "True, if I saw a tile as quicksand, it will also be quicksand in this reflection.") {
      return false;
    } else {
      return true;
    }
  }
};

const counterfactualComprehensionLoop = {
  timeline: [
    counterfactualInstructions,
    {
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
    },
    counterfactualComprehensionFailedTrial
  ],
  loop_function: (data) => {
    const responses = data.values()[1].response;
    if (responses.quicksandGeneration === "True, if I saw a tile as quicksand, it will also be quicksand in this reflection.") {
      return false;
    } else {
      gs.session_data.comprehensionAttempts += 1;
      return true;
    }
  }
}

const counterfactualPracticeIntroduction = trialTemplates.instructionsTrial([
  `<p>Great job!</p> \
  <p>On the next screen, you will practice doing this: making a plan, observing the outcome, and making an alternative route that you believe would have been better for the agent to follow!</p>`
], true);

const counterfactualPractice = {
  timeline: [
    counterfactualPracticeIntroduction,
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
      start_position: () => { return getFromLastTrial("quicksand-planner", "start_position") },
      goal_position: () => { return getFromLastTrial("quicksand-planner", "goal_position") },
      wall_positions: () => { return getFromLastTrial("quicksand-planner", "wall_positions") },
      dist: counterfactual_practice.dist,
      condition: () => { return gs.study_metadata.condition },
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
  ],
  on_start: () => { gs.session_data.startSimPracticeTS = Date.now() }
}

/* #endregion */

/* #region Exam Trial Instructions */

const examInstructions = () => { 
  return trialTemplates.instructionsTrial([
    `<p>Great work! There's one last thing you need to know.</p> \
    <p>After you've gained some experience helping the agent navigate the desert, we will test your understanding of which tiles are likely to be quicksand.</p>`,
    
    `<p>You will do this by marking whether you think each tile is safe or unsafe by <b>clicking</b> on it.</p> \
    <p>You can click once if you think the tile is ${gs.game_info.exam_first_click === "safe" ? "safe" : "unsafe"}, and twice if you think it's ${gs.game_info.exam_first_click === "safe" ? "unsafe" : "safe"}. Clicking more will lead the tile color to change back and forth.</p> \
    <img width="550" src="assets/instructions/figs/8_exam_${gs.game_info.exam_first_click}.gif"> \
    <p>You must click <b>all</b> tiles at least once to proceed.</p>
    `,
] , true)}; // allow_keys for now always true

/* #endregion */

const instructionsConclusion = [
  `<p>Congrats! You've learned everything you need to know about to help guide the agent to the goal.</p> \
  <p>In total, you will be guiding the agent through two deserts, for 15 days each.</p> \
  <p>Please click the continue button to get started. Thank you for participating!</p>`
]
