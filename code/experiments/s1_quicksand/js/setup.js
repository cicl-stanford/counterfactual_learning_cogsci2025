
import { setupInstructions } from '../assets/instructions/instructions.mjs';
import { createQuicksandProcedure, trialTemplates, exitSurvey, countNonTimelineTrials } from './jspsych_folder/jspsych_utils.mjs';
import { getJsPsych } from './jspsych_folder/jspsychSingleton.mjs';

async function loadStims() {
  const response = await fetch(`./assets/stimuli/${gs.study_metadata.project}-${gs.study_metadata.experiment}-${gs.study_metadata.iteration_name}-${gs.study_metadata.stimuli_index}.json`);
  const stims = await response.json();
  return stims;
}

async function setJsPsychProperties(jsPsych, stimID) {
  gs.session_data.gameID = UUID();
  gs.session_data.stimID = stimID;
  const urlParams = jsPsych.data.urlVariables();
  jsPsych.data.addProperties(
    {
      prolificID: urlParams.PROLIFIC_PID,
      studyID: urlParams.STUDY_ID,
      sessionID: urlParams.SESSION_ID,
      gameID: gs.session_data.gameID,
      stimID: gs.session_data.stimID,
      condition: gs.study_metadata.condition,
      iteration: gs.study_metadata.iteration_name,
      dev_mode: gs.study_metadata.dev_mode,
      project: gs.study_metadata.project,
      experiment: gs.study_metadata.experiment,
    }
  );
}

async function setupGame() {
  const stims = await loadStims();
  const jsPsych = await getJsPsych(); // Wait for jsPsych initialization and condition fetching
  await setJsPsychProperties(jsPsych, stims.id);
  const instructions = setupInstructions(true); // gs.study_metadata.dev_mode
  const quicksandProcedure = createQuicksandProcedure(stims, gs.study_metadata.condition);

  let trials = [];
  trials.push(instructions.landingInstructions);
  if (!gs.study_metadata.dev_mode) {
    trials.push(instructions.consent);
    trials.push(trialTemplates.fullScreenTrial());
    trials.push(trialTemplates.browserCheckTrial());  
  }
  trials.push(instructions.preload);
  trials.push(instructions.planningInstructions);
  trials.push(instructions.planningPractice);
  trials.push(instructions.planningPracticeConclusion);
  trials.push(instructions.planningComprehensionLoop);
  if (gs.study_metadata.condition !== "observation") {
    trials.push(instructions.simulationInstructions);
    trials.push(instructions.simulationPractice);
    trials.push(instructions.simulationPracticeConclusion);
    trials.push(instructions.simulationComprehensionLoop);
  }
  // // TODO: make these conditional depending on if we have an evaluation phase or not from stims)
  // // For now, we just comment it out.
  // // trials.push(instructions.examInstructions);
  // // trials.push(instructions.navigationInstructions);

  trials.push(instructions.instructionsConclusion);
  trials.push(quicksandProcedure);
  trials.push(trialTemplates.preSurveyMessage());
  trials.push(exitSurvey);
  trials.push(trialTemplates.dataPipeTrial());
  trials.push(trialTemplates.goodbyeTrial('https://app.prolific.com/submissions/complete?cc=C6KQO5F3'));

  gs.game_info.num_trials = countNonTimelineTrials(trials);

  jsPsych.run(trials);
}

export { setupGame };
