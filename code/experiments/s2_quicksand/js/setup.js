
import { setupInstructions } from '../assets/instructions/instructions.mjs';
import { createQuicksandProcedure, trialTemplates, exitSurvey, countNonTimelineTrials } from './jspsych_folder/jspsych_utils.mjs';
import { getJsPsych } from './jspsych_folder/jspsychSingleton.mjs';

async function loadStims() {
  const stimuli_file = `./assets/stimuli/${gs.study_metadata.project}-${gs.study_metadata.experiment}-${gs.study_metadata.iteration_name}-${gs.study_metadata.stimuli_index}.json`;
  const response = await fetch(stimuli_file);
  const stims = await response.json();
  if (gs.study_metadata.dev_mode) { console.log(`Stims recieved from ${stimuli_file}`, stims); }
  return stims;
}

async function setJsPsychProperties(jsPsych, stimID) {
  gs.session_data.gameID = UUID();
  gs.session_data.stimID = stimID;
  gs.game_info.exam_first_click = Math.random() < 0.5 ? "safe" : "unsafe";
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
  const jsPsych = await getJsPsych();

  // Discretely recieve condition from URL parameter, e.g.,
  // http://localhost:8000/s2_quicksand/?condition=0
  // http://localhost:8000/s2_quicksand/?condition=1
  gs.study_metadata.condition = (() => {
    let conditionCode = parseInt(jsPsych.data.getURLVariable('condition'));
    if (conditionCode === 0) return "observation";
    if (conditionCode === 1) return "counterfactual";
    console.error("Invalid condition code", conditionCode);
    return null
  })();
  if (gs.study_metadata.dev_mode) { console.log("Condition:", gs.study_metadata.condition); }

  await setJsPsychProperties(jsPsych, stims.id);
  const instructions = setupInstructions(true); // previously, gs.study_metadata.dev_mode was passed in for allow_keys
  const quicksandProcedure = createQuicksandProcedure(stims, gs.study_metadata.condition);

  let trials = [];
  trials.push(instructions.landingInstructions);
  if (!gs.study_metadata.dev_mode) {
    trials.push(instructions.consent);
    trials.push(trialTemplates.fullScreenTrial());
    trials.push(trialTemplates.browserCheckTrial());  
  }
  trials.push(instructions.preload);
  trials.push(instructions.planningComprehensionLoop);
  trials.push(instructions.planningPractice);
  if (gs.study_metadata.condition !== "observation") {
    trials.push(instructions.counterfactualComprehensionLoop);
    trials.push(instructions.counterfactualPractice);
  }
  trials.push(instructions.examInstructions());
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
