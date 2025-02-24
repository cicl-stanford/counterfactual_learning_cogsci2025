let jsPsych = null;

export async function getJsPsych() {
  if (!jsPsych) {
    const condition = await jsPsychPipe.getCondition(gs.study_metadata.datapipe_experiment_id);
    gs.study_metadata.condition = (function() {
      if (condition === 0) return "observation";
      if (condition === 1) return "counterfactual";
      if (condition === 2) return "hypothetical";
      console.error("Invalid condition code");
    })();
    if (gs.study_metadata.dev_mode) { console.log("Condition fetched", gs.study_metadata.condition); }

    jsPsych = initJsPsych({
      show_progress_bar: true,
      auto_update_progress_bar: false,
      on_trial_finish: (data) => {
        if (gs.study_metadata.dev_mode) { console.log("trial finished", data); }
        jsPsych.setProgressBar((data.trial_index + 1) / gs.game_info.num_trials)
      },
    });
  }
  return jsPsych;
}