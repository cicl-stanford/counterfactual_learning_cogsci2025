let jsPsych = null;

export async function getJsPsych() {
  if (!jsPsych) {
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