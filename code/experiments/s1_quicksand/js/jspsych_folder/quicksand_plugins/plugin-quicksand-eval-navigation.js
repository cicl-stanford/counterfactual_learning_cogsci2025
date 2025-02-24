var jsPsychQuicksandEvalNavigation = (function (jspsych) {
  "use strict";

  const info = {
    name: "quicksand-eval-navigation",
    parameters: {
      environment: {
        type: jspsych.ParameterType.OBJECT,
        default: undefined,
      },
      start_position: {
        type: jspsych.ParameterType.OBJECT,
        default: undefined,
      },
      goal_position: {
        type: jspsych.ParameterType.OBJECT,
        default: undefined,
      },
      wall_positions: {
        type: jspsych.ParameterType.ARRAY,
        default: [],
      },
    },
  };

  /**
   * **QUICKSAND-EVAL-NAVIGATION**
   *
   * A jsPsych plugin for evaluating quicksand navigation performance
   *
   * @author (Justin Yang)
   */
  class QuicksandEvalNavigationPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const occluder = document.getElementById("occluder");
      const occluderText = document.getElementById("occluderText");
      occluderText.innerHTML = `How would you get to the goal from here?`;
      occluder.style.display = "flex";
      occluder.style.background = gs.background_color.randColor;
      this.jsPsych.pluginAPI.setTimeout(() => { occluder.style.display = "none" }, gs.game_info.occluder_duration);

      const trial_start = Date.now();
      const environment = trial.environment;
      environment.gridWorld.remainingSteps = gs.game_info.total_steps;

      environment.beginSimulationPhase(trial.start_position, trial.goal_position, trial.wall_positions, "hypothetical", true);

      const submitBtn = document.getElementById("submitBtn");
      const updateTrialPhase = () => {
        if (environment.gamePhase === "hypothetical" && environment.phaseComplete) {
          const trial_end = Date.now();
          const path = environment.gridWorld.getAgentPathLocs();
          const click_events = JSON.parse(JSON.stringify(environment.gridWorld.clickEvents));
          const trial_data = _.extend({}, _.omit(trial, "on_finish", "type"), {
            trial_phase: "evaluation",
            planning_time_ms: trial_end - trial_start,
            path: path,
            click_events: click_events,
            environment: environment,
          });
          this.jsPsych.finishTrial(trial_data);
          submitBtn.removeEventListener("click", updateTrialPhase);
        }
      };
      submitBtn.addEventListener("click", updateTrialPhase);
    }
  }
  QuicksandEvalNavigationPlugin.info = info;

  return QuicksandEvalNavigationPlugin;
})(jsPsychModule);
