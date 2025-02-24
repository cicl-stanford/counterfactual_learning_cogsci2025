var jsPsychQuicksandSimulate = (function (jspsych) {
  "use strict";

  const info = {
    name: "quicksand-simulate",
    parameters: {
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
      condition: {
        type: jspsych.ParameterType.STRING,
        default: undefined, // 'hypothetical' or 'counterfactual'
      },
      environment: {
        type: jspsych.ParameterType.OBJECT,
        default: undefined, // environment object from planner trial
      },
      sequence_uuid: {
        type: jspsych.ParameterType.STRING,
        default: undefined,
      },
      instance_uuid: {
        type: jspsych.ParameterType.STRING,
        default: undefined,
      },
      dist: {
        type: jspsych.ParameterType.NUMBER,
        default: undefined,
      },
    },
  };

  /**
   * **QUICKSAND-SIMULATE**
   *
   * A jsPsych plugin for a hypothetical or counterfactual simulation trial in the Quicksand environment
   *
   * @author {Justin Yang}
   */
  class QuicksandSimulatePlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      // let observe_start;
      const trial_start = Date.now();
      const environment = trial.environment;
      if (trial.dist && gs.game_info.limit_steps) {
        environment.gridWorld.remainingSteps = trial.dist + 2;
      } else {
        environment.gridWorld.remainingSteps = gs.game_info.total_steps;
      }

      environment.beginSimulationPhase(trial.start_position, trial.goal_position, trial.wall_positions, trial.condition);

      const submitBtn = document.getElementById("submitBtn");
      const updateTrialPhase = () => {
        if (environment.gamePhase === trial.condition && environment.phaseComplete) {
          const trial_end = Date.now();
          const path = environment.gridWorld.getAgentPathLocs();
          const click_events = JSON.parse(JSON.stringify(environment.gridWorld.clickEvents));
          const trial_data = _.extend({}, _.omit(trial, "on_finish", "type"), {
            trial_phase: "learning",
            planning_time_ms: trial_end - trial_start,
            // observe_time_ms: trial_end - observe_start,
            path: path,
            click_events: click_events,
            environment: environment,
          });
          this.jsPsych.finishTrial(trial_data);
          submitBtn.removeEventListener("click", updateTrialPhase);  
        }
      }
      submitBtn.addEventListener("click", updateTrialPhase);
    }
  }
  QuicksandSimulatePlugin.info = info;

  return QuicksandSimulatePlugin;
})(jsPsychModule);

/**
 * In case we still want to watch the agent follow the simulated path
{
  if (
    environment.gamePhase === "observation" &&
    environment.phaseComplete
  ) {
    const trial_end = Date.now();
    const path = environment.gridWorld.getAgentPathLocs();
    const click_events = JSON.parse(
      JSON.stringify(environment.gridWorld.clickEvents)
    );
    const trial_data = _.extend({}, _.omit(trial, "on_finish", "type"), {
      trial_phase: "learning",
      planning_time_ms: trial_end - trial_start,
      observe_time_ms: trial_end - observe_start,
      path: path,
      click_events: click_events,
      environment: environment,
    });
    this.jsPsych.finishTrial(trial_data);
    submitBtn.removeEventListener("click", updateTrialPhase);
  } else if (
    environment.gamePhase === trial.condition &&
    environment.phaseComplete
  ) {
    observe_start = Date.now();
    environment.beginObservationPhase(true);
  }
};
 */
