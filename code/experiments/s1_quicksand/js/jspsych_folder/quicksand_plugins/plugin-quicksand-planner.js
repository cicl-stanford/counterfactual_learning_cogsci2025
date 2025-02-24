var jsPsychQuicksandPlanner = (function (jspsych) {
  "use strict";

  const info = {
    name: "quicksand-planner",
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
      environment_instance: {
        type: jspsych.ParameterType.OBJECT,
        default: null,
      },
      sequence_uuid: {
        type: jspsych.ParameterType.STRING,
        default: undefined,
      },
      feedback: {
        type: jspsych.ParameterType.BOOL,
        default: false,
      }
    },
  };

  /**
   * **QUICKSAND-PLANNER**
   *
   * A jsPsych plugin for a planning (+observation) trial in the Quicksand environment
   *
   * @author {Justin Yang}
   */
  class QuicksandPlannerPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      let observe_start;
      let bonus = gs.bonus_info.trial_bonus;
      const trial_start = Date.now();
      const environment = trial.environment;
      const instance_uuid = UUID();

      environment.beginPlanningPhase(trial.start_position, trial.goal_position, trial.wall_positions, trial.environment_instance);

      const submitBtn = document.getElementById("submitBtn");
      const updateTrialPhase = () => {
        if (
          environment.gamePhase === "observation" &&
          environment.phaseComplete
        ) {
          const trial_end = Date.now();
          const path = environment.gridWorld.getAgentPath();
          const pathLocs = environment.gridWorld.getAgentPathLocs();
          const quicksand_info = environment.gridWorld.getQuicksandInfo();
          const click_events = JSON.parse(
            JSON.stringify(environment.gridWorld.clickEvents)
          );
          path.forEach((tile) => {
            bonus = Number(Math.max(0, bonus - gs.bonus_info.quicksand_penalty * tile.isQuicksand).toFixed(2));
          })

          if (trial.feedback) {
            const occluder = document.getElementById("occluder");
            const occluderText = document.getElementById("occluderText");
            occluder.style.background = gs.background_color.randColor;
  
            const percent =
              (path.filter((tile) => tile.isQuicksand).length / (path.length - 2)) * 100;
  
            const bonusText = `You earned $${bonus} for this trial.`;
            if (percent < 25) {
              occluderText.innerHTML = `<p>Great job 😊</p><p>${bonusText}</p>`;
            } else if (percent <= 40) {
              occluderText.innerHTML = `<p>Good effort 🙂</p><p>${bonusText}</p>`;
            } else {
              occluderText.innerHTML = `<p>Better luck next time 😐</p><p>${bonusText}</p>`;
            }
            occluder.style.display = "flex";  
            setTimeout(() => {
              occluder.style.display = "none";

              const trial_data = _.extend({}, _.omit(trial, "on_finish", "type"), {
                  trial_phase: "learning",
                  instance_uuid: instance_uuid,
                  observe_time_ms: trial_end - observe_start,
                  planning_time_ms: trial_end - trial_start,
                  path: pathLocs,
                  click_events: click_events,
                  quicksand_info: quicksand_info,
                  environment: environment,
                  bonus: bonus,
                }
              );
              this.jsPsych.finishTrial(trial_data);
              submitBtn.removeEventListener("click", updateTrialPhase);
            }, gs.game_info.occluder_duration);
          } else {
            const trial_data = _.extend({}, _.omit(trial, "on_finish", "type"), {
              trial_phase: "learning",
              instance_uuid: instance_uuid,
              observe_time_ms: trial_end - observe_start,
              planning_time_ms: trial_end - trial_start,
              path: pathLocs,
              click_events: click_events,
              quicksand_info: quicksand_info,
              environment: environment,
              bonus: bonus,
            });
            this.jsPsych.finishTrial(trial_data);
            submitBtn.removeEventListener("click", updateTrialPhase);
          }
        } else if (
          environment.gamePhase === "planning" &&
          environment.phaseComplete
        ) {
          observe_start = Date.now();
          environment.beginObservationPhase(false);
        }
      };
      submitBtn.addEventListener("click", updateTrialPhase);
    }
  }
  QuicksandPlannerPlugin.info = info;

  return QuicksandPlannerPlugin;
})(jsPsychModule);
