var jsPsychQuicksandDestroy = (function (jspsych) {
  "use strict";

  const info = {
    name: "quicksand-destroy",
    parameters: {
      environment: {
        type: jspsych.ParameterType.OBJECT,
        default: undefined,
      },
      sequence_uuid: {
        type: jspsych.ParameterType.STRING,
        default: null,
      },
    },
  };

  /**
   * **QUICKSAND-DESTROY**
   *
   * A jsPsych plugin for removing the quicksand object.
   *
   * @author {Justin Yang}
   */
  class QuicksandDestroyPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const trial_end = Date.now();
      trial.environment.destroy();
      trial.environment = null;
      display_element.innerHTML = "";
      this.jsPsych.finishTrial({
        sequence_end: trial_end,
        sequence_uuid: trial.sequence_uuid,
      });
    }
  }
  QuicksandDestroyPlugin.info = info;

  return QuicksandDestroyPlugin;
})(jsPsychModule);
