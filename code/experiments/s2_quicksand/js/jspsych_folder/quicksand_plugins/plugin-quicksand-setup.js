var jsPsychQuicksandSetup = (function (jspsych) {
  "use strict";

  const info = {
    name: "quicksand-setup",
    parameters: {
      spec: {
        type: jspsych.ParameterType.OBJECT,
        default: undefined,
      },
    },
  };

  /**
   * **QUICKSAND-SETUP**
   *
   * A jsPsych plugin for setting up the Quicksand environment object.
   *
   * @author {Justin Yang}
   */
  class QuicksandSetupPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const trial_start = Date.now();
      const sequence_uuid = UUID();

      const { width, height } = trial.spec.metadata;
      const maxCells = Math.max(width, height);
      const maxViewportWidth = window.innerWidth * 0.8;
      const maxViewportHeight = window.innerHeight * 0.6;
      const maxCanvasSize = Math.min(maxViewportWidth, maxViewportHeight);

      const cellSize = Math.min(Math.floor(maxCanvasSize / maxCells), 80);
      const canvasWidth = width * cellSize;
      const canvasHeight = height * cellSize;
      const instructionWidth = canvasWidth * 1.25;
      const remainingStepsDiv = gs.game_info.limit_steps
        ? `<div id="remainingSteps">Remaining Steps: ${gs.game_info.total_steps}</div>`
        : "";

      let html = `
                <div id="quicksandContainer">
                    <div id="instructionText" style="width: ${instructionWidth}px;">
                        Plan a path for the agent to reach the goal in as little time as possible.
                    </div>
                    <img src="assets/palm_tree.svg" alt="Palm Tree" style="display: block; margin: 0 auto; height: 65px;" />
                    <div style="width: ${canvasWidth}px; height: ${canvasHeight}px; position: relative; margin: 0 auto;">
                        <canvas id="quicksandCanvas" width="${canvasWidth}" height="${canvasHeight}" style="display: block;"></canvas>
                    </div>
                    <div id="tileEnumerator" style="display: flex; justify-content: space-between; width: ${canvasWidth}px;"></div>
                    ${remainingStepsDiv}
                    <button disabled id="submitBtn">Submit</button>
                </div>
                <div id="occluder">
                    <p id="occluderText"></p>
                </div>
            `;
      display_element.innerHTML = html;

      const canvas = document.getElementById("quicksandCanvas");
      const environment = new Environment(trial.spec, canvas);

      const tileEnumerator = document.getElementById("tileEnumerator");

      // Create the numbers 1 to 8 and append them to the tileEnumerator div
      for (let i = 1; i <= 8; i++) {
        const numberSpan = document.createElement("span");
        numberSpan.textContent = i;
        tileEnumerator.appendChild(numberSpan);
      }
                  
      this.jsPsych.finishTrial({
        environment: environment,
        sequence_start: trial_start,
        sequence_uuid: sequence_uuid,
        spec: trial.spec,
      });
    }
  }
  QuicksandSetupPlugin.info = info;

  return QuicksandSetupPlugin;
})(jsPsychModule);
