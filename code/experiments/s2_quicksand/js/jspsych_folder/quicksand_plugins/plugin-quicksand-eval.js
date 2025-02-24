var jsPsychQuicksandEval = (function (jspsych) {
  "use strict";

  const info = {
    name: "quicksand-eval",
    parameters: {
      spec: {
        type: jspsych.ParameterType.OBJECT,
        default: undefined,
      },
      exam_first_click: {
        type: jspsych.ParameterType.STRING,
        default: undefined,
      },
      sequence_uuid: {
        type: jspsych.ParameterType.STRING,
        default: null,
      },
    },
  };

  /**
   * **QUICKSAND-EVAL**
   *
   * A jsPsych plugin for evaluating understanding of quicksand safety
   *
   * @author (Justin Yang)
   */
  class QuicksandEvalPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const trial_start = Date.now();

      const { width, height } = trial.spec.metadata;
      const maxCells = Math.max(width, height);
      const maxViewportWidth = window.innerWidth * 0.8;
      const maxViewportHeight = window.innerHeight * 0.6;
      const maxCanvasSize = Math.min(maxViewportWidth, maxViewportHeight);

      const cellSize = Math.min(Math.floor(maxCanvasSize / maxCells), 80);
      const canvasWidth = width * cellSize;
      const canvasHeight = height * cellSize;
      const instructionWidth = canvasWidth * 1.25;

      let html = `
        <div id="quicksandContainer">
            <div id="instructionText" style="width: ${instructionWidth}px;">
                How safe is each tile?</br> Click once for ${trial.exam_first_click === "safe" ? "safe" : "unsafe"}, and twice for ${trial.exam_first_click === "safe" ? "unsafe" : "safe"}.
            </div>
            <img src="assets/palm_tree.svg" alt="Palm Tree" style="display: block; margin: 0 auto; height: 65px;" />
            <div style="width: ${canvasWidth}px; height: ${canvasHeight}px; position: relative; margin: 0 auto;">
                <canvas id="quicksandCanvas" width="${canvasWidth}" height="${canvasHeight}" style="display: block;"></canvas>
            </div>
            <div id="tileEnumerator" style="display: flex; justify-content: space-between; width: ${canvasWidth}px;"></div>
            <div id="legendContainer" style="margin: 10px 0 0; display: flex; justify-content: center; align-items: center; gap: 20px;">
              <canvas id="legendCanvas" width="${cellSize * 4}" height="${cellSize}" style="display: block;"></canvas>
            </div>
            <div id="legendLabels" style="display: flex; justify-content: space-between; width: ${cellSize * 4}px;     height: 20px; position: relative; font-size: 16px; font-weight: bold;"></div>
            <button disabled id="submitBtn">Submit</button>
        </div>`;
      display_element.innerHTML = html;

      const canvas = document.getElementById("quicksandCanvas");
      const ctx = canvas.getContext("2d");
      let gridworld = new GridWorld(trial.spec, canvas, trial.exam_first_click);
      gridworld.draw(ctx);
      gridworld.addEventListeners();
      
      // tile numbers setup
      const tileEnumerator = document.getElementById("tileEnumerator");
      for (let i = 1; i <= 8; i++) {
        const numberSpan = document.createElement("span");
        numberSpan.textContent = i;
        tileEnumerator.appendChild(numberSpan);
      }

      // Legend Setup
      const legendCanvas = document.getElementById("legendCanvas");
      const legendCtx = legendCanvas.getContext("2d");
      const legendColors = trial.exam_first_click === "safe" ? [ gs.tile.colors.safe, gs.tile.colors.unsafe ] : [ gs.tile.colors.unsafe, gs.tile.colors.safe ];
      const legendLabels = trial.exam_first_click === "safe" ? ["safe", "unsafe"] : ["unsafe", "safe"];
      const legendLabelsContainer = document.getElementById("legendLabels");

      const legendTilePositions = [];
      legendColors.forEach((color, index) => {
        const x = (index * 2 + 0.5)
        legendTilePositions.push(x * cellSize);
        const tile = new Tile(x, 0, cellSize, null);
        tile.fillStyle = color;
        tile.draw(legendCtx);
      });

      legendLabelsContainer.innerHTML = "";
      legendTilePositions.forEach((x, index) => {
        const label = document.createElement("span");
        label.textContent = legendLabels[index];
            label.style.position = "absolute";
            label.style.left = `${x}px`;
            label.style.width = `${cellSize}px`;
            label.style.textAlign = "center";
            legendLabelsContainer.appendChild(label);
          });

      // end trial
      const submitBtn = document.getElementById("submitBtn");
      submitBtn.addEventListener("click", () => {
        const trial_end = Date.now();
        const quicksand_info = gridworld.getQuicksandInfo();
        const click_events = JSON.parse(JSON.stringify(gridworld.clickEvents));
        const trial_data = _.extend({}, _.omit(trial, "on_finish", "type", "on_start"), {
          trial_phase: "evaluation",
          eval_time_ms: trial_end - trial_start,
          quicksand_info: quicksand_info,
          click_events: click_events,
        });

        // Destroy gridworld
        gridworld.removeEventListeners();
        gridworld.destroy();
        gridworld = null;
        display_element.innerHTML = "";

        this.jsPsych.finishTrial(trial_data);
      });
    }
  }
  QuicksandEvalPlugin.info = info;

  return QuicksandEvalPlugin;
})(jsPsychModule);
