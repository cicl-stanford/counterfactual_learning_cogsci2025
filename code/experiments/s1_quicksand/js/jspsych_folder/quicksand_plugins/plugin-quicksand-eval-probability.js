var jsPsychQuicksandEvalProbability = (function (jspsych) {
  "use strict";

  const info = {
    name: "quicksand-eval-probability",
    parameters: {
      spec1: {
        type: jspsych.ParameterType.OBJECT,
        default: undefined,
      },
      spec2: {
        type: jspsych.ParameterType.OBJECT,
        default: undefined,
      },
    },
  };

  /**
   * **QUICKSAND-EVAL-PROBABILITY**
   *
   * A jsPsych plugin for evaluating understanding of quicksand probabilities
   *
   * @author (Justin Yang)
   */
  class QuicksandEvalProbabilityPlugin {
    constructor(jsPsych) {
      this.jsPsych = jsPsych;
    }
    trial(display_element, trial) {
      const trial_start = Date.now();

      let html = `
                <div id="quicksandContainer" style="display: flex; flex-direction: column; align-items: center;">
                    <div style="margin-bottom: 10px; text-align: center;">
                        <p style="font-size:22px">Please evaluate the likelihood of the two deserts below.</br> Which one do you think is more likely to have been the one you were in?</p>
                    </div>
                    <div id="gridworlds" style="display: flex; justify-content: space-around; width: 100%; margin-bottom: 20px;">
                        <div id="gridworld1" class="gridworld-container">
                            <canvas id="gridworldCanvas1" width="450" height="450" style="margin: 0 50px;"></canvas>
                        </div>
                        <div id="gridworld2" class="gridworld-container">
                            <canvas id="gridworldCanvas2" width="450" height="450" style="margin: 0 50px;"></canvas>
                        </div>
                    </div>
                    <div id="sliderContainer" style="width: 80%; text-align: center; margin-bottom: 20px;">
                        <div id="evalSlider" class="evalSlider" style="width: 100%; margin-bottom: 10px;"></div>
                        <div style="display: flex; justify-content: space-between;">
                        <span>Most likely desert 1</span>
                        <span>Equally likely</span>
                        <span>Most likely desert 2</span>
                    </div>
                    </div>
                    <button id="submitBtn" class="jspsych-btn" style="margin-top: 20px;" disabled>Submit</button>
                </div>
            `;
      display_element.innerHTML = html;

      // Initialize gridworlds
      const canvas1 = document.getElementById("gridworldCanvas1");
      const canvas2 = document.getElementById("gridworldCanvas2");
      const ctx1 = canvas1.getContext("2d");
      const ctx2 = canvas2.getContext("2d");
      const gridworld1 = new GridWorld(trial.spec1, canvas1);
      const gridworld2 = new GridWorld(trial.spec2, canvas2);

      // Interpolate colors based on probabilities
      gridworld1.tiles.flat().forEach((tile) => {
        if (tile.probabilityOfQuicksand !== undefined) {
          tile.fillStyle = interpolateColor(
            gs.tile.colors.sand,
            gs.tile.colors.quicksand,
            tile.probabilityOfQuicksand
          );
        }
      });
      gridworld2.tiles.flat().forEach((tile) => {
        if (tile.probabilityOfQuicksand !== undefined) {
          tile.fillStyle = interpolateColor(
            gs.tile.colors.sand,
            gs.tile.colors.quicksand,
            tile.probabilityOfQuicksand
          );
        }
      });

      // Draw gridworlds
      gridworld1.draw(ctx1);
      gridworld2.draw(ctx2);

      // Initialize slider
      $("#evalSlider")
        .slider({
          min: 0,
          max: 100,
          create: function () {
            $(this).find(".ui-slider-handle").hide();
          },
        })
        .slider("pips", {
          first: "pip",
          last: "pip",
          step: 100,
        });

      // Handle slider interaction
      const submitBtn = document.getElementById("submitBtn");
      $("#evalSlider").on("slidestart", function () {
        $(this).find(".ui-slider-handle").show();
        $("#submitBtn").prop("disabled", false);
      });

      // Handle submit button click
      submitBtn.addEventListener("click", () => {
        const trial_end = Date.now();
        const response = $("#evalSlider").slider("option", "value");
        const trial_data = {
          trial_phase: "evaluation",
          eval_time_ms: trial_end - trial_start,
          slider_value: response,
          quicksand_info_left: () => {
            return gridworld1.getQuicksandInfo();
          },
          quicksand_info_right: () => {
            return gridworld2.getQuicksandInfo();
          },
        };
        this.jsPsych.finishTrial(trial_data);
      });
    }
  }
  QuicksandEvalProbabilityPlugin.info = info;

  return QuicksandEvalProbabilityPlugin;
})(jsPsychModule);
