gs = {
  study_metadata: {
    project: "counterfactual_learning",
    experiment: "s2_quicksand",
    datapipe_experiment_id: "jbEe0AXceqHt",
    iteration_name: "friendsandfamily2",
    stimuli_index: 0,
    dev_mode: true,
    condition: undefined, // observation, counterfactual
  },
  session_data: {
    gameID: undefined,
    stimID: undefined,
    startPlanInstructionTS: undefined,
    startPlanPracticeTS: undefined,
    startSimInstructionTS: undefined,
    startSimPracticeTS: undefined,
    startExperimentTS: undefined,
    endExperimentTS: undefined,
    comprehensionAttempts: 0,
  },
  game_info: {
    fps: 60,
    total_steps: 1000,
    observation_delay: 350,
    occluder_duration: 1500,
    limit_steps: false,
    num_trials: null,
    exam_first_click: null, // safe/unsafe; randomly selected on init
  },
  bonus_info: {
    trial_bonus: 0.04,
    quicksand_penalty: 0.01,
  },
  agent: {
    names: {
      default: "Paul", // no longer relevant
    },
    colors: {
      default: { h: 4, s: 85, l: 40 },
      simulation: { h: 4, s: 85, l: 40 },//{ h: 190, s: 85, l: 40 },
    },
    size: {
      radius: 0.33,
      eye_radius: 0.3,
      pupil_radius: 0.5,
      smile_radius: 0.33,
    },
    animations: {
      travel_frames: 10, // number of frames to travel 1 tile
      quicksand_multiplier: 6, // how much quicksand slows you down
      simulation_multiplier: 1.5,
    },
  },
  tile: {
    colors: {
      default: { h: 45, s: 35, l: 63 },
      sand: { h: 30, s: 70, l: 75 },
      quicksand: { h: 30, s: 70, l: 35 },
      plan_path: { h: 24, s: 35, l: 63 },
      factive_path: { h: 24, s: 35, l: 50 },
      start: { h: 30, s: 75, l: 50 },
      goal: { h: 120, s: 75, l: 25 },
      wall: { h: 358, s: 65, l: 41 },
      safe: { h: 20, s: 28, l: 80 },
      unsafe: { h: 20, s: 28, l: 30 },
    },
    size: {
      base: 0.94,
      path: 0.85,
      real: 0.68,
      wall: 0.8,
    },
    animations: {
      transition: {
        frames: 10, // # of frames to transition tile
        delay: 30, // # of milisseconds to offset each tiles' transition start
      },
      pulse: {
        period: 1000, // speed of one pulse
        separation: 6, // separation between pulses, in half-sine waves
        saturation_range: 30, // maximum saturation increase
        tile_range: 0.05, // maximum tile size increase
      },
    },
    corner_radius: 8,
    wall_line_width: 5,
  },
  background_color: {
    default: { h: 40, s: 80, l: 90 },
    hue_range: 12,
    saturation_range: 10,
    lightness_range: 2,
  }
};
