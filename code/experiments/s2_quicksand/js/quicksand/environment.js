/**
 * Class handling interactions between Agent, Gridworld, and participant UI.
 * @param {object} spec - The specification object that generates a GridWorld.
 * @property {object} canvas - The canvas element for the GridWorld.
 * @property {object} ctx - The 2D context of the canvas.
 * @property {object} submitBtn - The button that submits the path.
 * @property {object} gridWorld - The GridWorld object.
 * @property {object} agent - The Agent object.
 * @property {string} gamePhase - The current phase of the task.
 * @property {Array} movementQueue - A queue for movements the agent takes.
 * @function initGameElements - Initializes the game elements.
 * @function startGameLoop - Starts the game loop.
 * @function loop - The main animation loop.
 * @function updateEnvironment - Updates the environment.
 * @function updateAgent - Updates the agent.
 * @function updateUI - Updates the UI - particularly the submit button.
 * @function handleSubmission - Handles the submission button click.
 * @function beginSimulation - Begins the simulation phase of the task.
 * @function resetForSimulation - Resets the environment for the simulation phase.
 * @function resetEnvironment - Resets the environment to a new trial.
 * @function initializeStartTile - Initializes the start tile of the environment.
 * @function prepareMovementQueue - Converts plan to movement queue for the agent.
 */
class Environment {
  constructor(spec, canvas) {
    this.initGameElements(spec, canvas);
    this.startGameLoop();
  }

  initGameElements(spec, canvas) {
    this.spec = spec;
    this.canvas = canvas;
    this.ctx = this.canvas.getContext("2d");
    this.submitBtn = document.getElementById("submitBtn");
    this.remainingStepsText = document.getElementById("remainingSteps");
    this.instructionsText = document.getElementById("instructionText");
    this.frameDuration = 1000 / gs.game_info.fps;
    this.lastFrameTime = window.performance.now();

    this.gridWorld = new GridWorld(this.spec, this.canvas);
    this.agent = null;

    this.gamePhase = null;
    this.phaseComplete = false;
    this.movementQueue = [];
  }

  setupAgent(start_position, color) {
    this.agent = new Agent(
      start_position.x,
      start_position.y,
      this.gridWorld.cellSize,
      color
    );
  }

  startGameLoop() {
    this.animationFrameId = requestAnimationFrame(() => this.loop());
  }

  loop() {
    if (document.hasFocus() || gs.study_metadata.dev_mode) {
      const now = window.performance.now();
      const timeElapsed = now - this.lastFrameTime;
      if (timeElapsed > this.frameDuration) {
        this.lastFrameTime = now - (timeElapsed % this.frameDuration);
        this.updateEnvironment();
        this.updateUI();
      }
    }
    this.animationFrameId = requestAnimationFrame(() => this.loop());
  }

  updateEnvironment() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.gridWorld.draw(this.ctx);
    if (this.agent) {
      this.updateAgent();
    }
  }

  updateAgent() {
    if (this.movementQueue.length > 0) {
      const moveComplete = this.agent.moveAcrossTile(
        this.movementQueue[0].tile,
        this.movementQueue[0].direction,
        this.movementQueue[0].reveal
      );
      if (moveComplete) {
        this.movementQueue.shift();
        if (this.movementQueue.length > 1) {
          this.movementQueue[1].tile.prepareTileForMovement(
            this.movementQueue[1].reveal
          );
        } else if (this.movementQueue.length === 0) {
          setTimeout(() => {
            this.phaseComplete = true;
          }, gs.game_info.observation_delay);
        }
      }
    }
    this.agent.draw(this.ctx);
  }

  updateUI() {
    if (this.gamePhase === "planning") {
      let lastTile = this.gridWorld.lastTileInPath;
      this.phaseComplete = lastTile && lastTile.isGoal;
      this.submitBtn.disabled = !this.phaseComplete;
      this.submitBtn.textContent = "Submit Path";
    } else if (this.gamePhase === "observation") {
      this.submitBtn.disabled = !this.phaseComplete;
      this.submitBtn.textContent = "Continue";
    } else if (
      this.gamePhase === "counterfactual" ||
      this.gamePhase === "hypothetical"
    ) {
      let lastTile = this.gridWorld.lastTileInPath;
      this.phaseComplete = lastTile && lastTile.isSimGoal;
      this.submitBtn.disabled = !this.phaseComplete;
      this.submitBtn.textContent = "Submit Path";
      this.submitBtn.textContent = "Continue";
    } else if (this.gamePhase === "submit") {
      this.submitBtn.disabled = !this.phaseComplete;
      this.submitBtn.textContent = "Submit!";
    }
    if (gs.game_info.limit_steps) {
      this.remainingStepsText.textContent = `Remaining Steps: ${this.gridWorld.remainingSteps}`;
    }
  }

  beginPlanningPhase(startPosition, goalPosition, wallPositions, instance) {
    this.gamePhase = "planning";
    this.instructionsText.textContent = "Plan the quickest path to the goal.";
    this.setupAgent(startPosition, gs.agent.colors.default);
    this.resetForPlanning(instance);
    this.initializeStartTile(startPosition);
    this.initializeGoalTile(goalPosition);
    this.initializeWallTiles(wallPositions);
    this.gridWorld.addEventListeners();
    this.gridWorld.remainingSteps = gs.game_info.total_steps;
    this.phaseComplete = false;
  }

  beginObservationPhase(simulation = false) {
    this.gamePhase = "observation";
    this.instructionsText.textContent = "Watch the agent follow your planned path!";
    this.gridWorld.removeEventListeners();
    setTimeout(() => { this.prepareMovementQueue(simulation); }, 1000);
    this.phaseComplete = false;
  }

  beginSimulationPhase(startPosition, goalPosition, wallPositions, condition, exam = false) {
    this.gamePhase = condition;
    if (condition === "hypothetical" || exam) {
      this.instructionsText.textContent = `How would you get to the goal from here?`;
    } else if (condition === "counterfactual") {
      this.instructionsText.textContent = `Given the path you took, how would you best reach the goal from here?`;
    }
    this.setupAgent(startPosition, exam ? gs.agent.colors.default : gs.agent.colors.simulation);
    this.resetForSimulation(condition);
    this.initializeStartTile(startPosition);
    this.initializeGoalTile(goalPosition, true);
    this.initializeWallTiles(wallPositions);
    this.gridWorld.addEventListeners();
    this.phaseComplete = false;
  }

  resetForSimulation(condition) {
    this.gridWorld.tiles.flat().forEach((tile) => {
      // if in simulation and either the start or goal tile, do this
      if (
        tile.isSimulation &&
        (tile.pathIndex === 0 ||
          this.gridWorld.getAgentPath().splice(-1)[0].pathIndex === tile.pathIndex)
      ) {
        tile.isObserved = false;
        tile.isStart = false;
        tile.isGoal = false;
      }

      if (condition === "counterfactual") {
        tile.isFactive = tile.isObserved || tile.isFactive;
        // TODO: eventually handle multiple counterfactuals per observation (i.e., reset counterfactual start location only, but not observed one). Tabled to later. 
      } else if (condition === "hypothetical") {
        tile.isFactive = false;
        tile.isStart = false;
        tile.isGoal = false;
      }
      tile.isInPath = false;
      tile.isObserved = false;
      tile.isSimulation = false;
      tile.isSimGoal = false;
      tile.isWall = false;
      tile.pathIndex = -1;
      tile.transitionFrame = 0;
      tile.lastTransitionFrame = 0;
    });
    this.gridWorld.resetClickEvents();
  }

  resetForPlanning(instance) {
    this.gridWorld.tiles.flat().forEach((tile) => {
      tile.isInPath = false;
      tile.isObserved = false;
      tile.isStart = false;
      tile.isGoal = false;
      tile.isWall = false;
      tile.isFactive = false;
      tile.isSimulation = false;
      tile.isSimGoal = false;
      tile.pathIndex = -1;
      tile.transitionFrame = 0;
      tile.lastTransitionFrame = 0;
    });
    if (!instance) {
      this.gridWorld.rollQuicksand();
    } else {
      this.gridWorld.setQuicksand(instance);
    }
    this.gridWorld.resetClickEvents();
  }

  initializeStartTile(startPosition) {
    const startTile = this.gridWorld.tiles[startPosition.y][startPosition.x];
    this.gridWorld.lastTileInPath = startTile;
    startTile.isStart = true;
    startTile.isInPath = true;
    startTile.isObserved = true;
    startTile.isQuicksand = false;
    startTile.pathIndex = 0;
    startTile.transitionFrame = 1;
  }

  initializeGoalTile(goalPosition, simulation = false) {
    const goalTile = this.gridWorld.tiles[goalPosition.y][goalPosition.x];
    goalTile.isGoal = true;
    goalTile.isSimGoal = simulation;
    goalTile.isInPath = false;
    goalTile.isObserved = true;
    goalTile.isQuicksand = false;
  }

  initializeWallTiles(wallPositions) {
    if (wallPositions.length === 0) {
      return;
    }
    wallPositions.forEach((wallPosition) => {
      const wallTile = this.gridWorld.tiles[wallPosition.y][wallPosition.x];
      wallTile.isWall = true;
      wallTile.isInPath = false;
      wallTile.isObserved = true;
      wallTile.isQuicksand = false;
    });
  }

  prepareMovementQueue(simulation = false) {
    const pathTiles = this.gridWorld.getAgentPath();
    pathTiles.forEach((currentTile, index) => {
      currentTile.isSimulation = simulation;
      if (index < pathTiles.length - 1) {
        const nextTile = pathTiles[index + 1];
        this.movementQueue.push({
          tile: currentTile,
          direction: [nextTile.x - currentTile.x, nextTile.y - currentTile.y],
          reveal: !simulation,
        });
      }
    });
    this.movementQueue[1].tile.prepareTileForMovement(!simulation);
  }

  destroy() {
    cancelAnimationFrame(this.animationFrameId);
    this.gridWorld.removeEventListeners();
    this.gridWorld.destroy();
    this.gridWorld = null;
    this.agent = null;
    this.canvas = null;
    this.ctx = null;
    this.submitBtn = null;
    this.remainingStepsText = null;
  }
}
