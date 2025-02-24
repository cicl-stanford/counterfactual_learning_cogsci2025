/**
 * Class representing the Gridworld in the quicksand environment.
 * @param {object} spec - The specification object that generates the GridWorld.
 * @property {number} width - The width of the grid.
 * @property {number} height - The height of the grid.
 * @property {number} cellSize - The size of the grid cell in pixels.
 * @property {Array} tiles - The tiles in the grid.
 * @property {object} lastTileInPath - The last tile in the planned path.
 * @property {object} canvas - The canvas element for the GridWorld.
 * @property {function} handleCanvasClickBound - The bound handleCanvasClick function.
 * @function addEventListeners - Adds event listeners to the grid canvas.
 * @function removeEventListeners - Removes event listeners from the grid canvas.
 * @function initializeTiles - Initializes the tiles in the grid.
 * @function draw - Draws the grid on the canvas.
 * @function rollQuicksand - Resets the GridWorld with new quicksand tiles.
 * @function handleCanvasClick - Handles the click event on the canvas.
 * @function isAdjacentToLastTile - Checks if the clicked tile is adjacent to the last tile in the path.
 * @function removePathFromTile - Removes the path from the clicked tile and all subsequent tiles.
 * @function getGridCoordinates - Gets the grid coordinates of the clicked tile.
 * @function getAgentPath - Gets the path of the agent.
 */
class GridWorld {
  constructor(spec, canvas, exam_first_click = null) {
    this.width = spec.metadata.width;
    this.height = spec.metadata.height;
    this.canvas = canvas;
    this.ctx = this.canvas.getContext("2d");
    this.cellSize = this.canvas.width / this.width;
    this.remainingSteps = gs.game_info.total_steps;

    this.tiles = [];
    this.lastTileInPath = null;
    this.initializeTiles(spec.states);
    this.exam_first_click = exam_first_click;
    this.handleCanvasClickBound = this.exam_first_click === null ? this.handleCanvasClick.bind(this) : this.handleCanvasClickEval.bind(this);

    this.clickEvents = [];
  }

  addEventListeners() {
    this.canvas.addEventListener("click", this.handleCanvasClickBound);
  }

  removeEventListeners() {
    this.canvas.removeEventListener("click", this.handleCanvasClickBound);
  }

  initializeTiles(states) {
    for (let y = 0; y < this.height; y++) {
      this.tiles[y] = [];
      for (let x = 0; x < this.width; x++) {
        this.tiles[y][x] = new Tile(x, y, this.cellSize, states[`(${x},${y})`]);
        if (this.exam_first_click !== null) {
          this.tiles[y][x].numClicks = 0;
        }
      }
    }
  }

  draw(ctx) {
    this.tiles.forEach((row) => row.forEach((tile) => tile.draw(ctx)));
  }

  rollQuicksand() {
    this.tiles.forEach((row) => row.forEach((tile) => tile.reset()));
  }

  setQuicksand(quicksandInstance) {
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        this.tiles[y][x].isQuicksand = quicksandInstance[`(${x},${y})`];
      }
    }
  }

  handleCanvasClick(event) {
    const { x, y } = this.getGridCoordinates(event);
    const clickedTile = this.tiles[y][x];

    if (clickedTile.isInPath && clickedTile !== this.lastTileInPath) {
      this.clickEvents.push({ x, y, timestamp: Date.now() });
      const removedSteps = this.removePathFromTile(clickedTile);
      this.remainingSteps += removedSteps;
    } else if (this.lastTileInPath.isSimGoal || this.lastTileInPath.isGoal) {
      // don't need this mess as our goal is always the same in simulation and planning now
      //(this.lastTileInPath.isGoal && !this.lastTileInPath.isGoal)) {
      // only push if goal tile is not last element in clickEvents
      // const lastClickEvent = this.clickEvents[this.clickEvents.length - 1];
      // if (lastClickEvent.x !== x || lastClickEvent.y !== y) {
      //   this.clickEvents.push({ x, y, timestamp: Date.now() });
      // }
    } else if (this.isAdjacentToLastTile(clickedTile) && !clickedTile.isWall && this.remainingSteps > 0) {
      this.clickEvents.push({ x, y, timestamp: Date.now() });
      clickedTile.isInPath = true;
      clickedTile.pathIndex = this.lastTileInPath.pathIndex + 1;
      clickedTile.transitionFrame = 1;
      this.lastTileInPath = clickedTile;
      this.remainingSteps--;
    }
  }

  handleCanvasClickEval(event) {
    const colorCycle = [gs.tile.colors.safe, gs.tile.colors.unsafe]
    const { x, y } = this.getGridCoordinates(event);
    const clickedTile = this.tiles[y][x];

    clickedTile.numClicks += 1;
    const colorIndex = ((this.exam_first_click === 'safe' ? 1 : 0) + clickedTile.numClicks) % colorCycle.length;
    clickedTile.fillStyle = colorCycle[colorIndex];
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.draw(this.ctx);
    this.clickEvents.push({ x, y, timestamp: Date.now(), clicks: clickedTile.numClicks, rating: ["safe", "unsafe"][colorIndex] });

    // check if numClicks >= 1 for all tiles, and enable document.getElementById("submitBtn"); if so
    const allClicked = this.tiles.every(row => row.every(tile => tile.numClicks >= 1));
    if (allClicked) {
      document.getElementById("submitBtn").disabled = false;
    }
  }

  isAdjacentToLastTile(tile) {
    const last = this.lastTileInPath;
    return Math.abs(last.x - tile.x) + Math.abs(last.y - tile.y) === 1;
  }

  removePathFromTile(tile) {
    const index = tile.pathIndex + 1;
    const removedTiles = this.tiles.flat().filter((t) => t.pathIndex >= index);
    removedTiles.forEach((t) => {
      t.isInPath = false;
      t.pathIndex = -1;
      t.transitionFrame = gs.tile.animations.transition.frames - 1;
    });
    this.lastTileInPath = tile;
    return removedTiles.length;
  }

  getGridCoordinates(event) {
    const rect = event.target.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / this.cellSize);
    const y = Math.floor((event.clientY - rect.top) / this.cellSize);
    return { x, y };
  }

  getAgentPath() {
    return this.tiles
      .flat()
      .filter((tile) => tile.isInPath)
      .sort((a, b) => a.pathIndex - b.pathIndex);
  }

  getAgentPathLocs() {
    return this.getAgentPath().map((tile) => [tile.x, tile.y]);
  }

  resetClickEvents() {
    this.clickEvents = [];
  }

  getQuicksandInfo() {
    const colorCycle = [gs.tile.colors.safe, gs.tile.colors.unsafe]

    return this.tiles.flat().reduce((info, tile) => {
      let val = {
        tile_type: tile.isWall ? "wall" : tile.isStart ? "start" : tile.isGoal ? "goal" : "normal",
        is_quicksand: tile.isQuicksand,
        prob_quicksand: tile.probabilityOfQuicksand,
      };
      if (this.exam_first_click !== null) {        
        let colorIndex = ((this.exam_first_click === 'safe' ? 1 : 0) + tile.numClicks) % colorCycle.length;
        val.clicks = tile.numClicks;
        val.rating = ["safe", "unsafe"][colorIndex];
      }
      info[`(${tile.x},${tile.y})`] = val;
      return info;
    }, {});
  }

  destroy() {
    // nullify references to DOM elements or large objects
    this.tiles = null;
    this.canvas = null;
    this.handleCanvasClickBound = null;
    this.lastTileInPath = null;
  }
}
