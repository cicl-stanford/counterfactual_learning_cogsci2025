/**
 * Class representing a single grid in the quicksand environment.
 * @param {number} x - The x-coordinate of the tile on the grid.
 * @param {number} y - The y-coordinate of the tile on the grid.
 * @param {number} cellSize - The size of the grid cell in pixels.
 * @param {string} state - The state of the tile (wall, start, goal, or probability of quicksand).
 * @property {number} probabilityOfQuicksand - The probability of quicksand on the tile.
 * @property {bool} isQuicksand - Whether the tile is quicksand.
 * @property {bool} isObserved - Whether the tile has been observed.
 * @property {bool} isWall - Whether the tile is a wall.
 * @property {bool} isStart - Whether the tile is the start tile.
 * @property {bool} isGoal - Whether the tile is the goal tile.
 * @property {bool} isInPath - Whether the tile is in the planned path.
 * @property {bool} isFactive - Whether the tile is factive (part of the factual path in a counterfactual).
 * @property {bool} isSimulation - Whether to treat the agent movement as actual movement or mental simulation.
 * @property {number} pathIndex - The index of the tile in the planned path.
 * @property {number} transitionFrame - The current frame of the transition animation.
 * @property {number} lastTransitionFrame - The last frame of the transition animation.
 * @function initializeState - Initializes tile properties based on the input state.
 * @function draw - Draws the tile on the canvas.
 * @function getTileProperties - Determines the fillStyle and size of the tile.
 * @function determineFillStyle - Gets default fillStyle of the tile.
 * @function pulsingPathProperties - Determines the fillStyle and size of tiles in the path animation.
 * @function transitionPathProperties - Determines the fillStyle and size of tiles being transitioned.
 * @function calculateWavePoint - Calculates the wave point for the pulsing path animation.
 * @function interpolateFactiveColor - Interpolates the color of the tile based on the path and plan path colors.
 * @function prepareTileForMovement - Prepares the tile for movement by setting the transition frame.
 * @function reset - Resets the tile based on the probability of quicksand.
 */
class Tile {
  constructor(x, y, cellSize, state) {
    this.x = x;
    this.y = y;
    this.cellSize = cellSize;
    this.initializeState(state);

    this.pathIndex = -1;
    this.transitionFrame = 0;
    this.lastTransitionFrame = 0;
  }

  initializeState(state) {
    this.isQuicksand = false;
    this.isObserved = state === "wall";
    this.isInPath = false;
    this.isWall = state === "wall";
    this.isStart = false;
    this.isGoal = false;
    this.isFactive = false;
    this.isSimulation = false;
    this.isSimGoal = false;

    if (typeof state === "number") {
      this.probabilityOfQuicksand = state;
      this.isQuicksand = Math.random() < state;
    }
  }

  draw(ctx) {
    const [fillStyle, tileSize] = this.getTileProperties();
    const wallSize = tileSize * gs.tile.size.wall;
    drawRoundedRect(
      ctx,
      this.x * this.cellSize + (this.cellSize * (1 - tileSize)) / 2,
      this.y * this.cellSize + (this.cellSize * (1 - tileSize)) / 2,
      this.cellSize * tileSize,
      this.cellSize * tileSize,
      gs.tile.corner_radius,
      convertToHSL(fillStyle)
    );
    if (this.isWall) {
      drawRoadblock(
        ctx,
        this.x * this.cellSize + (this.cellSize * (1 - wallSize)) / 2,
        this.y * this.cellSize + (this.cellSize * (1 - wallSize)) / 2,
        this.cellSize * wallSize,
        this.cellSize * wallSize,
        gs.tile.wall_line_width,
        convertToHSL(gs.tile.colors.wall)
      );
    }
  }

  getTileProperties() {
    if (!this.isSimulation && this.transitionFrame > 0 && this.transitionFrame < gs.tile.animations.transition.frames) {
      return this.transitionPathProperties();
    } else if (this.isInPath) {
      return this.pulsingPathProperties(this.isObserved);
    }
    return [this.determineFillStyle(), gs.tile.size.base];
  }

  determineFillStyle() {
    if (this.isFactive && !this.isSimGoal && !this.isWall) {
      let color = this.isStart ? gs.tile.colors.start : 
                  this.isGoal ? gs.tile.colors.goal : 
                  this.isQuicksand ? gs.tile.colors.quicksand : 
                  gs.tile.colors.sand;
      return { h: color.h, s: color.s * 0.6, l: color.l + 10 }; // TODO: think more about colors
    }
    if (this.isStart) return gs.tile.colors.start;
    if (this.isGoal) return gs.tile.colors.goal;
    if (this.fillStyle) return this.fillStyle;
    return gs.tile.colors.default;
  }

  pulsingPathProperties(observed = true) {
    let baseColor = this.isStart ? gs.tile.colors.start :
                    this.isGoal ? gs.tile.colors.goal : 
                    observed ? 
                      this.isQuicksand ? gs.tile.colors.quicksand : 
                                         gs.tile.colors.sand : 
                      this.isFactive ? this.interpolateFactiveColor() : 
                                       gs.tile.colors.plan_path;

    const wavePoint = this.calculateWavePoint();
    const saturation =
      baseColor.s + wavePoint * gs.tile.animations.pulse.saturation_range;
    const tileSize =
      gs.tile.size.path + wavePoint * gs.tile.animations.pulse.tile_range;
    return [{ h: baseColor.h, s: saturation, l: baseColor.l }, tileSize];
  }

  transitionPathProperties() {
    let hslPre;
    let hslPost;
    let tileSizePre;
    let tileSizePost;
    if (this.isObserved && !this.isGoal && !this.isStart) {
      // transition from planned path to observed path
      [hslPre, tileSizePre] = this.pulsingPathProperties(false);
      [hslPost, tileSizePost] = this.pulsingPathProperties(true);
    } else {
      // transition from base tile to planned path
      [hslPre, tileSizePre] = [this.determineFillStyle(), gs.tile.size.base];
      [hslPost, tileSizePost] = this.pulsingPathProperties(false);
    }
    const t =
      (this.transitionFrame - 1) / (gs.tile.animations.transition.frames - 1);

    const h = interpolateHue(hslPre.h, hslPost.h, t);
    const s = hslPre.s + (hslPost.s - hslPre.s) * t;
    const l = hslPre.l + (hslPost.l - hslPre.l) * t;
    const tileSize = tileSizePre + (tileSizePost - tileSizePre) * t;

    if (this.transitionFrame > this.lastTransitionFrame) {
      this.lastTransitionFrame = this.transitionFrame;
      this.transitionFrame++;
      if (t === 1) {
        this.lastTransitionFrame = this.transitionFrame;
      }
    } else {
      this.lastTransitionFrame = this.transitionFrame;
      this.transitionFrame--;
      if (t === 0) {
        this.lastTransitionFrame = this.transitionFrame;
      }
    }

    return [{ h: h, s: s, l: l }, tileSize];
  }

  calculateWavePoint() {
    const time = Date.now();
    const wavePosition =
      2 *
      Math.PI *
      (time / gs.tile.animations.pulse.period - this.pathIndex / 10);
    const separation = gs.tile.animations.pulse.separation * Math.PI;
    return (wavePosition % separation < Math.PI) * Math.sin(wavePosition);
  }

  interpolateFactiveColor() {
    const planPathColor = gs.tile.colors.plan_path;
    const currentColor = this.determineFillStyle();
    return {
      h: interpolateHue(currentColor.h, planPathColor.h, 0.5),
      s: (currentColor.s + planPathColor.s) / 2,
      l: (currentColor.l + planPathColor.l) / 2,
    };
  }

  prepareTileForMovement(reveal = true) {
    this.isObserved = reveal;
    this.transitionFrame = 1;
    this.lastTransitionFrame = 0;
  }

  reset() {
    if (!this.isWall && !this.isStart && !this.isGoal) {
      this.isQuicksand = Math.random() < this.probabilityOfQuicksand; // Reset based on probability if not special tile
    }
  }
}
